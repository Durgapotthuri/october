
from sentence_transformers import CrossEncoder
import os
import json
import subprocess
import requests
import threading
from pathlib import Path
from fastapi import FastAPI, Query
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

app = FastAPI()

# ---------------- Configuration ----------------
DOCS_DIR = "/home/autolab/durga/RAG_documents"
CHAT_DIR = "chat_sessions"
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL_PATH = "/home/autolab/durga/models/all-MiniLM-L6-v2"
METADATA_FILE = "processed_files.json"
PDF_CONVERTED_DIR = "converted_pdfs"


CROSS_ENCODER_MODEL = "/home/autolab/durga/models/ms-marco-MiniLM-L6-v2"
FINAL_TOP_K = 5

# Cross-encoder for reranking
try:
    reranker = CrossEncoder(CROSS_ENCODER_MODEL)
except Exception as e:
    print(f"[WARN] Could not load cross-encoder: {e}")
    reranker = None

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
SIMILARITY_THRESHOLD = 0.3
EMBED_BATCH_SIZE = 500
MAX_CHROMA_BATCH_SIZE = 5461

os.makedirs(CHAT_DIR, exist_ok=True)
os.makedirs(PDF_CONVERTED_DIR, exist_ok=True)

# ---------------- Embeddings + VectorDB ----------------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
vectordb = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)
vectorstore_lock = threading.Lock()

# ---------------- Chat history handling ----------------
def get_chat_path(session_id):
    return os.path.join(CHAT_DIR, f"{session_id}.json")

def load_chat_history(session_id):
    path = get_chat_path(session_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_chat_history(session_id, history):
    with open(get_chat_path(session_id), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

# ---------------- DOCX → PDF conversion ----------------
def convert_docx_to_pdf(docx_path):
    pdf_path = os.path.join(PDF_CONVERTED_DIR, Path(docx_path).stem + ".pdf")
    if os.path.exists(pdf_path):
        if os.path.getmtime(pdf_path) >= os.path.getmtime(docx_path):
            return pdf_path
    subprocess.run([
        "libreoffice", "--headless", "--convert-to", "pdf", "--outdir", PDF_CONVERTED_DIR, docx_path
    ], check=True)
    if os.path.exists(pdf_path):
        return pdf_path
    else:
        raise ValueError(f"Conversion failed for {docx_path}")

# ---------------- Prompt builder ----------------
def build_prompt(question, context=None, chat_history=None, references=None):
    prompt = (
        "You are a knowledgeable assistant. "
        "Use the provided context to answer the question as accurately and thoroughly as possible. "
        "If the answer is not explicitly stated, analyze the context deeply and infer a helpful response. "
        "Do not guess if the context is clearly irrelevant, but try to synthesize and summarize when possible.\n\n"
    )
    if chat_history:
        prompt += "Chat history (latest 5 turns):\n"
        for turn in chat_history[-5:]:
            prompt += f"Q: {turn['question']}\nA: {turn['answer']}\n"
    if context:
        prompt += f"\nContext:\n{context}\n"
    else:
        prompt += "\nNo relevant context available.\n"
    if references:
        prompt += f"\nReferences:\n" + "\n".join(references) + "\n"
    prompt += f"\nCurrent question: {question}\n"
    return prompt

# ---------------- Metadata ----------------
def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_metadata(metadata):
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

def get_file_metadata(file_path):
    return {"path": str(file_path), "modified": os.path.getmtime(file_path)}

# ---------------- Document loading ----------------
def load_documents(file_paths):
    documents = []
    for path in file_paths:
        ext = Path(path).suffix.lower()
        try:
            if ext == ".txt":
                docs = TextLoader(path).load()
            elif ext == ".pdf":
                docs = PyPDFLoader(path).load()
            elif ext == ".docx":
                pdf_path = convert_docx_to_pdf(path)
                docs = PyPDFLoader(pdf_path).load()
            else:
                continue
            for doc in docs:
                doc.metadata["source"] = path
                if "page" not in doc.metadata:
                    doc.metadata["page"] = "N/A"
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] Loading {path} failed: {e}")
    return documents

# ---------------- Split documents ----------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = "Unknown"
        if "page" not in chunk.metadata:
            chunk.metadata["page"] = "N/A"
    return chunks

# ---------------- Vectorstore helper ----------------
def add_chunks_to_vectorstore(chunks, path):
    with vectorstore_lock:
        total = len(chunks)
        print(f"[EMBED] Embedding {total} chunks from {path}")
        for i in range(0, total, EMBED_BATCH_SIZE):
            batch = chunks[i:i + EMBED_BATCH_SIZE]
            if len(batch) > MAX_CHROMA_BATCH_SIZE:
                batch = batch[:MAX_CHROMA_BATCH_SIZE]
            vectordb.add_documents(batch)
            percent = ((i + len(batch)) / total) * 100
            print(f"[EMBED] {i + len(batch)}/{total} chunks processed ({percent:.2f}%)")
        print(f"[EMBED] Completed embeddings for {path}")

# ---------------- Vectorstore update ----------------
def update_vectorstore_for_file(path, event_type):
    metadata = load_metadata()
    old_meta = metadata.get(path)
    mtime = os.path.getmtime(path) if os.path.exists(path) else None

    if event_type in ("created", "modified"):
        if old_meta and old_meta.get("modified") == mtime:
            return
        new_docs = load_documents([path])
        if new_docs:
            new_chunks = split_documents(new_docs)
            if new_chunks:
                add_chunks_to_vectorstore(new_chunks, path)
        metadata[path] = get_file_metadata(path)
        print(f"[UPDATE] {event_type.title()} {path}")
    elif event_type == "deleted":
        with vectorstore_lock:
            vectordb.delete(where={"source": path})
        metadata.pop(path, None)
        print(f"[DELETE] {path} removed from vectorstore")

    save_metadata(metadata)

# ---------------- Watchdog ----------------
class DocsEventHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            update_vectorstore_for_file(event.src_path, "created")
    def on_modified(self, event):
        if not event.is_directory:
            update_vectorstore_for_file(event.src_path, "modified")
    def on_deleted(self, event):
        if not event.is_directory:
            update_vectorstore_for_file(event.src_path, "deleted")

def start_watcher():
    observer = Observer()
    handler = DocsEventHandler()
    observer.schedule(handler, DOCS_DIR, recursive=True)
    observer.daemon = True
    observer.start()
    print("[WATCHDOG] Monitoring documents folder")

# ---------------- Startup ingestion ----------------
@app.on_event("startup")
def startup_event():
    print("[INIT] Ingesting existing documents...")
    all_files = [str(p) for p in Path(DOCS_DIR).rglob("*") if p.suffix.lower() in (".txt", ".pdf", ".docx")]
    for path in all_files:
        update_vectorstore_for_file(path, "created")
    print("[INIT] Initial ingestion complete.")
    start_watcher()

# ---------------- Chat Processor ----------------
def hybrid_retrieve_and_rerank(query, top_k=FINAL_TOP_K):
    retrieved_docs = vectordb.similarity_search_with_score(query, k=TOP_K)
    if not retrieved_docs:
        return [], []
 
    # Rerank with ms-marco-MiniLM-L6-v2
    if reranker:
        pairs = [(query, doc.page_content) for doc, _ in retrieved_docs]
        scores = reranker.predict(pairs)
        ranked_docs = [
            doc for doc, score in sorted(
                zip([d for d, _ in retrieved_docs], scores),
                key=lambda x: x[1],
                reverse=True
            )
        ]
    else:
        ranked_docs = [
            doc for doc, _ in sorted(retrieved_docs, key=lambda x: x[1], reverse=True)
        ]
 
    matched_chunks, references = [], []
    for doc in ranked_docs[:top_k]:
        matched_chunks.append(doc.page_content.strip())
 
        metadata = doc.metadata or {}
        source = os.path.basename(metadata.get("source", "Unknown"))
        page = metadata.get("page", "N/A")
 
        # Format as stm32.pdf(64)
        if page not in (None, "N/A", "", "null"):
            references.append(f"{source}({page})")
        else:
            references.append(f"{source}")
 
    return matched_chunks, references

def process_chat(session_id: str, question: str):
    history = load_chat_history(session_id)
																	  
    matched_chunks, references = hybrid_retrieve_and_rerank(question)
    context = "\n\n".join(matched_chunks) if matched_chunks else None
    prompt = build_prompt(question, context, chat_history=history, references=references)

    payload = {
            "model": "mistral:latest",
        "prompt": prompt,
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 40,
        "repeat_penalty": 1,
        "stream": False
    }

    try:
        res = requests.post("http://10.207.20.55:11434/api/generate", json=payload)
        answer = res.json().get("response", res.text)
    except Exception as e:
        answer = f"Error: {str(e)}"

    history.append({"question": question, "answer": answer})
    save_chat_history(session_id, history)

    return {
        "session_id": session_id,
        "question": question,
        "matched_chunks": matched_chunks,
        "references": references,
        "response": answer
    }

# ---------------- FastAPI endpoint ----------------
@app.get("/chat")											
def chat(q: str = Query(...), session_id: str = Query(...)):
    return process_chat(session_id, q)





