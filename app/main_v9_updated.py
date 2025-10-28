from sentence_transformers import CrossEncoder
import os
import json
import subprocess
import requests
from pathlib import Path
from fastapi import FastAPI, Query, UploadFile, File
from fastapi.responses import JSONResponse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil

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
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
SIMILARITY_THRESHOLD = 0.3
EMBED_BATCH_SIZE = 500

# ---------------- Directories ----------------
os.makedirs(CHAT_DIR, exist_ok=True)
os.makedirs(PDF_CONVERTED_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# ---------------- Embeddings + VectorDB ----------------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
vectordb = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)

# ---------------- Cross Encoder ----------------
try:
    reranker = CrossEncoder(CROSS_ENCODER_MODEL)
except Exception as e:
    print(f"[WARN] Could not load cross-encoder: {e}")
    reranker = None

# ---------------- Chat history ----------------
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
    subprocess.run([
        "libreoffice", "--headless", "--convert-to", "pdf", "--outdir", PDF_CONVERTED_DIR, docx_path
    ], check=True)
    return pdf_path

# ---------------- Prompt builder ----------------
def build_prompt(question, context=None, chat_history=None, references=None):
    prompt = (
        "You are a knowledgeable assistant. "
        "Use the provided context to answer accurately. "
        "If answer not found, reply 'I don't know.'\n\n"
    )
    if chat_history:
        prompt += "Chat history (latest 5 turns):\n"
        for turn in chat_history[-5:]:
            prompt += f"Q: {turn['question']}\nA: {turn['answer']}\n"
    if context:
        prompt += f"\nContext:\n{context}\n"
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
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)

# ---------------- Vectorstore helper ----------------
def add_chunks_to_vectorstore(chunks, path):
    total = len(chunks)
    print(f"[EMBED] Embedding {total} chunks from {path}")
    vectordb.add_documents(chunks)
    print(f"[EMBED] Completed embeddings for {path}")

# ---------------- Upload document endpoint ----------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        os.makedirs(DOCS_DIR, exist_ok=True)
        file_path = os.path.join(DOCS_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        print(f"[UPLOAD] File saved to {file_path}")

        docs = load_documents([file_path])
        chunks = split_documents(docs)
        add_chunks_to_vectorstore(chunks, file_path)

        metadata = load_metadata()
        metadata[file_path] = get_file_metadata(file_path)
        save_metadata(metadata)

        return {"message": f"File '{file.filename}' uploaded and embedded successfully."}
    except Exception as e:
        print(f"[UPLOAD ERROR] {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------- Reset app endpoint ----------------
@app.post("/reset")
def reset_app():
    try:
        # Delete documents, vectorstore, metadata
        for d in [DOCS_DIR, VECTORSTORE_DIR, PDF_CONVERTED_DIR]:
            if os.path.exists(d):
                shutil.rmtree(d)
                os.makedirs(d, exist_ok=True)
        for f in [METADATA_FILE, "processed_pdfs.json"]:
            if os.path.exists(f):
                os.remove(f)

        # Recreate empty vectorstore
        global vectordb
        vectordb = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)

        return {"message": "App reset successfully. All data cleared."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------- List chat sessions endpoint ----------------
@app.get("/list_chat_sessions")
def list_chat_sessions():
    sessions = [Path(f).stem for f in Path(CHAT_DIR).glob("*.json")]
    return {"sessions": sessions}

# ---------------- Hybrid retrieval ----------------
def hybrid_retrieve_and_rerank(query, top_k=FINAL_TOP_K):
    retrieved_docs = vectordb.similarity_search_with_score(query, k=TOP_K)
    if not retrieved_docs:
        return [], []

    if reranker:
        pairs = [(query, doc.page_content) for doc, _ in retrieved_docs]
        scores = reranker.predict(pairs)
        ranked_docs = [doc for doc, score in sorted(zip([d for d, _ in retrieved_docs], scores), key=lambda x: x[1], reverse=True)]
    else:
        ranked_docs = [doc for doc, _ in sorted(retrieved_docs, key=lambda x: x[1], reverse=True)]

    matched_chunks, references = [], []
    for doc in ranked_docs[:top_k]:
        matched_chunks.append(doc.page_content.strip())
        meta = doc.metadata or {}
        src = os.path.basename(meta.get("source", "Unknown"))
        page = meta.get("page", "N/A")
        ref = f"{src}({page})" if page not in ("N/A", "", None) else src
        references.append(ref)
    return matched_chunks, references

# ---------------- Chat endpoint ----------------
@app.get("/chat")
def chat(q: str = Query(...), session_id: str = Query(...)):
    history = load_chat_history(session_id)
    matched_chunks, references = hybrid_retrieve_and_rerank(q)
    context = "\n\n".join(matched_chunks) if matched_chunks else None
    prompt = build_prompt(q, context, chat_history=history, references=references)

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

    history.append({"question": q, "answer": answer})
    save_chat_history(session_id, history)

    return {
        "session_id": session_id,
        "question": q,
        "matched_chunks": matched_chunks,
        "references": references,
        "response": answer
    }

# ---------------- Startup ingestion ----------------
@app.on_event("startup")
def startup_event():
    print("[INIT] Loading existing documents from DOCS_DIR...")
    all_files = [str(p) for p in Path(DOCS_DIR).rglob("*") if p.suffix.lower() in (".txt", ".pdf", ".docx")]
    if all_files:
        for path in all_files:
            docs = load_documents([path])
            chunks = split_documents(docs)
            add_chunks_to_vectorstore(chunks, path)
    print("[INIT] Initialization complete.")
