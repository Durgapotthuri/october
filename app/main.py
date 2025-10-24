from fastapi.responses import JSONResponse
from fastapi import HTTPException
from fastapi import FastAPI, Query, UploadFile, File
from sentence_transformers import CrossEncoder
import os
import json
import shutil
import subprocess
import requests
import threading
import time
from pathlib import Path
from fastapi import FastAPI, Query
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

app = FastAPI()
db_write_lock = threading.Lock()

# ---------------- Configuration ----------------
DOCS_DIR = "/home/autolab/durga/RAG_documents"
CHAT_DIR = "/home/autolab/durga/october/uV8_watchdog_prompt_UI/chat_sessions"
VECTORSTORE_DIR = "/home/autolab/durga/october/uV8_watchdog_prompt_UI/vectorstore"
EMBEDDING_MODEL_PATH = "/home/autolab/durga/models/all-MiniLM-L6-v2"
METADATA_FILE = "/home/autolab/durga/october/uV8_watchdog_prompt_UI/processed_files.json"
PDF_CONVERTED_DIR = "/home/autolab/durga/october/uV8_watchdog_prompt_UI/converted_pdfs"
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
TOP_K = 20
SIMILARITY_THRESHOLD = 0.3
EMBED_BATCH_SIZE = 500
MAX_CHROMA_BATCH_SIZE = 5461

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#----------------------------------Load Session---------------------

@app.get("/get_history/{session_id}")
def get_history(session_id: str):
    path = os.path.join(CHAT_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Session not found")
 
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
 
    # Convert your chat format (Q/A pairs) into frontend-friendly format
    messages = []
    for item in data:
        messages.append({"role": "user", "content": item["question"]})
        messages.append({"role": "bot", "content": item["answer"]})
 
    return {"session_id": session_id, "messages": messages}

#------------------------list of sessions------------------------
@app.get("/list_sessions")
def list_sessions():
    try:
        sessions = [
            f.replace(".json", "")
            for f in os.listdir(CHAT_DIR)
            if f.endswith(".json")
        ]
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


os.makedirs(CHAT_DIR, exist_ok=True)
os.makedirs(PDF_CONVERTED_DIR, exist_ok=True)

# ----------------------Resetting--------------------------

@app.post("/reset")
def reset_app_state():
    global vectordb
    try:
        # 1. Delete all files in DOCS_DIR
        if os.path.exists(DOCS_DIR):
            for f in os.listdir(DOCS_DIR):
                fp = os.path.join(DOCS_DIR, f)
                if os.path.isfile(fp):
                    os.remove(fp)

        # 2. Delete converted PDFs
        if os.path.exists(PDF_CONVERTED_DIR):
            for f in os.listdir(PDF_CONVERTED_DIR):
                fp = os.path.join(PDF_CONVERTED_DIR, f)
                if os.path.isfile(fp):
                    os.remove(fp)

        # 3. Delete VECTORSTORE_DIR
        if os.path.exists(VECTORSTORE_DIR):
            shutil.rmtree(VECTORSTORE_DIR, ignore_errors=True)

        # 4. Recreate VECTORSTORE_DIR
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)
        os.chmod(VECTORSTORE_DIR, 0o777)
        subprocess.run(["chown", "-R", os.getenv("USER"), VECTORSTORE_DIR])

        # 5. Remove leftover lock files
        for root, dirs, files in os.walk(VECTORSTORE_DIR):
            for file in files:
                if "LOCK" in file or file.endswith((".wal", ".shm")):
                    os.remove(os.path.join(root, file))

        # 6. Apply chmod 666 to all files
        for root, dirs, files in os.walk(VECTORSTORE_DIR):
            for file in files:
                os.chmod(os.path.join(root, file), 0o666)

        # 7. Check if VECTORSTORE_DIR is writable
        if not os.access(VECTORSTORE_DIR, os.W_OK):
            raise HTTPException(status_code=500, detail="VECTORSTORE_DIR is not writable (read-only mount?)")

        # 8. Remove metadata file
        if os.path.exists(METADATA_FILE):
            os.remove(METADATA_FILE)

        # 9. Initialize Chroma
        time.sleep(1.0)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
        vectordb = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)

        # 10. Apply chmod 666 to chroma.sqlite3 and related files
        for root, dirs, files in os.walk(VECTORSTORE_DIR):
            for file in files:
                if file.startswith("chroma.sqlite3"):
                    os.chmod(os.path.join(root, file), 0o666)

        # 11. Finalize permissions and start watcher
        ensure_permissions()
        start_watcher()

        print("[RESET] Application state has been reset successfully.")
        return {"status": "success", "message": "Application state has been reset."}

    except Exception as e:
        print(f"[RESET ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")
    except Exception as e:
        print(f"[RESET ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

# ---------------- Embeddings + VectorDB ----------------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
vectordb = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)

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


#--------------------------uploading file through choose file-------------------
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    with db_write_lock:
        try:
            file_path = os.path.join(DOCS_DIR, file.filename)

            # Check and log DOCS_DIR permissions
            if not os.path.exists(DOCS_DIR):
                os.makedirs(DOCS_DIR, exist_ok=True)
                print(f"[UPLOAD] Created DOCS_DIR: {DOCS_DIR}")

            permissions = oct(os.stat(DOCS_DIR).st_mode)[-3:]
            print(f"[UPLOAD] Docs directory permissions: {permissions}")

            # Save the uploaded file
            with open(file_path, "wb") as f:
                f.write(await file.read())
            print(f"[UPLOAD] File saved to {file_path}")

            # Wait for file to be fully written
            for _ in range(10):  # up to 5 seconds
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    break
                time.sleep(0.5)
            else:
                raise HTTPException(status_code=400, detail="File is empty or failed to upload properly.")

            # Log vectorstore directory permissions
            if os.path.exists(VECTORSTORE_DIR):
                permissions = oct(os.stat(VECTORSTORE_DIR).st_mode)[-3:]
                subprocess.run(["chmod", "-R", "777", VECTORSTORE_DIR])
                print(f"[UPLOAD] Vectorstore directory permissions: {permissions}")
            else:
                print(f"[UPLOAD] Vectorstore directory does not exist: {VECTORSTORE_DIR}")

            # Index the uploaded file
            update_vectorstore_for_file(file_path, "created")

            return JSONResponse(content={"message": f"{file.filename} uploaded and indexed successfully."})

        except Exception as e:
            print(f"[UPLOAD ERROR] {str(e)}")
            return JSONResponse(status_code=500, content={"error": f"Upload failed: {str(e)}"})
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
    with db_write_lock:
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
    with db_write_lock:
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
    def wait_until_file_ready(self, path, retries=5, delay=1):
        for _ in range(retries):
            if os.path.exists(path) and os.path.getsize(path) > 0:
                try:
                    with open(path, "rb") as f:
                        f.read(1)  # Try reading a byte
                    return True
                except Exception:
                    pass
            time.sleep(delay)
        return False

    def on_created(self, event):
        if not event.is_directory:
            if self.wait_until_file_ready(event.src_path):
                update_vectorstore_for_file(event.src_path, "created")
            else:
                print(f"[WATCHDOG WARNING] File not ready or unreadable: {event.src_path}")

    def on_modified(self, event):
        if not event.is_directory:
            if self.wait_until_file_ready(event.src_path):
                update_vectorstore_for_file(event.src_path, "modified")
            else:
                print(f"[WATCHDOG WARNING] File not ready or unreadable: {event.src_path}")

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


def ensure_permissions():
    try:
        os.makedirs(DOCS_DIR, exist_ok=True)
        os.chmod(DOCS_DIR, 0o777)
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)
        os.chmod(VECTORSTORE_DIR, 0o777)

        for root, dirs, files in os.walk(VECTORSTORE_DIR):
            for file in files:
                if file.endswith(".db") or ".sqlite3" in file:
                    os.chmod(os.path.join(root, file), 0o666)

        if os.path.exists(METADATA_FILE):
            os.chmod(METADATA_FILE, 0o664)

        print(f"[PERMISSIONS] DOCS_DIR: {oct(os.stat(DOCS_DIR).st_mode)[-3:]}")
        print(f"[PERMISSIONS] VECTORSTORE_DIR: {oct(os.stat(VECTORSTORE_DIR).st_mode)[-3:]}")

        print("[PERMISSIONS] All paths set to writable.")
    except Exception as e:
        print(f"[PERMISSIONS ERROR] {e}")


# ---------------- Startup ingestion ----------------
@app.on_event("startup")
def startup_event():
    ensure_permissions()
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
    seen = set()

    for doc in ranked_docs:
        content = doc.page_content.strip()
        if content not in seen:
            matched_chunks.append(content)
            seen.add(content)

            metadata = doc.metadata or {}
            source = os.path.basename(metadata.get("source", "Unknown"))
            page = metadata.get("page", "N/A")

            if page not in (None, "N/A", "", "null"):
                references.append(f"{source}({page})")
            else:
                references.append(f"{source}")

        if len(matched_chunks) >= top_k:
            break

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








