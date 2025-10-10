import os
import json
import subprocess
import aiohttp
import re
import time
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from dotenv import load_dotenv

# ===============================
# טעינת משתני סביבה
# ===============================
load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")
SUMMARY_FILE = "data/index_summary.json"
ZAPIER_WEBHOOK_URL = "https://hooks.zapier.com/hooks/catch/5499574/u5u0yfy/"

# ===============================
# הגדרת אפליקציית FastAPI
# ===============================
app = FastAPI(title="AI Shabaton – ללא GPT")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# בדיקה שאינדקס קיים
# ===============================
def ensure_index_exists():
    """ודא שהאינדקס קיים וקריא."""
    print(f"📁 Using Chroma dir: {CHROMA_DIR}")
    print(f"📄 Looking for summary file: {SUMMARY_FILE}")

    if os.path.exists(SUMMARY_FILE):
        try:
            with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
                json.load(f)
            print("✅ Index summary found — skipping rebuild.")
            return
        except Exception as e:
            print(f"⚠️ Failed to read index summary: {e}")
    else:
        print("⚠️ No index summary found. You might need to rerun indexer.py manually.")


ensure_index_exists()

# ===============================
# פונקציה לקיצוץ טקסט חכם
# ===============================
def clean_and_trim_text(text: str, max_length: int = 400) -> str:
    """מסיר רווחים וקוטע בסוף משפט."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_length:
        trimmed = text[:max_length]
        end = max(trimmed.rfind("."), trimmed.rfind("!"), trimmed.rfind("?"))
        if end > 100:
            text = trimmed[:end + 1]
        else:
            text = trimmed + "..."
    return text

# ===============================
# API – /status
# ===============================
@app.get("/status")
def get_status():
    """בודק את מצב האינדקס."""
    status = {
        "index_dir_exists": os.path.exists(CHROMA_DIR),
        "index_summary_exists": os.path.exists(SUMMARY_FILE),
        "files_in_index_dir": [],
        "indexed_pages": None,
        "total_chunks": None,
        "chroma_collection_docs": None,
        "errors": [],
    }

    if os.path.exists(CHROMA_DIR):
        try:
            files = os.listdir(CHROMA_DIR)
            status["files_in_index_dir"] = files
            print(f"📂 Files in index dir: {files}")
        except Exception as e:
            status["errors"].append(f"Error reading index dir: {e}")

    if os.path.exists(SUMMARY_FILE):
        try:
            with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
                summary = json.load(f)
            status["indexed_pages"] = len(summary.get("files", []))
            status["total_chunks"] = summary.get("total_chunks", 0)
        except Exception as e:
            status["errors"].append(f"Error reading summary: {e}")

    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection("shabaton_faq")
        count = collection.count()
        status["chroma_collection_docs"] = count
        print(f"📊 Chroma collection docs: {count}")
    except Exception as e:
        status["errors"].append(f"Error accessing ChromaDB: {e}")

    return status

# ===============================
# API – /query
# ===============================
@app.post("/query")
async def query(request: Request):
    """מענה לשאלות מהאינדקס בלבד (ללא GPT)."""
    data = await request.json()
    question = data.get("query", "").strip()
    if not question:
        return {"answer": "לא התקבלה שאלה.", "sources": []}

    print(f"🧠 Query received: {question}")
    sources = []
    answer_text = ""

    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection("shabaton_faq")
        count = collection.count()
        print(f"📦 Docs in collection: {count}")

        if count == 0:
            print("⚠️ No documents found in collection.")
            return {
                "answer": "האינדקס עדיין נבנה או ריק. אנא נסו שוב מאוחר יותר.",
                "sources": [],
            }

        results = collection.query(query_texts=[question], n_results=3)
        print(f"🔍 Raw Chroma results: {results}")

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        if not docs:
            print("⚠️ No relevant results found.")
            answer_text = "לא נמצא מידע רלוונטי במאגר."
        else:
            combined = []
            for i, d in enumerate(docs):
                url = metas[i].get("url", "לא ידוע")
                snippet = clean_and_trim_text(d)
                combined.append(f"🔹 מקור: {url}\n{snippet}")
                sources.append(url)
            answer_text = "\n\n".join(combined)

    except Exception as e:
        print(f"❌ Error querying Chroma: {e}")
        answer_text = f"אירעה שגיאה בגישה למידע: {e}"

    # שליחת לוגים ל־Zapier (לא חובה)
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(
                ZAPIER_WEBHOOK_URL,
                json={
                    "timestamp": datetime.utcnow().isoformat(),
                    "question": question,
                    "answer": answer_text,
                    "sources": sources,
                    "page": request.headers.get("Referer", "Unknown"),
                },
            )
    except Exception as e:
        print(f"⚠️ Failed to send to Zapier: {e}")

    return {"answer": answer_text, "sources": sources}
