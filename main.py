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
# התחברות ל-ChromaDB
# ===============================
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection("shabaton_faq")


# ===============================
# אינדוקס ראשוני עם resume
# ===============================
def ensure_index_exists():
    """
    ודא שהאינדקס קיים, או המשך אינדוקס במקטעים אם הוא לא הושלם.
    אם האינדוקס הופסק באמצע, הקובץ indexer.py ירוץ שוב עד שכל הדפים יאונדקסו.
    """
    max_runtime_minutes = 60  # עד שעה להרצה רציפה
    start = time.time()

    # רוץ רק אם אין אינדקס תקין
    if os.path.exists(SUMMARY_FILE):
        print("✅ Index summary found – skipping initial indexing.")
        return

    while time.time() - start < max_runtime_minutes * 60:
        print("🔄 Running incremental indexing batch...", flush=True)
        result = subprocess.run(["python", "indexer.py"], capture_output=True, text=True)

        print(result.stdout)
        if result.stderr.strip():
            print(result.stderr)

        # אם מופיעה הודעה שסיים הכול, עצור
        if "✅" in result.stdout or "completed" in result.stdout.lower() or "done" in result.stdout.lower():
            print("✅ Indexing fully completed.")
            break

        # אם נוצר קובץ תקין — עצור
        if os.path.exists(SUMMARY_FILE):
            print("✅ Index summary detected, proceeding.")
            break

        # חכה 10 שניות בין סבבים
        print("⏸ Waiting 10 seconds before next batch...")
        time.sleep(10)

    if not os.path.exists(SUMMARY_FILE):
        print("⚠️ Index summary not found after full run.")

# הרצת בדיקה
ensure_index_exists()


# ===============================
# פונקציה לקיצוץ טקסט חכם
# ===============================
def clean_and_trim_text(text: str, max_length: int = 400) -> str:
    """מסיר רווחים וקוטע בסוף משפט"""
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
    status = {
        "index_dir_exists": os.path.exists(CHROMA_DIR),
        "index_summary_exists": os.path.exists(SUMMARY_FILE),
        "files_in_index_dir": None,
        "indexed_pages": None,
        "total_chunks": None,
        "chroma_collection_docs": None,
        "errors": []
    }

    # כמה קבצים יש בתיקייה ./data/index
    if os.path.exists(CHROMA_DIR):
        try:
            status["files_in_index_dir"] = len(os.listdir(CHROMA_DIR))
        except Exception as e:
            status["errors"].append(f"Error reading index dir: {e}")

    # קריאת index_summary.json
    if os.path.exists(SUMMARY_FILE):
        try:
            with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
                summary = json.load(f)
                status["indexed_pages"] = len(summary.get("files", []))
                status["total_chunks"] = summary.get("total_chunks", 0)
        except Exception as e:
            status["errors"].append(f"Error reading summary file: {e}")

    # קריאת מצב אוסף המסמכים ב־ChromaDB
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection("shabaton_faq")
        status["chroma_collection_docs"] = collection.count()
    except Exception as e:
        status["errors"].append(f"Error accessing ChromaDB: {e}")

    return status


# ===============================
# API – /query
# ===============================
@app.post("/query")
async def query(request: Request):
    """מענה לשאלות מהאינדקס בלבד (ללא GPT)"""
    data = await request.json()
    question = data.get("query", "").strip()
    if not question:
        return {"answer": "לא התקבלה שאלה.", "sources": []}

    answer_text = ""
    sources = []

    try:
        # חיפוש ב-Chroma לפי הטקסט
        results = collection.query(
            query_texts=[question],
            n_results=3
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        if docs:
            combined = []
            for i, d in enumerate(docs):
                url = metas[i].get("url", "לא ידוע")
                snippet = clean_and_trim_text(d)
                combined.append(f"🔹 מקור: {url}\n{snippet}")
                sources.append(url)
            answer_text = "\n\n".join(combined)
        else:
            answer_text = "לא נמצא מידע רלוונטי, מוזמנים לפנות לצוות שבתון במייל info@shabaton.co.il"

    except Exception as e:
        print(f"⚠️ Error querying Chroma: {e}")
        answer_text = "אירעה שגיאה בגישה למידע."

    # שליחה ל-Zapier (לא חובה)
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(ZAPIER_WEBHOOK_URL, json={
                "timestamp": datetime.utcnow().isoformat(),
                "question": question,
                "answer": answer_text,
                "sources": sources,
                "page": request.headers.get("Referer", "Unknown"),
            })
    except Exception as e:
        print(f"⚠️ Failed to send to Zapier: {e}")

    return {"answer": answer_text, "sources": sources}
