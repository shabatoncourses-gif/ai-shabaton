import os
import json
import subprocess
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import chromadb

from fastapi import Request
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/query")
async def query(request: Request):
    """מקבל שאלה מהאתר ומחזיר תשובה מהמודל"""
    data = await request.json()
    question = data.get("query", "")

    if not question.strip():
        return {"answer": "לא התקבלה שאלה.", "sources": []}

    # שולחים את השאלה למודל (אפשר לשנות לפי הצורך)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "אתה עוזר חכם שמבוסס על מידע משבתון."},
            {"role": "user", "content": question}
        ]
    )

    answer = response.choices[0].message.content.strip()

    return {"answer": answer, "sources": []}




# --- טעינת משתני סביבה ---
load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")
SUMMARY_FILE = os.path.join("data", "index_summary.json")


# --- הגדרות CORS (גישה חופשית לדפדפן) ---


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Shabaton API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # או ["https://www.shabaton.online"] אם רוצים רק את האתר שלך
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- פונקציה לאינדוקס אוטומטי ---
def ensure_index_exists():
    """אם אין אינדקס — מריץ indexer.py ובונה אחד חדש."""
    if os.path.exists(SUMMARY_FILE):
        print("✅ Found existing index_summary.json — skipping rebuild.")
        return

    print("⚙️ No index found — running indexer.py to build a new one...")
    try:
        # נריץ את הסקריפט indexer.py
        result = subprocess.run(
            ["python", "indexer.py"],
            capture_output=True,
            text=True,
            timeout=300  # עד 5 דקות
        )

        print("📜 --- indexer.py output ---")
        print(result.stdout)
        print(result.stderr)
        print("📜 -------------------------")

        if os.path.exists(SUMMARY_FILE):
            print("✅ Index successfully created!")
        else:
            print("⚠️ index_summary.json not found after indexing.")
    except Exception as e:
        print(f"❌ Failed to run indexer.py: {e}")

# --- נוודא שהאינדקס נבנה כששרת עולה ---
ensure_index_exists()

# --- דף ראשי ---
@app.get("/")
def root():
    return {
        "message": "✅ Shabaton API is running",
        "docs": "/docs",
        "index_summary": "/index-summary",
        "indexed_pages": "/indexed-pages",
        "chroma_status": "/chroma-status"
    }

# --- קריאה לתקציר האינדוקס ---
@app.get("/index-summary")
def get_index_summary():
    if not os.path.exists(SUMMARY_FILE):
        return {"status": "not_found", "message": "index_summary.json not found yet."}

    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "status": "ok",
        "files_indexed": len(data.get("files", [])),
        "total_chunks": data.get("total_chunks", 0),
        "details": data.get("files", [])
    }

# --- רשימת הדפים המאונדקסים ---
@app.get("/indexed-pages")
def get_indexed_pages():
    if not os.path.exists(SUMMARY_FILE):
        return {"status": "not_found", "message": "No index summary found."}

    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages = [item["source"] for item in data.get("files", [])]
    return {
        "status": "ok",
        "total_pages": len(pages),
        "pages": pages
    }

# --- בדיקת מצב חיבור למסד הנתונים של Chroma ---
@app.get("/chroma-status")
def chroma_status():
    if not os.path.exists(CHROMA_DIR):
        return {"status": "not_found", "message": f"Chroma directory not found: {CHROMA_DIR}"}
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collections = client.list_collections()
        return {
            "status": "ok",
            "collections": [c.name for c in collections],
            "path": CHROMA_DIR
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- חיבור למסד הנתונים של Chroma ---
if os.path.exists(CHROMA_DIR):
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        print(f"✅ Connected to Chroma at {CHROMA_DIR}")
    except Exception as e:
        print(f"⚠️ Could not connect to Chroma: {e}")
else:
    print(f"⚠️ Chroma directory {CHROMA_DIR} not found.")


