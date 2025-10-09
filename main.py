import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import chromadb

# --- טעינת משתני סביבה ---
load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")
SUMMARY_FILE = os.path.join("data", "index_summary.json")

# --- יצירת אפליקציית FastAPI ---
app = FastAPI(title="AI Shabaton API")

# --- הגדרות CORS (גישה חופשית לדפדפן) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ברירת מחדל / דף ראשי ---
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
    """מציג סיכום של הקבצים שאונדקסו."""
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
    """מציג רשימה של כל הדפים (URLs) שאונדקסו."""
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
    """בודק אם מסד הנתונים של Chroma מחובר ופועל."""
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
