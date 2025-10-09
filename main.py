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
        "index_summary": "/index-summary"
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

# --- חיבור למסד הנתונים של Chroma (לא חובה ל־endpoint הזה, רק לשירות הראשי) ---
if os.path.exists(CHROMA_DIR):
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        print(f"✅ Connected to Chroma at {CHROMA_DIR}")
    except Exception as e:
        print(f"⚠️ Could not connect to Chroma: {e}")
else:
    print(f"⚠️ Chroma directory {CHROMA_DIR} not found.")

