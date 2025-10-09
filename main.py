import os
import json
import subprocess
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import aiohttp  # לשליחת נתונים ל-Zapier

# --- טעינת משתני סביבה ---
load_dotenv()

# --- קבועים ---
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")
SUMMARY_FILE = os.path.join("data", "index_summary.json")
ZAPIER_WEBHOOK_URL = "https://hooks.zapier.com/hooks/catch/5499574/u5u0yfy/"

# --- הגדרת FastAPI ---
app = FastAPI(title="AI Shabaton API")

# --- הגדרות CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # אפשר לשים ["https://www.shabaton.online"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- חיבור ל-OpenAI ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- חיבור למסד הנתונים של Chroma ---
chroma_client = None
collection = None

if os.path.exists(CHROMA_DIR):
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("EMBED_MODEL", "text-embedding-3-small")
        )
        collection = chroma_client.get_or_create_collection("shabaton_faq", embedding_function=ef)
        print(f"✅ Connected to Chroma at {CHROMA_DIR}")
    except Exception as e:
        print(f"⚠️ Could not connect to Chroma: {e}")
else:
    print(f"⚠️ Chroma directory {CHROMA_DIR} not found.")


# --- פונקציה לאינדוקס אוטומטי ---
def ensure_index_exists():
    """אם אין אינדקס — מריץ indexer.py ובונה אחד חדש."""
    if os.path.exists(SUMMARY_FILE):
        print("✅ Found existing index_summary.json — skipping rebuild.")
        return

    print("⚙️ No index found — running indexer.py to build a new one...")
    try:
        result = subprocess.run(
            ["python", "indexer.py"],
            capture_output=True,
            text=True,
            timeout=300
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


# --- נקודת בדיקה ראשית ---
@app.get("/")
def root():
    return {
        "message": "✅ Shabaton API is running",
        "docs": "/docs",
        "index_summary": "/index-summary",
        "indexed_pages": "/indexed-pages",
        "chroma_status": "/chroma-status"
    }


# --- נקודת קצה ראשית לשאלות מהאתר ---
@app.post("/query")
async def query(request: Request):
    """מקבל שאלה מהאתר, מחפש במידע המאונדקס, ואם אין תוצאה — עונה תשובת fallback."""
    data = await request.json()
    question = data.get("query", "").strip()

    if not question:
        return {"answer": "לא התקבלה שאלה.", "sources": []}

    # 🔍 חיפוש במאגר Chroma
    answer_text = None
    sources = []

    if collection:
        try:
            results = collection.query(
                query_texts=[question],
                n_results=3
            )

            if results and results.get("documents") and results["documents"][0]:
                top_docs = results["documents"][0]
                sources = [m["source"] for m in results["metadatas"][0] if "source" in m]

                context = "\n\n".join(top_docs)
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "אתה עוזר חכם שמבוסס רק על מידע מתוך אתר שבתון."},
                        {"role": "user", "content": f"שאלה: {question}\n\nמידע רלוונטי מהאתר:\n{context}"}
                    ]
                )
                answer_text = completion.choices[0].message.content.strip()

        except Exception as e:
            print(f"⚠️ Error querying Chroma: {e}")

    # אם לא נמצאו תוצאות — תשובת fallback
    if not answer_text:
        answer_text = "לא נמצא מידע רלוונטי, מוזמנים לפנות לצוות שבתון במייל info@shabaton.co.il"
        sources = []

    # 📤 שליחת השאלה והתשובה ל-Zapier
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
                    "ip": request.client.host if request.client else "Unknown"
                }
            )
        print("📨 Sent to Zapier")
    except Exception as e:
        print(f"⚠️ Failed to send to Zapier: {e}")

    return {"answer": answer_text, "sources": sources}


# --- תקציר האינדוקס ---
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


# --- בדיקת מצב חיבור למסד הנתונים ---
@app.get("/chroma-status")
def chroma_status():
    if not os.path.exists(CHROMA_DIR):
        return {"status": "not_found", "message": f"Chroma directory not found: {CHROMA_DIR}"}
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collections = client.list_collections()
        return {"status": "ok", "collections": [c.name for c in collections]}
    except Exception as e:
        return {"status": "error", "message": str(e)}
