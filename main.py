import os
import json
import subprocess
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from openai import OpenAI

# --- טעינת משתני סביבה ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")
SUMMARY_FILE = os.path.join("data", "index_summary.json")

# --- אפליקציה ---
app = FastAPI(title="AI Shabaton API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- התחברות למסד הנתונים של Chroma ---
collection = None
if os.path.exists(CHROMA_DIR):
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_or_create_collection("shabaton_faq")
        print(f"✅ Connected to Chroma at {CHROMA_DIR}")
    except Exception as e:
        print(f"⚠️ Could not connect to Chroma: {e}")
else:
    print(f"⚠️ Chroma directory {CHROMA_DIR} not found.")

# --- נקודת קצה לשאלות מהצ'אט ---
@app.post("/query")
async def query(request: Request):
    data = await request.json()
    question = data.get("query", "").strip()
    if not question:
        return {"answer": "לא התקבלה שאלה.", "sources": []}

    chunks, sources = [], []

    # 🔍 שלב 1 — חיפוש רלוונטי במאגר Chroma
    if collection:
        try:
            results = collection.query(
                query_texts=[question],
                n_results=4
            )
            chunks = results.get("documents", [[]])[0]
            sources = [
                m.get("source") for m in results.get("metadatas", [[]])[0] if "source" in m
            ]
        except Exception as e:
            print("⚠️ Chroma query failed:", e)

    # 🧭 שלב 2 — אם אין תוצאות, תשובת fallback מותאמת
    if not chunks:
        print("⚠️ No relevant chunks found — using fallback message.")
        return {
            "answer": "לא נמצא מידע רלוונטי, מוזמנים לפנות לצוות שבתון במייל info@shabaton.co.il",
            "sources": []
        }

    # 🔗 שלב 3 — נבנה הקשר מהקטעים שנמצאו
    context = "\n\n---\n\n".join(chunks)
    prompt = (
        "ענה לשאלה הבאה בהתבסס אך ורק על המידע הבא מהאתר שבתון.\n"
        "אם אין תשובה מדויקת, כתוב:\n"
        "'לא נמצא מידע רלוונטי, מוזמנים לפנות לצוות שבתון במייל info@shabaton.co.il'\n\n"
        f"שאלה: {question}\n\n"
        f"מקטעים רלוונטיים:\n{context}"
    )

    # 🤖 שלב 4 — שליחת הבקשה למודל
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "אתה עוזר מבוסס ידע של אתר שבתון."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content.strip()

    return {"answer": answer, "sources": list(set(sources))}

# --- דף בדיקה בסיסי ---
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Shabaton AI API is running",
        "query_endpoint": "/query"
    }
