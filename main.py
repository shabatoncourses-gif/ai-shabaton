import os, json, subprocess, aiohttp, re
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")
SUMMARY_FILE = "data/index_summary.json"
ZAPIER_WEBHOOK_URL = "https://hooks.zapier.com/hooks/catch/5499574/u5u0yfy/"

app = FastAPI(title="AI Shabaton – ללא GPT")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# התחברות ל-ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection("shabaton_faq")

def ensure_index_exists():
    """ודא שהאינדקס קיים"""
    if not os.path.exists(SUMMARY_FILE):
        subprocess.run(["python", "indexer.py"], check=False)

ensure_index_exists()

# קיצוץ טקסט חכם
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
