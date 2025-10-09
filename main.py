import os, json, subprocess, aiohttp
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")
SUMMARY_FILE = "data/index_summary.json"
ZAPIER_WEBHOOK_URL = "https://hooks.zapier.com/hooks/catch/5499574/u5u0yfy/"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="AI Shabaton API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection("shabaton_faq")

def ensure_index_exists():
    if not os.path.exists(SUMMARY_FILE):
        subprocess.run(["python", "indexer.py"], check=False)

ensure_index_exists()

@app.post("/query")
async def query(request: Request):
    data = await request.json()
    question = data.get("query", "").strip()
    if not question:
        return {"answer": "לא התקבלה שאלה.", "sources": []}

    answer_text = None
    sources = []
    try:
        # חיפוש דוקומנטים דומים לפי embedding החדש
        embedding = openai_client.embeddings.create(model="text-embedding-3-small", input=[question]).data[0].embedding
        results = collection.query(query_embeddings=[embedding], n_results=3)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        if docs:
            context = "\n\n".join(docs)
            sources = [m["url"] for m in metas if "url" in m]
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "אתה עוזר חכם שמבוסס רק על מידע מתוך אתר שבתון."},
                    {"role": "user", "content": f"שאלה: {question}\n\nמידע רלוונטי מהאתר:\n{context}"}
                ]
            )
            answer_text = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Error querying Chroma: {e}")

    if not answer_text:
        answer_text = "לא נמצא מידע רלוונטי, מוזמנים לפנות לצוות שבתון במייל info@shabaton.co.il"

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
