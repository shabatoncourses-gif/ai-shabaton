# main.py — גרסה יציבה ל־Render + Chroma 0.4.24 + OpenAI 1.30.1
import os
import json
import traceback
import chromadb
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ================================
#   הגדרות בסיסיות
# ================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ Missing OPENAI_API_KEY environment variable")

app = FastAPI(title="Shabaton AI API")

# הרשאות CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ================================
#   טעינת מסד הנתונים
# ================================
def init_chroma():
    """מפעיל את מסד הנתונים ומוודא שאין שדות חסרים"""
    try:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        # מחיקה יזומה של collections פגומות אם קיימות
        try:
            collections = client.list_collections()
            for c in collections:
                if "topic" in c.name.lower():  # הגנה על שדות ישנים
                    print(f"⚠️ Removing outdated collection: {c.name}")
                    client.delete_collection(c.name)
        except Exception as e:
            print(f"⚠️ Skipping collection cleanup: {e}")

        # יצירת אוסף תקין
        return client.get_or_create_collection(name="shabaton_faq")

    except Exception as e:
        print(f"❌ Failed to initialize Chroma: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Database init failed")

collection = init_chroma()

# ================================
#   Embedding פונקציה
# ================================
def embed_query(text: str):
    try:
        res = openai_client.embeddings.create(input=[text], model=EMBED_MODEL)
        return res.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

# ================================
#   בקשות API
# ================================
class Query(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def root():
    return {"status": "ok", "message": "🧠 Shabaton AI is running."}

@app.post("/ask")
def ask(q: Query):
    query_text = q.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        query_emb = embed_query(query_text)
        results = collection.query(query_embeddings=[query_emb], n_results=q.top_k)

        if not results or not results.get("documents") or not results["documents"][0]:
            return {"answer": "לא נמצאו תוצאות רלוונטיות.", "sources": []}

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        urls = [m.get("url") for m in metas if m.get("url")]

        # שליחת השאלה ל־GPT למענה
        context = "\n\n".join(docs[:3])
        prompt = f"""ענה בעברית על השאלה הבאה בהתבסס על המידע הבא בלבד:

שאלה:
{query_text}

מידע רלוונטי:
{context}

ענה בצורה ברורה ותמציתית:
"""

        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        answer = completion.choices[0].message.content.strip()
        return {"answer": answer, "sources": urls}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

@app.get("/status")
def get_status():
    """בדיקת מצב בסיס הנתונים"""
    try:
        total = len(collection.get()["ids"])
    except Exception:
        total = 0
    return {
        "status": "ok",
        "collection": "shabaton_faq",
        "documents": total,
    }

@app.get("/summary")
def get_summary():
    """מחזיר את index_summary.json אם קיים"""
    try:
        path = os.path.join("data", "index_summary.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "not_found", "message": "index_summary.json not found yet."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
