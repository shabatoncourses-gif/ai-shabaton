import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import chromadb

# --- טעינת משתני סביבה ---
env_loaded = load_dotenv()
if env_loaded:
    print("✅ .env file loaded successfully.")
else:
    print("⚠️ .env file not found (this is normal on Render).")

# --- משתני קונפיגורציה ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "4"))
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")
BACKEND_ORIGIN = os.getenv("BACKEND_ORIGIN", "*")

# --- בדיקה על המפתח ---
if not OPENAI_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY missing. Please set it in Render > Environment Variables.")
    raise RuntimeError("OPENAI_API_KEY missing")
else:
    print("✅ OPENAI_API_KEY loaded successfully.")

# --- הצגת פרטי קונפיגורציה ---
print("🔧 Configuration Summary:")
print(f"  → EMBED_MODEL: {EMBED_MODEL}")
print(f"  → LLM_MODEL: {LLM_MODEL}")
print(f"  → CHROMA_DIR: {CHROMA_DIR}")
print(f"  → TOP_K: {TOP_K}")
print(f"  → BACKEND_ORIGIN: {BACKEND_ORIGIN}")

# --- אתחול OpenAI ---
openai.api_key = OPENAI_API_KEY

# --- יצירת תיקייה ל-ChromaDB אם לא קיימת ---
os.makedirs(CHROMA_DIR, exist_ok=True)

# --- אתחול לקוח Chroma ---
client = chromadb.Client(chromadb.config.Settings(persist_directory=CHROMA_DIR))

# --- בדיקה אם הקולקציה קיימת ---
try:
    collection = client.get_collection("shabaton_faq")
    print("✅ Chroma collection 'shabaton_faq' loaded successfully.")
except Exception:
    print("⚠️ Collection 'shabaton_faq' not found — creating a new one...")
    collection = client.create_collection("shabaton_faq")
    print("✅ New collection 'shabaton_faq' created successfully.")

# --- אתחול FastAPI ---
app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[BACKEND_ORIGIN] if BACKEND_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- מודל בקשה ---
class QueryIn(BaseModel):
    query: str

# --- בניית פרומפט ---
def build_prompt(question, contexts):
    header = (
        "אתה סוכן שירות רשמי עבור אתר Shabaton.online. "
        "ענה בעברית בלבד, בסגנון מקצועי, מבוסס על המידע הנתון בלבד. "
        "אם אין מידע — אמור שאין מידע זמין והפנה לעמוד יצירת קשר.\n\n"
    )
    ctx_texts = []
    for i, c in enumerate(contexts):
        s = f"[מקור {i+1}] {c.get('source', '')}\n{c.get('content', '')}\n"
        ctx_texts.append(s)
    return header + "\n\n".join(ctx_texts) + f"\n\nשאלה: {question}\n\nתשובה (בעברית):"

# --- API: ביצוע שאילתה ---
@app.post("/query")
async def query(q: QueryIn):
    qtext = q.query.strip()
    if not qtext:
        raise HTTPException(status_code=400, detail="Empty query")

    # שליפת קטעים רלוונטיים מ-Chroma
    try:
        res = collection.query(
            query_texts=[qtext],
            n_results=TOP_K,
            include=["documents", "metadatas"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chroma query failed: {e}")

    docs = []
    for doc, meta in zip(res.get("documents", [[]])[0], res.get("metadatas", [[]])[0]):
        docs.append({"content": doc, "source": meta.get("source", "")})

    prompt = build_prompt(qtext, docs)

    # קריאה ל-OpenAI
    try:
        completion = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "אתה עוזר מועיל העונה בעברית בלבד."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=700,
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

    sources = list({d["source"] for d in docs if d.get("source")})
    return {"status": "ok", "answer": answer, "sources": sources}

# --- שורש לבדיקה ---
@app.get("/")
async def root():
    return {"status": "ok", "message": "Shabaton FAQ API is running 🚀"}
