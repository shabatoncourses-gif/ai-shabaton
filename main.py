import os
import json
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# --- טעינת משתני סביבה ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "4"))
SUMMARY_FILE = os.path.join("data", "index_summary.json")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ Missing OPENAI_API_KEY — please add it in Render environment")

# --- אתחול האפליקציה ---
app = FastAPI(title="Shabaton FAQ Backend")

# --- הגדרת CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- חיבור למסד הנתונים ---
client = chromadb.PersistentClient(path=CHROMA_DIR)

# --- פונקציית Embedding תואמת ---
class SafeOpenAIEmbeddingFunction:
    def __init__(self, api_key, model_name):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def __call__(self, texts):
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]

ef = SafeOpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBED_MODEL)

# --- טעינת קולקציה ---
collection = client.get_or_create_collection("shabaton_faq", embedding_function=ef)


# --- ברירת מחדל (דף בית) ---
@app.get("/")
def root():
    return {"status": "✅ Shabaton FAQ backend is running", "model": LLM_MODEL}


# --- שאילתה לטקסטים באינדקס ---
@app.get("/query")
def query_faq(q: str = Query(..., description="שאלה או נושא לחיפוש")):
    """ביצוע חיפוש בשאלות/תשובות המאונדקסות."""
    results = collection.query(query_texts=[q], n_results=TOP_K)

    response = []
    for i in range(len(results["documents"][0])):
        response.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "distance": results["distances"][0][i] if "distances" in results else None,
        })

    return {"query": q, "results": response}


# --- תקציר האינדוקס ---
@app.get("/index-summary")
def get_index_summary():
    """הצגת סיכום של קבצים שאונדקסו."""
    if not os.path.exists(SUMMARY_FILE):
        return {"error": "❌ No summary file found. Run indexer.py first."}

    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {"summary": data}


# --- בריאות השרת ---
@app.get("/health")
def health_check():
    return {"status": "ok", "openai_model": EMBED_MODEL, "chroma_dir": CHROMA_DIR}
