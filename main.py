# main.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import chromadb

# --- טעינת משתני סביבה ---
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'text-embedding-3-small')
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4o-mini')
TOP_K = int(os.getenv('TOP_K', '4'))
CHROMA_DIR = os.getenv('CHROMA_DB_DIR', './data/index')
BACKEND_ORIGIN = os.getenv('BACKEND_ORIGIN', '*')

# --- בדיקת קיום מפתח ---
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY לא נמצא (MISSING)")
    raise RuntimeError('OPENAI_API_KEY missing')
else:
    print(f"✅ OPENAI_API_KEY נמצא: {OPENAI_API_KEY[:8]}********")

# --- הגדרות OpenAI + Chroma ---
openai.api_key = OPENAI_API_KEY
client = chromadb.Client(chromadb.config.Settings(persist_directory=CHROMA_DIR))
collection = client.get_collection('shabaton_faq')

# --- FastAPI setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[BACKEND_ORIGIN] if BACKEND_ORIGIN != '*' else ['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

# --- מודל לקלט ---
class QueryIn(BaseModel):
    query: str

# --- בניית prompt ---
def build_prompt(question, contexts):
    header = (
        'אתה סוכן שירות רשמי עבור אתר Shabaton.online. ענה בעברית בלבד, בסגנון רשמי ומקצועי. '
        'הסתמך אך ורק על המידע המסופק בקטעים שלמטה. אם לא קיים מידע מתאים — אמור שאין מידע זמין והפנה לאמצעי יצירת קשר באתר.\n\n'
    )
    ctx_texts = []
    for i, c in enumerate(contexts):
        s = f"[מקור {i+1}] {c.get('source','')}\n{c.get('content','')}\n"
        ctx_texts.append(s)
    prompt = header + '\n\n'
