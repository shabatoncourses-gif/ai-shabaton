import os
import json
import subprocess
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import aiohttp
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")
SUMMARY_FILE = "data/index_summary.json"
ZAPIER_WEBHOOK_URL = "https://hooks.zapier.com/hooks/catch/5499574/u5u0yfy/"

app = FastAPI(title="AI Shabaton API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- התחברות ל-Chroma ---
collection = None
if os.path.exists(CHROMA_DIR):
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=os.getenv("EMBED_MODEL", "text-embedding-3-small")
    )
    collection = chroma_client.get_or_create_collection("shabaton_faq", embedding_function=ef)

# --- בניית אינדקס אם חסר ---
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
    sources =
