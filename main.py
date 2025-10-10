import os
import json
import subprocess
import aiohttp
import re
import time
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from dotenv import load_dotenv

# ===============================
# טעינת משתני סביבה
# ===============================
load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")
SUMMARY_FILE = "data/index_summary.json"
ZAPIER_WEBHOOK_URL = "https://hooks.zapier.com/hooks/catch/5499574/u5u0yfy/"

# ===============================
# הגדרת אפליקציית FastAPI
# ===============================
app = FastAPI(title="AI Shabaton – ללא GPT")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# אינדוקס ראשוני עם מנגנון חכם
# ===============================
def ensure_index_exists():
    """ודא שהאינדקס קיים – אם כבר נוצר, אל תרוץ שוב"""
    if os.path.exists(SUMMARY_FILE):
        print("✅ Index summary found – skipping indexing.")
        return

    print("🟢 No index found – starting incremental indexing...")
    max_runtime_minutes = 60
    start = time.time()

    while time.time() - start < max_runtime_minutes * 60:
        print("🔄 Running incremental indexing batch...", flush=True)
        result = subprocess.run(["python", "indexer.py"], capture_output=True, text=True)

        if result.stdout:
            print(result.stdout)
        if result.stderr.strip():
            print(result.stderr)

        if "✅" in result.stdout or "completed" in result.stdout.lower() or "done" in result.stdout.lower():
            print("✅ Indexing fully completed.")
            break

        if os.path.exists(SUMMARY_FILE):
            print("✅ Index summary detected mid-run – stopping further indexing.")
            break

        print("⏸ Waiting 10 seconds before next batch...")
        time.sleep(10)

    if not os.path.exists(SUMMARY_FILE):
        print("⚠️ Index summary not found after full run.")

# הפעלת בדיקה בתחילת ההרצה
ensure_index_exists()

# =
