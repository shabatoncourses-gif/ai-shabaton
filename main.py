import os
import json
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import chromadb
from chromadb.config import Settings

# === הגדרות כלליות ===
app = FastAPI(title="AI Shabaton API")

# הגדרות CORS (מאפשר קריאות מאתרים אחרים)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # תוכל לצמצם לכתובות הספציפיות שלך
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === הגדרות סביבת עבודה ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")

# === חיבור לקליינט של OpenAI ===
try:
    openai_client = OpenAI()  # הספרייה יודעת לקחת את המפתח מה-ENV
    print("✅ OpenAI client initialized successfully.")
except Exception as e:
    print("❌ Failed to initialize OpenAI client:", str(e))
    openai_client = None

# === חיבור למסד הנתונים של Chroma ===
try:
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    print(f"✅ Connected to ChromaDB at: {CHROMA_DB_DIR}")
except Exception as e:
    print("❌ Failed to connect to ChromaDB:", str(e))
    chroma_client = None


# === נקודת בדיקה בסיסית ===
@app.get("/")
def root():
    return {"status": "ok", "message": "AI Shabaton API is running"}


# === קריאה לאינדקס (בדיקה) ===
@app.get("/index/status")
def index_status():
    try:
        summary_path = os.path.join(CHROMA_DB_DIR, "index_summary.json")
        if not os.path.exists(summary_path):
            raise FileNotFoundError("index_summary.json not found")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        return {"status": "ready", "summary": summary}
    except Exception as e:
        return {"status": "not_found", "message": str(e)}


# === נקודת חיפוש ===
@app.get("/search")
def search(query: str, top_k: int = 5):
    if not chroma_client:
        raise HTTPException(status_code=500, detail="ChromaDB client not initialized")

    try:
        collection = chroma_client.get_or_create_collection("pages")
        results = collection.query(query_texts=[query], n_results=top_k)

        if not results or "documents" not in results or not results["documents"][0]:
            return {"results": []}

        response = [
            {
                "text": doc,
                "metadata": meta,
                "distance": dist,
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
        return {"results": response}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# === נקודת דוגמה לתשובה של AI (שילוב עם GPT) ===
@app.post("/ask")
def ask(data: dict):
    question = data.get("question", "")
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' field")

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "אתה עוזר חכם מבית שבתון, ענה בעברית ברורה וקצרה."},
                {"role": "user", "content": question},
            ],
        )
        answer = completion.choices[0].message.content
        return {"question": question, "answer": answer}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")


# === הפעלה ישירה (לבדיקה מקומית) ===
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting local server on http://127.0.0.1:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
