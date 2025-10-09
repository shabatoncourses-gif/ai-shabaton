import os
import json
import hashlib
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# --- טעינת משתני סביבה ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

SUMMARY_FILE = os.path.join("data", "index_summary.json")
CACHE_FILE = os.path.join("data", "index_cache.json")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ Missing OPENAI_API_KEY — please set it in Render or .env file")

# --- יצירת תיקיות ---
os.makedirs("data", exist_ok=True)
os.makedirs("data/pages", exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# --- חיבור למסד הנתונים ---
client = chromadb.PersistentClient(path=CHROMA_DIR)

# --- הגדרת embedding function ---
class SafeOpenAIEmbeddingFunction:
    def __init__(self, api_key, model_name):
        import openai
        openai.api_key = api_key
        self.client = openai
        self.model_name = model_name

    def __call__(self, texts):
        res = self.client.Embedding.create(model=self.model_name, input=texts)
        return [d["embedding"] for d in res["data"]]

ef = SafeOpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBED_MODEL)

# --- טעינת / יצירת collection ---
try:
    collection = client.get_collection(name="shabaton_faq")
    print("✅ Loaded existing collection 'shabaton_faq'")
except Exception:
    collection = client.create_collection(name="shabaton_faq", embedding_function=ef)
    print("🆕 Created new collection 'shabaton_faq'")

# --- טעינת cache ---
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
else:
    cache = {}

# --- איתור קבצי טקסט ---
pages_dir = "data/pages"
files = [f for f in os.listdir(pages_dir) if f.endswith(".txt")]

if not files:
    print("⚠️ No .txt files found in data/pages — skipping indexing.")
    exit(0)

index_summary = {"files": [], "total_chunks": 0}
updated_files = 0
skipped_files = 0

for fname in files:
    path = os.path.join(pages_dir, fname)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # חישוב hash של התוכן
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    if cache.get(fname) == text_hash:
        skipped_files += 1
        print(f"⏩ Skipped (no change): {fname}")
        continue  # לא השתנה, מדלגים

    # חלוקה לקטעים
    max_chars = int(os.getenv("MAX_CHUNK_TOKENS", "800")) * 4
    chunks = [
        text[i:i + max_chars]
        for i in range(0, len(text), max_chars)
        if len(text[i:i + max_chars].strip()) > 50
    ]

    ids = [f"{fname}#chunk{i}" for i in range(len(chunks))]
    metas = [{"source": f"https://www.shabaton.online/{fname.replace('.txt', '')}"} for _ in chunks]

    if chunks:
        try:
            # מחיקת גרסה קודמת של אותו דף (אם קיימת)
            existing = [id for id in ids if id in collection.get(ids)["ids"]]
            if existing:
                collection.delete(ids=existing)

            # הוספת הנתונים החדשים
            collection.add(documents=chunks, metadatas=metas, ids=ids)
            updated_files += 1
            index_summary["files"].append({
                "file": fname,
                "chunks": len(chunks),
                "source": metas[0]["source"]
            })
            print(f"[+] Indexed {fname} ({len(chunks)} chunks)")
            cache[fname] = text_hash  # עדכון cache
        except Exception as e:
            print(f"[!] Failed to add {fname}: {e}")

index_summary["total_chunks"] = sum(f["chunks"] for f in index_summary["files"])

# --- שמירת סיכום ו־cache ---
with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(index_summary, f, ensure_ascii=False, indent=2)

with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)

# --- סיכום ---
print("\n📦 Indexing Summary:")
print(f"   • Files indexed/updated: {updated_files}")
print(f"   • Files skipped (no change): {skipped_files}")
print(f"   • Total chunks: {index_summary['total_chunks']}")
print(f"   • Saved summary: {SUMMARY_FILE}")
print("✅ Incremental indexing complete!")
