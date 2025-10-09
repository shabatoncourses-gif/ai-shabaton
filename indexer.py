import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# --- טעינת משתני סביבה ---
env_loaded = load_dotenv()
if env_loaded:
    print("✅ .env file loaded successfully.")
else:
    print("⚠️ .env file not found (this is normal on Render).")

# --- משתנים ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "800"))

# --- בדיקה על המפתח ---
if not OPENAI_API_KEY:
    raise RuntimeError("❌ Missing OPENAI_API_KEY — add it in your Render environment variables.")

# --- יצירת תיקייה ל-ChromaDB ---
os.makedirs(CHROMA_DIR, exist_ok=True)

print("🔧 Configuration Summary:")
print(f"  → CHROMA_DIR: {CHROMA_DIR}")
print(f"  → EMBED_MODEL: {EMBED_MODEL}")
print(f"  → MAX_CHUNK_TOKENS: {MAX_CHUNK_TOKENS}")

# --- אתחול חיבור ל-Chroma ---
client = chromadb.PersistentClient(path=CHROMA_DIR)

# --- פונקציית Embedding ---
ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name=EMBED_MODEL,
)

# --- יצירת או טעינת קולקציה ---
try:
    collection = client.get_collection(name="shabaton_faq")
    print("✅ Loaded existing collection 'shabaton_faq'")
except Exception:
    collection = client.create_collection(name="shabaton_faq", embedding_function=ef)
    print("🆕 Created new collection 'shabaton_faq'")

# --- קריאת קבצי טקסט ---
pages_dir = "data/pages"
if not os.path.exists(pages_dir):
    raise FileNotFoundError(f"❌ Directory '{pages_dir}' not found — create it and add .txt files.")

files = [f for f in os.listdir(pages_dir) if f.endswith(".txt")]
if not files:
    print("⚠️ No .txt files found in data/pages — add content files before indexing.")
    exit(0)
else:
    print(f"📚 Found {len(files)} text files to index.\n")

# --- אינדוקס קבצים ---
for fname in files:
    path = os.path.join(pages_dir, fname)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        print(f"⚠️ Skipping empty file: {fname}")
        continue

    # חלוקה לקטעים
    max_chars = MAX_CHUNK_TOKENS * 4
    chunks = [
        text[i:i + max_chars]
        for i in range(0, len(text), max_chars)
        if len(text[i:i + max_chars].strip()) > 50
    ]

    ids = [f"{fname}#chunk{i}" for i in range(len(chunks))]
    metas = [
        {
            "source": f"https://www.shabaton.online/{fname.replace('_', '.').replace('.txt', '')}"
        }
        for _ in chunks
    ]

    if chunks:
        try:
            collection.add(documents=chunks, metadatas=metas, ids=ids)
            print(f"[+] Indexed {fname} ({len(chunks)} chunks)")
        except Exception as e:
            print(f"[!] Failed to add {fname}: {e}")
    else:
        print(f"⚠️ No valid text chunks found in {fname}")

print("\n✅ Indexing complete! Your data is ready for querying 🚀")
