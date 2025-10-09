import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# --- טעינת משתני סביבה ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ Missing OPENAI_API_KEY — add it in your .env file or in Render environment.")

# --- יצירת תיקייה למסד הנתונים אם לא קיימת ---
os.makedirs(CHROMA_DIR, exist_ok=True)

# --- חיבור ל־ChromaDB ---
client = chromadb.PersistentClient(path=CHROMA_DIR)

# --- יצירת פונקציית Embedding ---
ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name=EMBED_MODEL,
)

# --- טעינת או יצירת קולקציה ---
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

print(f"📚 Found {len(files)} text files to index.\n")

# --- אינדוקס קבצים ---
total_chunks = 0

for fname in files:
    path = os.path.join(pages_dir, fname)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # חלוקה לקטעים (chunks)
    max_chars = int(os.getenv("MAX_CHUNK_TOKENS", "800")) * 4
    chunks = [
        text[i:i + max_chars]
        for i in range(0, len(text), max_chars)
        if len(text[i:i + max_chars].strip()) > 50
    ]

    ids = [f"{fname}#chunk{i}" for i in range(len(chunks))]
    metas = [
        {"source": f"https://www.shabaton.online/{fname.replace('_', '.').replace('.txt', '')}"}
        for _ in chunks
    ]

    if chunks:
        try:
            collection.add(documents=chunks, metadatas=metas, ids=ids)
            total_chunks += len(chunks)
            print(f"[+] Indexed {fname} ({len(chunks)} chunks)")
        except Exception as e:
            print(f"[!] Failed to add {fname}: {e}")

# --- סיכום ---
print("\n📦 Indexing Summary:")
print(f"   • Files indexed: {len(files)}")
print(f"   • Total chunks:  {total_chunks}")
print("\n✅ Indexing complete! Your data is ready for querying.")
