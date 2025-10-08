import os
from dotenv import load_dotenv

# --- טעינת משתני סביבה מוקדמת ---
load_dotenv()

# הגדרת מפתחות לפני כל ייבוא של chromadb
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY and not os.getenv("CHROMA_OPENAI_API_KEY"):
    os.environ["CHROMA_OPENAI_API_KEY"] = OPENAI_API_KEY

# עכשיו אפשר לייבא chromadb בבטחה
import openai
import chromadb
from chromadb.utils import embedding_functions

# --- הגדרות ---
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

os.makedirs(CHROMA_DIR, exist_ok=True)

# --- חיבור למסד הנתונים של Chroma ---
client = chromadb.Client(chromadb.config.Settings(persist_directory=CHROMA_DIR))

# --- פונקציית embedding ---
ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("CHROMA_OPENAI_API_KEY"),
    model_name=EMBED_MODEL
)

# --- יצירת / טעינת אוסף ---
try:
    collection = client.get_collection(name="shabaton_faq")
except Exception:
    collection = client.create_collection(name="shabaton_faq", embedding_function=ef)

# --- אינדוקס קבצים ---

pages_dir = 'data/pages'
os.makedirs(pages_dir, exist_ok=True)  

files = [f for f in os.listdir(pages_dir) if f.endswith('.txt')]

for fname in files:
    path = os.path.join(pages_dir, fname)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # חיתוך לטקסטים קטנים
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
            print(f"[+] Indexed {fname} ({len(chunks)} chunks)")
        except Exception as e:
            print(f"[!] Failed to add {fname}: {e}")


print("✅ Indexing done successfully.")


