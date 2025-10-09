import os
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

# --- טעינת משתני סביבה ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
SUMMARY_FILE = os.path.join("data", "index_summary.json")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ Missing OPENAI_API_KEY — add it in your .env file or Render environment")

# --- יצירת תיקיות ---
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# --- חיבור למסד הנתונים ---
client = chromadb.PersistentClient(path=CHROMA_DIR)

# --- מחלקת Embedding תואמת ל־OpenAI SDK החדש ---
from openai import OpenAI
class SafeOpenAIEmbeddingFunction:
    def __init__(self, api_key, model_name):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def __call__(self, texts):
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]

ef = SafeOpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBED_MODEL)

# --- טעינת או יצירת קולקציה ---
try:
    collection = client.get_collection(name="shabaton_faq")
    print("✅ Loaded existing collection 'shabaton_faq'")
except Exception:
    collection = client.create_collection(name="shabaton_faq", embedding_function=ef)
    print("🆕 Created new collection 'shabaton_faq'")

# --- קריאת קבצי טקסט מקומיים ---
pages_dir = "data/pages"
local_files = []
if os.path.exists(pages_dir):
    local_files = [os.path.join(pages_dir, f) for f in os.listdir(pages_dir) if f.endswith(".txt")]

# --- Sitemap URLs ---
SITEMAPS = [
    "https://www.shabaton.online/sitemap.xml",
    "https://www.morim.boutique/sitemap.xml"
]

def fetch_sitemap_urls(sitemap_url):
    """קורא sitemap.xml ומחזיר רשימת כתובות URL"""
    urls = []
    try:
        r = requests.get(sitemap_url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "xml")
        for loc in soup.find_all("loc"):
            url = loc.text.strip()
            if any(url.endswith(ext) for ext in [".jpg", ".png", ".pdf", ".mp4"]):
                continue
            urls.append(url)
    except Exception as e:
        print(f"[!] Failed to fetch sitemap {sitemap_url}: {e}")
    return urls

def fetch_page_text(url):
    """מוריד ומנקה את התוכן מטקסט HTML"""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html5lib")
        # הסרה של סקריפטים וסטיילים
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text
    except Exception as e:
        print(f"[!] Failed to fetch {url}: {e}")
        return ""

# --- איסוף כל הדפים מה-sitemaps ---
all_urls = []
for sm in SITEMAPS:
    urls = fetch_sitemap_urls(sm)
    print(f"🌐 Fetched {len(urls)} URLs from {sm}")
    all_urls.extend(urls)

# --- אינדוקס ---
total_chunks = 0
index_summary = {"files": [], "total_chunks": 0}

def add_chunks_to_collection(chunks, source_name, source_url):
    global total_chunks
    ids = [f"{source_name}#chunk{i}" for i in range(len(chunks))]
    metas = [{"source": source_url} for _ in chunks]
    try:
        collection.add(documents=chunks, metadatas=metas, ids=ids)
        total_chunks += len(chunks)
        index_summary["files"].append({
            "file": source_name,
            "chunks": len(chunks),
            "source": source_url
        })
        print(f"[+] Indexed {source_name} ({len(chunks)} chunks)")
    except Exception as e:
        print(f"[!] Failed to index {source_name}: {e}")

max_chars = int(os.getenv("MAX_CHUNK_TOKENS", "800")) * 4

# --- אינדוקס קבצי טקסט מקומיים ---
for path in local_files:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [
        text[i:i + max_chars]
        for i in range(0, len(text), max_chars)
        if len(text[i:i + max_chars].strip()) > 50
    ]
    fname = os.path.basename(path)
    add_chunks_to_collection(chunks, fname, f"https://www.shabaton.online/{fname}")

# --- אינדוקס דפי האתר מתוך ה־sitemap ---
for url in tqdm(all_urls, desc="🌍 Indexing website pages"):
    text = fetch_page_text(url)
    if not text or len(text) < 100:
        continue
    chunks = [
        text[i:i + max_chars]
        for i in range(0, len(text), max_chars)
        if len(text[i:i + max_chars].strip()) > 50
    ]
    name = url.replace("https://", "").replace("/", "_")
    add_chunks_to_collection(chunks, name, url)

# --- שמירת תקציר ---
index_summary["total_chunks"] = total_chunks
with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(index_summary, f, ensure_ascii=False, indent=2)

# --- סיכום ---
print("\n📦 Indexing Summary:")
print(f"   • Local files: {len(local_files)}")
print(f"   • Sitemap pages: {len(all_urls)}")
print(f"   • Total chunks:  {total_chunks}")
print(f"   • Saved summary: {SUMMARY_FILE}")
print("\n✅ Indexing complete! Your data is ready for querying.")
