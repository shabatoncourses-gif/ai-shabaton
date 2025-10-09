import os
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import chromadb
from urllib.parse import urlparse
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

# --- חיבור למסד הנתונים של Chroma ---
client = chromadb.PersistentClient(path=CHROMA_DIR)

# --- מחלקת Embedding תואמת ל־OpenAI SDK החדש ---
class SafeOpenAIEmbeddingFunction:
    def __init__(self, api_key, model_name):
        from openai import OpenAI
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

# --- שליפת דפים מ-sitemap ---
SITEMAPS = [
    "https://www.shabaton.online/sitemap.xml",
    "https://www.morim.boutique/sitemap.xml"
]

def fetch_urls_from_sitemap(url):
    urls = []
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "xml")
        for loc in soup.find_all("loc"):
            link = loc.text.strip()
            if link.startswith("http"):
                urls.append(link)
        print(f"🌐 Found {len(urls)} URLs in {url}")
    except Exception as e:
        print(f"⚠️ Failed to fetch sitemap {url}: {e}")
    return urls

def fetch_text_from_url(url):
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html5lib")

        # מסיר סקריפטים וסטיילים
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        text = soup.get_text(separator=" ", strip=True)
        # מנקה רווחים כפולים
        return " ".join(text.split())
    except Exception as e:
        print(f"⚠️ Failed to fetch {url}: {e}")
        return ""

# --- שלב 1: אינדוקס קבצי טקסט ---
pages_dir = "data/pages"
local_files = [f for f in os.listdir(pages_dir) if f.endswith(".txt")] if os.path.exists(pages_dir) else []
total_chunks = 0
index_summary = {"files": [], "total_chunks": 0}

max_chars = int(os.getenv("MAX_CHUNK_TOKENS", "800")) * 4

for fname in local_files:
    path = os.path.join(pages_dir, fname)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars) if len(text[i:i + max_chars].strip()) > 50]
    ids = [f"{fname}#chunk{i}" for i in range(len(chunks))]
    metas = [{"source": f"https://www.shabaton.online/{fname.replace('_', '.').replace('.txt', '')}"} for _ in chunks]

    if chunks:
        collection.add(documents=chunks, metadatas=metas, ids=ids)
        total_chunks += len(chunks)
        index_summary["files"].append({
            "file": fname,
            "chunks": len(chunks),
            "source": metas[0]["source"]
        })
        print(f"[+] Indexed local file: {fname} ({len(chunks)} chunks)")

# --- שלב 2: אינדוקס דפי האתר (sitemaps) ---
all_urls = []
for sitemap in SITEMAPS:
    all_urls.extend(fetch_urls_from_sitemap(sitemap))

print(f"🌍 Total {len(all_urls)} URLs collected from all sitemaps")

for url in tqdm(all_urls, desc="Indexing sitemap pages"):
    text = fetch_text_from_url(url)
    if len(text) < 100:
        continue

    chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars) if len(text[i:i + max_chars].strip()) > 50]
    ids = [f"{urlparse(url).path}#chunk{i}" for i in range(len(chunks))]
    metas = [{"source": url} for _ in chunks]

    if chunks:
        try:
            collection.add(documents=chunks, metadatas=metas, ids=ids)
            total_chunks += len(chunks)
            index_summary["files"].append({
                "file": urlparse(url).path,
                "chunks": len(chunks),
                "source": url
            })
        except Exception as e:
            print(f"[!] Failed to add {url}: {e}")

# --- שמירת תקציר ---
index_summary["total_chunks"] = total_chunks
with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(index_summary, f, ensure_ascii=False, indent=2)

print("\n📦 Indexing Summary:")
print(f"   • Files indexed: {len(index_summary['files'])}")
print(f"   • Total chunks:  {total_chunks}")
print(f"   • Saved summary: {SUMMARY_FILE}")
print("\n✅ Indexing complete! All sitemap pages and local files are indexed.")
