import os
import json
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import chromadb

# --- טעינת משתני סביבה ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
SUMMARY_FILE = os.path.join("data", "index_summary.json")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ Missing OPENAI_API_KEY — add it in Render environment or .env")

# --- הגדרת כתובות הסייטמאפ ---
SITEMAPS = [
    "https://www.shabaton.online/sitemap.xml",
    "https://www.morim.boutique/sitemap.xml"
]

# --- יצירת תיקיות ---
os.makedirs("data", exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# --- חיבור למסד הנתונים ---
client = chromadb.PersistentClient(path=CHROMA_DIR)

# --- מחלקת Embedding תואמת ---
class SafeOpenAIEmbeddingFunction:
    def __init__(self, api_key, model_name):
        import openai
        openai.api_key = api_key
        self.model = model_name

    def __call__(self, texts):
        import openai
        response = openai.Embedding.create(model=self.model, input=texts)
        return [d["embedding"] for d in response["data"]]

ef = SafeOpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBED_MODEL)

# --- טעינת או יצירת קולקציה ---
try:
    collection = client.get_collection(name="shabaton_faq")
    print("✅ Loaded existing collection 'shabaton_faq'")
except Exception:
    collection = client.create_collection(name="shabaton_faq", embedding_function=ef)
    print("🆕 Created new collection 'shabaton_faq'")

# --- פונקציה: חילוץ URLs מתוך Sitemap ---
def extract_urls_from_sitemap(url):
    try:
        res = requests.get(url, timeout=20)
        res.raise_for_status()
        root = ET.fromstring(res.text)
        return [elem.text for elem in root.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    except Exception as e:
        print(f"⚠️ Failed to read sitemap {url}: {e}")
        return []

# --- פונקציה: המרת HTML לטקסט נקי ---
def extract_text_from_url(url):
    try:
        res = requests.get(url, timeout=20)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        # הסרת סקריפטים וסגנונות
        for s in soup(["script", "style", "noscript"]):
            s.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text
    except Exception as e:
        print(f"⚠️ Failed to fetch {url}: {e}")
        return None

# --- שליפת כל ה־URLs ---
all_urls = []
for sitemap in SITEMAPS:
    urls = extract_urls_from_sitemap(sitemap)
    all_urls.extend(urls)
print(f"🌐 Found {len(all_urls)} URLs from {len(SITEMAPS)} sitemaps.")

if not all_urls:
    print("⚠️ No URLs found — stopping indexing.")
    exit(0)

# --- אינדוקס דפים ---
max_chars = int(os.getenv("MAX_CHUNK_TOKENS", "800")) * 4
total_chunks = 0
index_summary = {"files": [], "total_chunks": 0}

for url in all_urls:
    text = extract_text_from_url(url)
    if not text or len(text) < 100:
        print(f"⏭️ Skipping (too short or failed): {url}")
        continue

    chunks = [
        text[i:i + max_chars]
        for i in range(0, len(text), max_chars)
        if len(text[i:i + max_chars].strip()) > 50
    ]

    ids = [f"{url}#chunk{i}" for i in range(len(chunks))]
    metas = [{"source": url} for _ in chunks]

    try:
        collection.add(documents=chunks, metadatas=metas, ids=ids)
        total_chunks += len(chunks)
        index_summary["files"].append({
            "url": url,
            "chunks": len(chunks)
        })
        print(f"[+] Indexed {url} ({len(chunks)} chunks)")
    except Exception as e:
        print(f"[!] Failed to add {url}: {e}")

# --- שמירת סיכום ---
index_summary["total_chunks"] = total_chunks
with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(index_summary, f, ensure_ascii=False, indent=2)

print("\n📦 Indexing Summary:")
print(f"   • Pages indexed: {len(index_summary['files'])}")
print(f"   • Total chunks:  {total_chunks}")
print(f"   • Saved to: {SUMMARY_FILE}")
print("✅ Indexing complete!")
