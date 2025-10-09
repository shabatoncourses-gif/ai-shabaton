import os
import json
import hashlib
import requests
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime

# === הגדרות בסיס ===
load_dotenv()

SITEMAPS = [
    os.getenv("SITEMAP_URL_1", "https://www.shabaton.online/sitemap.xml"),
    os.getenv("SITEMAP_URL_2", "https://www.morim.boutique/sitemap.xml"),
]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

SUMMARY_FILE = os.path.join("data", "index_summary.json")
CACHE_FILE = os.path.join("data", "index_cache.json")
HISTORY_FILE = os.path.join("data", "index_history.json")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ Missing OPENAI_API_KEY — please set it in Render or .env file")

os.makedirs("data", exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# === חיבור למסד Chroma ===
client = chromadb.PersistentClient(path=CHROMA_DIR)
ef = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBED_MODEL)

try:
    collection = client.get_collection("shabaton_faq")
    print("✅ Loaded existing collection 'shabaton_faq'")
except Exception:
    collection = client.create_collection("shabaton_faq", embedding_function=ef)
    print("🆕 Created new collection 'shabaton_faq'")

# === פונקציות עזר ===
def get_sitemap_links(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "xml")
        return [loc.text.strip() for loc in soup.find_all("loc")]
    except Exception as e:
        print(f"⚠️ Failed to load sitemap {url}: {e}")
        return []

def text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

# === טעינת cache ===
cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)

# === טעינת היסטוריה ===
history = []
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

# === קריאת sitemap-ים ===
urls = []
for sm in SITEMAPS:
    found = get_sitemap_links(sm)
    print(f"🌍 Found {len(found)} URLs in {sm}")
    urls.extend(found)

urls = list(set(urls))  # מניעת כפילויות

if not urls:
    print("⚠️ No URLs found in any sitemap — skipping indexing.")
    exit(0)

index_summary = {"pages": [], "total_chunks": 0}
updated_pages = 0
skipped_pages = 0
new_pages = 0

changes = {
    "timestamp": datetime.utcnow().isoformat(),
    "new": [],
    "updated": [],
    "skipped": [],
}

for url in urls:
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            print(f"⚠️ Skipped (HTTP {r.status_code}): {url}")
            continue

        html = r.text
        text = text_from_html(html)
        if len(text) < 100:
            print(f"⚠️ Skipped (too short): {url}")
            continue

        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        previous_hash = cache.get(url)

        if previous_hash == text_hash:
            skipped_pages += 1
            changes["skipped"].append(url)
            print(f"⏩ Skipped (no change): {url}")
            continue

        # קביעה אם חדש או מעודכן
        if previous_hash is None:
            new_pages += 1
            changes["new"].append(url)
        else:
            updated_pages += 1
            changes["updated"].append(url)

        # יצירת chunks
        max_chars = int(os.getenv("MAX_CHUNK_TOKENS", "800")) * 4
        chunks = [
            text[i:i + max_chars]
            for i in range(0, len(text), max_chars)
            if len(text[i:i + max_chars].strip()) > 50
        ]

        ids = [f"{urlparse(url).path.strip('/') or 'index'}#chunk{i}" for i in range(len(chunks))]
        metas = [{"source": url} for _ in chunks]

        # מחיקת גרסה קודמת (אם קיימת)
        try:
            existing = collection.get(ids=ids)
            if existing and existing["ids"]:
                collection.delete(ids=existing["ids"])
        except Exception:
            pass

        # הוספת הנתונים
        collection.add(documents=chunks, metadatas=metas, ids=ids)
        cache[url] = text_hash

        index_summary["pages"].append({"url": url, "chunks": len(chunks)})
        print(f"[+] Indexed {url} ({len(chunks)} chunks)")

    except Exception as e:
        print(f"[!] Failed to process {url}: {e}")

index_summary["total_chunks"] = sum(p["chunks"] for p in index_summary["pages"])

# === שמירת קבצים ===
with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(index_summary, f, ensure_ascii=False, indent=2)

with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)

history.append(changes)
with open(HISTORY_FILE, "w", encoding="utf-8") as f:
    json.dump(history[-10:], f, ensure_ascii=False, indent=2)  # שמור רק 10 ריצות אחרונות

# === סיכום ===
print("\n📦 Indexing Summary:")
print(f"   • New pages: {new_pages}")
print(f"   • Updated pages: {updated_pages}")
print(f"   • Skipped (no change): {skipped_pages}")
print(f"   • Total chunks: {index_summary['total_chunks']}")
print(f"   • Saved summary: {SUMMARY_FILE}")
print(f"   • Saved history: {HISTORY_FILE}")
print("✅ Incremental multi-site sitemap indexing complete!")
