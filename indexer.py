import os
import json
import hashlib
import requests
import re
import gzip
from io import BytesIO
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime

# === טעינת משתני סביבה ===
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

# === הגדרת סביבה ===
if not OPENAI_API_KEY:
    raise RuntimeError("❌ Missing OPENAI_API_KEY")

os.makedirs("data", exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# === חיבור ל-Chroma ===
client = chromadb.PersistentClient(path=CHROMA_DIR)
ef = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBED_MODEL)

try:
    collection = client.get_collection("shabaton_faq")
    print("✅ Loaded existing collection 'shabaton_faq'")
except Exception:
    collection = client.create_collection("shabaton_faq", embedding_function=ef)
    print("🆕 Created new collection 'shabaton_faq'")

# === פונקציות עזר ===
def fetch_url(url):
    """מוריד דף HTML או XML, עם התחזות לדפדפן אמיתי כדי לעקוף חסימות 403."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "he,en-US;q=0.7,en;q=0.3",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
    }

    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 403:
            print(f"⚠️ Got 403 for {url} — retrying as Googlebot...")
            headers["User-Agent"] = "Googlebot/2.1 (+http://www.google.com/bot.html)"
            r = requests.get(url, headers=headers, timeout=30)

        r.raise_for_status()

        # טיפול בקבצים דחוסים (gzip)
        if url.endswith(".gz"):
            return gzip.decompress(r.content).decode("utf-8")
        elif r.headers.get("Content-Encoding") == "gzip":
            return gzip.decompress(r.content).decode("utf-8")
        else:
            return r.text

    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return None


def get_sitemap_links(url):
    """קורא sitemap רגיל או sitemap index ומחזיר את כל ה־URLs שבו."""
    xml = fetch_url(url)
    if not xml:
        return []

    soup = BeautifulSoup(xml, "xml")

    # אם זה sitemap index — נרד רמה אחת
    sitemap_tags = soup.find_all("sitemap")
    if sitemap_tags:
        urls = []
        for sm in sitemap_tags:
            loc = sm.find("loc")
            if loc and loc.text.strip():
                print(f"🗂 Found sub-sitemap: {loc.text.strip()}")
                urls.extend(get_sitemap_links(loc.text.strip()))
        return urls

    # אחרת — sitemap רגיל
    return [loc.text.strip() for loc in soup.find_all("loc") if loc.text.strip()]


def text_from_html(html):
    """מנקה HTML ומשאיר רק טקסט קריא."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

# === בניית האינדקס ===
def build_index():
    cache = json.load(open(CACHE_FILE, "r", encoding="utf-8")) if os.path.exists(CACHE_FILE) else {}
    history = json.load(open(HISTORY_FILE, "r", encoding="utf-8")) if os.path.exists(HISTORY_FILE) else []

    urls = []
    for sm in SITEMAPS:
        found = get_sitemap_links(sm)
        print(f"🌍 Found {len(found)} URLs in {sm}")
        urls.extend(found)

    urls = list(set(urls))
    if not urls:
        print("⚠️ No URLs found — skipping indexing.")
        return

    index_summary = {"files": [], "total_chunks": 0}

    for url in urls:
        html = fetch_url(url)
        if not html:
            continue
        text = text_from_html(html)
        if len(text) < 100:
            continue

        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if cache.get(url) == text_hash:
            continue

        max_chars = int(os.getenv("MAX_CHUNK_TOKENS", "800")) * 4
        chunks = [
            text[i:i + max_chars]
            for i in range(0, len(text), max_chars)
            if len(text[i:i + max_chars].strip()) > 50
        ]
        ids = [f"{urlparse(url).path.strip('/') or 'index'}#chunk{i}" for i in range(len(chunks))]
        metas = [{"url": url} for _ in chunks]

        try:
            collection.add(documents=chunks, metadatas=metas, ids=ids)
        except Exception as e:
            print(f"⚠️ Failed to add chunks for {url}: {e}")
            continue

        cache[url] = text_hash
        index_summary["files"].append({"url": url, "chunks": len(chunks)})

    index_summary["total_chunks"] = sum(p["chunks"] for p in index_summary["files"])

    json.dump(index_summary, open(SUMMARY_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(cache, open(CACHE_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"✅ Indexed {len(index_summary['files'])} pages, total {index_summary['total_chunks']} chunks.")

# === הפעלה ===
if __name__ == "__main__":
    build_index()
