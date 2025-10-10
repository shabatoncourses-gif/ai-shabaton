# indexer.py — גרסה עם resume, לוג אוטומטי ושלבים מדורגים
import os
import json
import hashlib
import time
import requests
import re
import gzip
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime

load_dotenv()

# קונפיג
SITEMAPS = [
    os.getenv("SITEMAP_URL_1", "https://www.shabaton.online/sitemap.xml"),
    os.getenv("SITEMAP_URL_2", "https://www.morim.boutique/sitemap.xml"),
]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
MAX_PAGES_PER_RUN = int(os.getenv("MAX_PAGES_PER_RUN", "100"))  # כמה דפים בכל ריצה

SUMMARY_FILE = os.path.join("data", "index_summary.json")
CACHE_FILE = os.path.join("data", "index_cache.json")
LOG_FILE = os.path.join("data", "index_log.txt")

# בדיקת סביבה
os.makedirs("data", exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

if not OPENAI_API_KEY:
    raise RuntimeError("❌ Missing OPENAI_API_KEY")

# חיבור ל-Chroma
client = chromadb.PersistentClient(path=CHROMA_DIR)
ef = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBED_MODEL)
try:
    collection = client.get_collection("shabaton_faq")
    print("✅ Loaded existing collection 'shabaton_faq'")
except Exception:
    collection = client.create_collection("shabaton_faq", embedding_function=ef)
    print("🆕 Created new collection 'shabaton_faq'")

# ----------------------------
#  לוגים
# ----------------------------
def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

# ----------------------------
#  הורדת URL
# ----------------------------
def fetch_url(url, max_retries=3):
    ua_browser = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ua_google = "Googlebot/2.1 (+http://www.google.com/bot.html)"
    headers = {"User-Agent": ua_browser}

    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=25)
            if r.status_code == 200:
                ce = r.headers.get("Content-Encoding", "").lower()
                content = r.content
                if "gzip" in ce:
                    try:
                        return gzip.decompress(content).decode("utf-8", errors="ignore")
                    except Exception:
                        return content.decode("utf-8", errors="ignore")
                else:
                    return content.decode("utf-8", errors="ignore")
            elif r.status_code == 403:
                headers["User-Agent"] = ua_google
                time.sleep(1)
        except Exception as e:
            log(f"⚠️ Failed to fetch {url}: {e}")
            time.sleep(2)
    return None

# ----------------------------
#  Sitemap parsing
# ----------------------------
def get_sitemap_links(url):
    xml = fetch_url(url)
    if not xml:
        return []
    soup = BeautifulSoup(xml, "xml")
    sitemap_tags = soup.find_all("sitemap")
    if sitemap_tags:
        urls = []
        for sm in sitemap_tags:
            loc = sm.find("loc")
            if loc and loc.text.strip():
                urls.extend(get_sitemap_links(loc.text.strip()))
        return urls
    return [loc.text.strip() for loc in soup.find_all("loc") if loc.text.strip()]

# ----------------------------
#  ניקוי HTML
# ----------------------------
def text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

# ----------------------------
#  אינדוקס בפועל
# ----------------------------
def build_index():
    cache = json.load(open(CACHE_FILE, "r", encoding="utf-8")) if os.path.exists(CACHE_FILE) else {}
    index_summary = {"files": [], "total_chunks": 0}
    urls = []

    for sm in SITEMAPS:
        found = get_sitemap_links(sm)
        log(f"🌍 Found {len(found)} URLs in {sm}")
        urls.extend(found)

    urls = list(dict.fromkeys(urls))
    if not urls:
        log("🚫 No URLs found.")
        return

    # רק דפים חדשים
    new_urls = [u for u in urls if u not in cache]
    if not new_urls:
        log("✅ All pages already indexed.")
        return

    # נחתוך לפי הגבלה לכל ריצה
    urls_to_index = new_urls[:MAX_PAGES_PER_RUN]
    log(f"⚙️ Starting partial index: {len(urls_to_index)} / {len(new_urls)} new pages")

    total_chunks = 0
    for url in urls_to_index:
        html = fetch_url(url)
        if not html:
            continue
        text = text_from_html(html)
        if len(text) < 150:
            continue

        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        max_chars = int(os.getenv("MAX_CHUNK_TOKENS", "800")) * 4
        chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
        ids = [f"{urlparse(url).path.strip('/') or 'index'}#chunk{i}" for i in range(len(chunks))]
        metas = [{"url": url} for _ in chunks]

        try:
            collection.add(documents=chunks, metadatas=metas, ids=ids)
            cache[url] = text_hash
            total_chunks += len(chunks)
            index_summary["files"].append({"url": url, "chunks": len(chunks)})
            log(f"[+] Indexed {url} ({len(chunks)} chunks)")
        except Exception as e:
            log(f"⚠️ Failed to add chunks for {url}: {e}")

    index_summary["total_chunks"] = total_chunks
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(index_summary, f, ensure_ascii=False, indent=2)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    log(f"✅ Run completed: {len(index_summary['files'])} pages, {total_chunks} chunks added.")

if __name__ == "__main__":
    build_index()
