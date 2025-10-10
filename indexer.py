# indexer.py — גרסה 2025-10 עם resume חכם ואינדוקס מתגלגל
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
from openai import OpenAI

# ===============================
#   הגדרות סביבה
# ===============================
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

if not OPENAI_API_KEY:
    raise RuntimeError("❌ Missing OPENAI_API_KEY")

os.makedirs("data", exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# ===============================
#   חיבור ל־Chroma + OpenAI
# ===============================
openai_client = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection("shabaton_faq")


# ===============================
#   פונקציית Embedding עם retry
# ===============================
def embed_texts(texts, retries=3):
    all_embeddings = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i + 50]
        for attempt in range(retries):
            try:
                res = openai_client.embeddings.create(input=batch, model=EMBED_MODEL)
                all_embeddings.extend([d.embedding for d in res.data])
                time.sleep(0.8)
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"⚠️ OpenAI error: {e}, waiting {wait}s")
                time.sleep(wait)
        else:
            print("❌ Failed to embed batch after retries.")
    return all_embeddings


# ===============================
#   Fetch URL — עם gzip ו־403 bypass
# ===============================
def fetch_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200:
            ce = r.headers.get("Content-Encoding", "").lower()
            if "gzip" in ce:
                try:
                    return gzip.decompress(r.content).decode("utf-8", errors="ignore")
                except:
                    return r.content.decode("utf-8", errors="ignore")
            else:
                return r.text
        print(f"⚠️ {url} returned {r.status_code}")
    except Exception as e:
        print(f"⚠️ Error fetching {url}: {e}")
    return None


# ===============================
#   Sitemap Parser
# ===============================
def get_sitemap_links(url):
    xml = fetch_url(url)
    if not xml:
        return []
    soup = BeautifulSoup(xml, "xml")
    subs = soup.find_all("sitemap")
    if subs:
        urls = []
        for s in subs:
            loc = s.find("loc")
            if loc:
                urls.extend(get_sitemap_links(loc.text.strip()))
        return urls
    return [loc.text.strip() for loc in soup.find_all("loc") if loc.text.strip()]


# ===============================
#   HTML → טקסט
# ===============================
def text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "header", "footer", "svg", "nav"]):
        t.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ===============================
#   Build Index — Rolling Resume
# ===============================
def build_index():
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)

    urls = []
    for sm in SITEMAPS:
        urls.extend(get_sitemap_links(sm))
    urls = list(dict.fromkeys(urls))

    if not urls:
        print("🚫 No URLs found in sitemap.")
        return

    print(f"🌐 {len(urls)} URLs total.")
    remaining = []
    for u in urls:
        if u not in cache:
            remaining.append(u)
        else:
            # בדוק אם התוכן השתנה
            pass

    print(f"🟡 {len(remaining)} new URLs to process.")

    index_summary = {"files": [], "total_chunks": 0}
    total_chunks = 0
    new_pages = 0

    start_time = time.time()
    MAX_RUN_TIME = 60 * 60  # שעה אחת

    for idx, url in enumerate(remaining, 1):
        if time.time() - start_time > MAX_RUN_TIME:
            print("⏹️ Stopping — reached 1 hour limit.")
            break

        html = fetch_url(url)
        if not html:
            continue

        text = text_from_html(html)
        if len(text) < 150:
            continue

        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if cache.get(url) == text_hash:
            continue

        chunk_size = int(os.getenv("MAX_CHUNK_TOKENS", "800")) * 4
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size) if len(text[i:i + chunk_size].strip()) > 50]
        if not chunks:
            continue

        ids = [f"{urlparse(url).path.strip('/') or 'index'}#chunk{i}" for i in range(len(chunks))]
        metas = [{"url": url} for _ in chunks]

        try:
            embs = embed_texts(chunks)
            collection.add(documents=chunks, embeddings=embs, metadatas=metas, ids=ids)
            cache[url] = text_hash
            total_chunks += len(chunks)
            new_pages += 1
            index_summary["files"].append({"url": url, "chunks": len(chunks)})
            print(f"[{idx}/{len(remaining)}] Indexed {url} ({len(chunks)} chunks)")
        except Exception as e:
            print(f"⚠️ Failed to index {url}: {e}")

        # שמירה תקופתית
        if idx % 10 == 0:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
                json.dump(index_summary, f, ensure_ascii=False, indent=2)
            print("💾 Progress saved.")

    # סיום
    index_summary["total_chunks"] = total_chunks
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(index_summary, f, ensure_ascii=False, indent=2)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"✅ Added {new_pages} new pages, total {total_chunks} chunks.")

if __name__ == "__main__":
    build_index()
