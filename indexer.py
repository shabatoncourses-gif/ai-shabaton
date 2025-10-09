# indexer.py - גרסה משודרגת עם bypass ל-403 ו-API החדש של OpenAI
import os, json, hashlib, time, requests, re, gzip
from io import BytesIO
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from datetime import datetime
import chromadb
from openai import OpenAI

load_dotenv()

SITEMAPS = [
    os.getenv("SITEMAP_URL_1", "https://www.shabaton.online/sitemap.xml"),
    os.getenv("SITEMAP_URL_2", "https://www.morim.boutique/sitemap.xml"),
]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/index")
SUMMARY_FILE = os.path.join("data", "index_summary.json")
CACHE_FILE = os.path.join("data", "index_cache.json")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ Missing OPENAI_API_KEY")

os.makedirs("data", exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# יצירת פונקציית embedding מותאמת לגרסה החדשה
client = OpenAI(api_key=OPENAI_API_KEY)

def embed_texts(texts):
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]

# יצירת collection אם לא קיים
client_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
try:
    collection = client_chroma.get_collection("shabaton_faq")
    print("✅ Loaded existing collection 'shabaton_faq'")
except Exception:
    collection = client_chroma.create_collection("shabaton_faq")
    print("🆕 Created new collection 'shabaton_faq'")

def fetch_url(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 403:
            headers["User-Agent"] = "Googlebot/2.1 (+http://www.google.com/bot.html)"
            r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 200:
            if "gzip" in r.headers.get("Content-Encoding", "") or url.endswith(".gz"):
                return gzip.decompress(r.content).decode("utf-8", errors="ignore")
            return r.text
    except Exception as e:
        print(f"⚠️ Failed to fetch {url}: {e}")
    return None

def get_sitemap_links(url):
    xml = fetch_url(url)
    if not xml:
        print(f"⚠️ No XML from {url}")
        return []
    soup = BeautifulSoup(xml, "xml")
    sitemap_tags = soup.find_all("sitemap")
    if sitemap_tags:
        urls = []
        for sm in sitemap_tags:
            loc = sm.find("loc")
            if loc:
                urls.extend(get_sitemap_links(loc.text.strip()))
        return urls
    return [loc.text.strip() for loc in soup.find_all("loc") if loc.text.strip()]

def text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def build_index():
    cache = json.load(open(CACHE_FILE, "r", encoding="utf-8")) if os.path.exists(CACHE_FILE) else {}
    urls = []
    for sm in SITEMAPS:
        found = get_sitemap_links(sm)
        print(f"🌍 Found {len(found)} URLs in {sm}")
        urls.extend(found)
    urls = list(dict.fromkeys(urls))
    if not urls:
        print("🚫 No URLs found.")
        return

    index_summary = {"files": [], "total_chunks": 0}
    total_chunks = 0
    for url in urls:
        html = fetch_url(url)
        if not html:
            continue
        text = text_from_html(html)
        if len(text) < 120:
            continue
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if cache.get(url) == text_hash:
            continue

        chunks = [text[i:i + 3200] for i in range(0, len(text), 3200)]
        embeddings = embed_texts(chunks)
        ids = [f"{urlparse(url).path.strip('/') or 'index'}#chunk{i}" for i in range(len(chunks))]
        metas = [{"url": url} for _ in chunks]
        collection.add(documents=chunks, metadatas=metas, ids=ids, embeddings=embeddings)

        cache[url] = text_hash
        index_summary["files"].append({"url": url, "chunks": len(chunks)})
        total_chunks += len(chunks)
        print(f"✅ Indexed {url} ({len(chunks)} chunks)")

    index_summary["total_chunks"] = total_chunks
    json.dump(index_summary, open(SUMMARY_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(cache, open(CACHE_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"🎯 Indexed {len(index_summary['files'])} pages, total {total_chunks} chunks.")

if __name__ == "__main__":
    build_index()
