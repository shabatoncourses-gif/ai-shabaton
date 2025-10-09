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
    """מוריד URL ומנסה לעקוף 403 ע"י התחזות לגוגלבוט"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.google.com/",
    }

    try:
        r = requests.get(url, headers=headers, timeout=30)
        print(f"🔎 GET {url} → {r.status_code}, {len(r.content)} bytes")
        if r.status_code == 403:
            print(f"⚠️ Got 403 for {url}, retrying as Googlebot...")
            headers["User-Agent"] = "Googlebot/2.1 (+http://www.google.com/bot.html)"
            r = requests.get(url, headers=headers, timeout=30)
            print(f"🕷 Retried as Googlebot → {r.status_code}, {len(r.content)} bytes")

        r.raise_for_status()

        # טיפול ב־gzip
        if url.endswith(".gz") or r.headers.get("Content-Encoding") == "gzip":
            return gzip.decompress(r.content).decode("utf-8", errors="ignore")
        return r.text
    except Exception as e:
        print(f"❌ fetch_url failed for {url}: {e}")
        return None


def get_sitemap_links(url):
    """תומך גם ב-sitemap index וגם בקובצי .gz"""
    xml = fetch_url(url)
    if not xml:
        print(f"⚠️ No XML content from {url}")
        return []

    soup = BeautifulSoup(xml, "xml")
    if soup.find("sitemap"):
        # Sitemap index – לרדת רמה אחת
        urls = []
        for sm in soup.find_all("sitemap"):
            loc = sm.find("loc")
            if loc and loc.text.strip():
                sub = loc.text.strip()
                print(f"🗂 Found sub-sitemap: {sub}")
                urls.extend(get_sitemap_links(sub))
        return urls

    locs = [loc.text.strip() for loc in soup.find_all("loc") if loc.text.strip()]
    return locs


def text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def build_index():
    cache = json.load(open(CACHE_FILE, "r", encoding="utf-8")) if os.path.exists(CACHE_FILE) else {}
    urls = []

    for sm in SITEMAPS:
        found = get_sitemap_links(sm)
        print(f"🌍 Found {len(found)} URLs in {sm}")
        urls.extend(found)

    urls = list(set(urls))
    if not urls:
        print("🚫 No URLs found — likely sitemap is blocked (403 or empty).")
        print("💡 Try opening the sitemap manually in browser to confirm.")
        return

    index_summary = {"files": [], "total_chunks": 0}
    for url in urls[:5]:  # 🔧 לבדיקה — מגביל ל-5 עמודים
        html = fetch_url(url)
        if not html:
            continue
        text = text_from_html(html)
        if len(text) < 100:
            continue

        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if cache.get(url) == text_hash:
            continue

        max_chars = 3200
        chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
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


if __name__ == "__main__":
    build_index()
