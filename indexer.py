# indexer.py - גרסה משודרגת עם resume, ריצה הדרגתית ויציבות גבוהה
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

try:
    collection = chroma_client.get_collection("shabaton_faq")
    print("✅ Loaded existing collection 'shabaton_faq'", flush=True)
except Exception:
    collection = chroma_client.create_collection("shabaton_faq")
    print("🆕 Created new collection 'shabaton_faq'", flush=True)


# ===============================
#   פונקציה ליצירת embeddings
# ===============================
def embed_texts(texts, retries=3):
    """יוצר embeddings עם retry והשהייה אוטומטית"""
    all_embeddings = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i + 50]
        for attempt in range(retries):
            try:
                res = openai_client.embeddings.create(input=batch, model=EMBED_MODEL)
                all_embeddings.extend([d.embedding for d in res.data])
                time.sleep(1.2)  # הפחת עומס על OpenAI
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"⚠️ OpenAI error (attempt {attempt+1}): {e}, sleeping {wait}s", flush=True)
                time.sleep(wait)
        else:
            print("❌ Failed to get embeddings after retries", flush=True)
    return all_embeddings


# ===============================
#   fetch_url — כולל gzip ו-bypass ל-403
# ===============================
def fetch_url(url, max_retries=3):
    ua_browser = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ua_google = "Googlebot/2.1 (+http://www.google.com/bot.html)"
    headers = {
        "User-Agent": ua_browser,
        "Accept": "application/xml,text/xml,text/html;q=0.9,*/*;q=0.8",
        "Accept-Language": "he,en-US;q=0.9,en;q=0.8",
    }

    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=25)
            print(f"🔎 GET {url} → {r.status_code}, {len(r.content)} bytes", flush=True)
            if r.status_code == 200:
                ce = r.headers.get("Content-Encoding", "").lower()
                content = r.content
                if "gzip" in ce:
                    try:
                        return gzip.decompress(content).decode("utf-8", errors="ignore")
                    except Exception:
                        print("⚠️ gzip header set but file not gzipped — decoding normally.", flush=True)
                        return content.decode("utf-8", errors="ignore")
                else:
                    return content.decode("utf-8", errors="ignore")

            elif r.status_code == 403:
                print(f"⚠️ 403 Forbidden for {url}, retrying as Googlebot...", flush=True)
                headers["User-Agent"] = ua_google
                continue

        except Exception as e:
            print(f"⚠️ Failed to fetch {url}: {e}", flush=True)
        time.sleep(1 + attempt)

    # ניסיון אחרון עם cloudscraper
    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper(browser={"custom": ua_browser})
        r = scraper.get(url, timeout=25)
        print(f"🧩 cloudscraper GET {url} → {r.status_code}, {len(r.content)} bytes", flush=True)
        if r.status_code == 200:
            return r.text
    except ModuleNotFoundError:
        pass
    except Exception as e:
        print(f"❌ cloudscraper error: {e}", flush=True)

    return None


# ===============================
#   שליפת קישורים מסייטמאפ
# ===============================
def get_sitemap_links(url):
    xml = fetch_url(url)
    if not xml:
        print(f"⚠️ No XML from {url}", flush=True)
        return []

    soup = BeautifulSoup(xml, "xml")
    sitemap_tags = soup.find_all("sitemap")
    if sitemap_tags:
        urls = []
        for sm in sitemap_tags:
            loc = sm.find("loc")
            if loc and loc.text.strip():
                sub = loc.text.strip()
                print(f"🗂 Found sub-sitemap: {sub}", flush=True)
                urls.extend(get_sitemap_links(sub))
        return urls

    locs = [loc.text.strip() for loc in soup.find_all("loc") if loc.text.strip()]
    return locs


# ===============================
#   ניקוי HTML לטקסט בלבד
# ===============================
def text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ===============================
#   בניית אינדקס (עם resume)
# ===============================
def build_index():
    cache = json.load(open(CACHE_FILE, "r", encoding="utf-8")) if os.path.exists(CACHE_FILE) else {}
    urls = []

    for sm in SITEMAPS:
        print(f"→ reading sitemap: {sm}", flush=True)
        found = get_sitemap_links(sm)
        print(f"🌍 Found {len(found)} URLs in {sm}", flush=True)
        urls.extend(found)

    urls = list(dict.fromkeys(urls))
    if not urls:
        print("🚫 No URLs found.", flush=True)
        return

    remaining = [u for u in urls if u not in cache]
    print(f"📦 {len(remaining)} pages to process (out of {len(urls)} total)", flush=True)

    index_summary = {"files": [], "total_chunks": 0}
    total_chunks = 0
    indexed = 0

    for idx, url in enumerate(remaining, start=1):
        html = fetch_url(url)
        if not html:
            continue
        text = text_from_html(html)
        if len(text) < 120:
            continue

        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if cache.get(url) == text_hash:
            continue

        max_chars = int(os.getenv("MAX_CHUNK_TOKENS", "800")) * 4
        chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars) if len(text[i:i + max_chars].strip()) > 50]
        if not chunks:
            continue

        ids = [f"{urlparse(url).path.strip('/') or 'index'}#chunk{i}" for i in range(len(chunks))]
        metas = [{"url": url} for _ in chunks]

        try:
            embs = embed_texts(chunks)
            collection.add(documents=chunks, embeddings=embs, metadatas=metas, ids=ids)
            indexed += 1
            total_chunks += len(chunks)
            index_summary["files"].append({"url": url, "chunks": len(chunks)})
            cache[url] = text_hash
            print(f"[{idx}/{len(remaining)}] [+] Indexed {url} ({len(chunks)} chunks)", flush=True)
        except Exception as e:
            print(f"⚠️ Failed to add chunks for {url}: {e}", flush=True)

        # שמירה כל 10 דפים כדי למנוע איבוד התקדמות
        if idx % 10 == 0:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
                json.dump(index_summary, f, ensure_ascii=False, indent=2)
            print(f"💾 Progress saved ({idx}/{len(remaining)})", flush=True)
            time.sleep(1)

    index_summary["total_chunks"] = total_chunks
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(index_summary, f, ensure_ascii=False, indent=2)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"✅ Indexed {indexed} new pages, total {total_chunks} chunks.", flush=True)


if __name__ == "__main__":
    build_index()
