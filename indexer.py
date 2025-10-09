# indexer.py - גרסה משודרגת עם bypass ל-403 (cloudscraper fallback)
import os
import json
import hashlib
import time
import requests
import re
import gzip
from io import BytesIO
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime

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

# לוודא סביבה
if not OPENAI_API_KEY:
    raise RuntimeError("❌ Missing OPENAI_API_KEY")

os.makedirs("data", exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# חיבור ל־Chroma
client = chromadb.PersistentClient(path=CHROMA_DIR)
ef = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBED_MODEL)
try:
    collection = client.get_collection("shabaton_faq")
    print("✅ Loaded existing collection 'shabaton_faq'")
except Exception:
    collection = client.create_collection("shabaton_faq", embedding_function=ef)
    print("🆕 Created new collection 'shabaton_faq'")

# --------------------------
# fetch_url משודרג: ניסיונות מרובים + cloudscraper אם קיים
# --------------------------
def _requests_fetch(url, headers=None, timeout=30):
    """פשוט GET עם requests, מחזיר tuple(status_code, content_bytes, headers) או (None, None, None)"""
    try:
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        return r.status_code, r.content, r.headers
    except Exception as e:
        print(f"❌ requests.get error for {url}: {e}")
        return None, None, None

def fetch_url(url, max_retries=2):
    """
    ניסיונות: 
      1) requests עם User-Agent דפדפן
      2) requests עם Googlebot UA (אם 403)
      3) אם מותקן -> cloudscraper (מנסים לעבור JS challenge)
    מחזיר טקסט (utf-8) או None.
    """
    # HEADERS בסיסיים
    ua_browser = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ua_google = "Googlebot/2.1 (+http://www.google.com/bot.html)"
    headers = {
        "User-Agent": ua_browser,
        "Accept": "application/xml,text/xml,text/html;q=0.9,*/*;q=0.8",
        "Accept-Language": "he,en-US;q=0.9,en;q=0.8",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
    }

    # ראשון: ניסיון פשוט
    for attempt in range(max_retries):
        code, content, resp_headers = _requests_fetch(url, headers=headers)
        if code is None:
            time.sleep(1 + attempt)
            continue

        print(f"🔎 GET {url} → {code}, {len(content) if content else 0} bytes (attempt {attempt+1})")

        # אם קיבלנו 200 — להחזיר תוכן (להתחשב בדחיסה)
        if code == 200:
            try:
                # gzip בדוק
                ce = resp_headers.get("Content-Encoding", "") if resp_headers else ""
                if url.endswith(".gz") or "gzip" in ce.lower():
                    try:
                        return gzip.decompress(content).decode("utf-8", errors="ignore")
                    except Exception:
                        # נסיון fallback
                        try:
                            return BytesIO(content).read().decode("utf-8", errors="ignore")
                        except Exception:
                            return None
                else:
                    return content.decode("utf-8", errors="ignore")
            except Exception as e:
                print(f"⚠️ decode error: {e}")
                return None

        # אם קיבלנו 403 — ננסה כ-Googlebot לפני cloudscraper
        if code == 403:
            print(f"⚠️ Received 403 for {url} on attempt {attempt+1}; will retry with Googlebot UA")
            headers["User-Agent"] = ua_google
            code2, content2, headers2 = _requests_fetch(url, headers=headers)
            print(f"🕷 Retried as Googlebot → {code2}, {len(content2) if content2 else 0} bytes")
            if code2 == 200:
                try:
                    ce = headers2.get("Content-Encoding", "") if headers2 else ""
                    if url.endswith(".gz") or "gzip" in ce.lower():
                        return gzip.decompress(content2).decode("utf-8", errors="ignore")
                    return content2.decode("utf-8", errors="ignore")
                except Exception:
                    return None
            # else ננסה שוב בלולאה או נמשיך לניסיון הבא

        # אם מצב אחר (5xx וכו') נחכה וננסה שוב
        time.sleep(1 + attempt)

    # ניסיון עם cloudscraper אם מותקן
    try:
        import cloudscraper
        print("🧰 cloudscraper available — trying to bypass JS challenge / Cloudflare")
        scraper = cloudscraper.create_scraper(
            browser={
                "custom": ua_browser
            }
        )
        r = scraper.get(url, timeout=30)
        print(f"⚙️ cloudscraper GET {url} → {r.status_code}, {len(r.content) if r.content else 0} bytes")
        if r.status_code == 200:
            # טיפ: אם gzip
            ce = r.headers.get("Content-Encoding", "")
            if url.endswith(".gz") or ("gzip" in (ce or "").lower()):
                return gzip.decompress(r.content).decode("utf-8", errors="ignore")
            return r.text
        else:
            print(f"⚠️ cloudscraper returned {r.status_code} for {url}")
    except ModuleNotFoundError:
        print("ℹ️ cloudscraper not installed — install it in requirements.txt if you need advanced bypass.")
    except Exception as e:
        print(f"❌ cloudscraper error: {e}")

    # סופית — כישלון
    return None

# --------------------------
# get_sitemap_links תומך גם ב־sitemap index וב־.gz
# --------------------------
def get_sitemap_links(url):
    xml = fetch_url(url)
    if not xml:
        print(f"⚠️ No XML content from {url}")
        return []

    # הדפסת קטע ראשון למטרות דיבאג
    print("---- sitemap head (first 600 chars) ----")
    print(xml[:600].replace("\n", " "))
    print("----------------------------------------")

    soup = BeautifulSoup(xml, "xml")
    # אם זה sitemap index — יורדים רמה
    sitemap_tags = soup.find_all("sitemap")
    if sitemap_tags:
        urls = []
        for sm in sitemap_tags:
            loc = sm.find("loc")
            if loc and loc.text.strip():
                sub = loc.text.strip()
                print(f"🗂 Found sub-sitemap: {sub}")
                urls.extend(get_sitemap_links(sub))
        return urls

    # אחרת — שליפת כל ה־loc
    locs = [loc.text.strip() for loc in soup.find_all("loc") if loc.text.strip()]
    return locs

# --------------------------
# ניקוי HTML וטקסט
# --------------------------
def text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

# --------------------------
# build_index
# --------------------------
def build_index():
    cache = json.load(open(CACHE_FILE, "r", encoding="utf-8")) if os.path.exists(CACHE_FILE) else {}
    urls = []
    for sm in SITEMAPS:
        print(f"→ reading sitemap: {sm}")
        found = get_sitemap_links(sm)
        print(f"🌍 Found {len(found)} URLs in {sm}")
        urls.extend(found)

    urls = list(dict.fromkeys(urls))  # שמירה על סדר + הסרת כפילויות
    if not urls:
        print("🚫 No URLs found after sitemap attempts.")
        return

    index_summary = {"files": [], "total_chunks": 0}
    total_chunks = 0
    indexed = 0

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

        max_chars = int(os.getenv("MAX_CHUNK_TOKENS", "800")) * 4
        chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars) if len(text[i:i + max_chars].strip()) > 50]
        if not chunks:
            continue

        ids = [f"{urlparse(url).path.strip('/') or 'index'}#chunk{i}" for i in range(len(chunks))]
        metas = [{"url": url} for _ in chunks]

        try:
            collection.add(documents=chunks, metadatas=metas, ids=ids)
            indexed += 1
            total_chunks += len(chunks)
            index_summary["files"].append({"url": url, "chunks": len(chunks)})
            cache[url] = text_hash
            print(f"[+] Indexed {url} ({len(chunks)} chunks)")
        except Exception as e:
            print(f"⚠️ Failed to add chunks for {url}: {e}")

    index_summary["total_chunks"] = total_chunks

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(index_summary, f, ensure_ascii=False, indent=2)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"✅ Indexed {indexed} pages, total {total_chunks} chunks.")

if __name__ == "__main__":
    build_index()
