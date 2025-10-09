import os
import json
import hashlib
import requests
import re
import smtplib
from email.mime.text import MIMEText
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

# === הגדרות מייל ===
EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM")
EMAIL_PASS = os.getenv("ALERT_EMAIL_PASS")
EMAIL_TO = os.getenv("ALERT_EMAIL_TO")

def send_email(subject, body):
    if not EMAIL_FROM or not EMAIL_TO or not EMAIL_PASS:
        print("⚠️ Email alert skipped (missing credentials)")
        return

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASS)
            server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())
        print(f"📧 Email alert sent to {EMAIL_TO}")
    except Exception as e:
        print(f"❌ Failed to send email alert: {e}")

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

# === בקשה עם User-Agent אמיתי כדי לעקוף חסימות ===
def fetch_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9,he;q=0.8",
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"⚠️ Failed to fetch {url}: {e}")
        return None

def get_sitemap_links(url):
    xml = fetch_url(url)
    if not xml:
        return []
    soup = BeautifulSoup(xml, "xml")
    return [loc.text.strip() for loc in soup.find_all("loc")]

def text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

# === טעינת cache והיסטוריה ===
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
    send_email("⚠️ Shabaton Indexing Failed", "No URLs found in any sitemap.")
    exit(0)

index_summary = {"pages": [], "total_chunks": 0}
changes = {"timestamp": datetime.utcnow().isoformat(), "new": [], "updated": [], "skipped": []}
new_pages = updated_pages = skipped_pages = 0

for url in urls:
    html = fetch_url(url)
    if not html:
        continue

    text = text_from_html(html)
    if len(text) < 100:
        continue

    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    prev = cache.get(url)

    if prev == text_hash:
        skipped_pages += 1
        changes["skipped"].append(url)
        continue

    if prev is None:
        new_pages += 1
        changes["new"].append(url)
    else:
        updated_pages += 1
        changes["updated"].append(url)

    max_chars = int(os.getenv("MAX_CHUNK_TOKENS", "800")) * 4
    chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars) if len(text[i:i + max_chars].strip()) > 50]
    ids = [f"{urlparse(url).path.strip('/') or 'index'}#chunk{i}" for i in range(len(chunks))]
    metas = [{"source": url} for _ in chunks]

    # נשמור תקציר
    index_summary = {
        "total_chunks": total_chunks,
        "files": indexed_files
    }

    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(index_summary, f, ensure_ascii=False, indent=2)

    print(f"✅ Indexing complete: {len(indexed_files)} files, {total_chunks} chunks total.")
    print(f"📄 Summary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    build_index()
