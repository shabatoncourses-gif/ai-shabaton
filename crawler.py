import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# הגדרות בסיס
BASE_URL = "https://example.com"   # ← שנה לכתובת שברצונך לסרוק
SAVE_DIR = "data"                  # ← שם התיקייה לשמירת הקבצים
visited = set()


def clean_text(text):
    """מנקה טקסט מרווחים מיותרים"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def fetch(url):
    """מביא תוכן HTML מכתובת"""
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"[!] fetch error {url}: {e}")
        return ""


def extract_links(html, base):
    """שולף קישורים מאותו דומיין בלבד"""
    soup = BeautifulSoup(html, 'html.parser')
    links = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        full = urljoin(base, href)
        parsed = urlparse(full)
        if parsed.netloc == urlparse(BASE_URL).netloc:
            clean = full.split('#')[0].split('?')[0]
            links.add(clean)
    return links


def text_from_html(html):
    """שולף טקסט נקי מתוך HTML"""
    soup = BeautifulSoup(html, 'html.parser')
    for s in soup(['script', 'style', 'noscript']):
        s.decompose()
    text = soup.get_text(separator='\n')
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def crawl(url):
    """סורק אתרים החל מכתובת בסיס"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    to_visit = set([url])

    while to_visit:
        u = to_visit.pop()
        if u in visited:
            continue

        visited.add(u)
        print(f"[+] Crawling {u}")

        html = fetch(u)
        if not html:
            continue

        text = text_from_html(html)
        if not text.strip():
            continue

        # שם הקובץ לפי ה־path
        filename = urlparse(u).path.strip('/') or 'index'
        filename = filename.replace('/', '_')
        filepath = os.path.join(SAVE_DIR, f"{filename}.txt")

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"    ↳ Saved: {filepath}")
        except Exception as e:
            print(f"    [!] Save error: {e}")

        # הוספת קישורים חדשים לרשימת הסריקה
        links = extract_links(html, u)
        for l in links:
            if l not in visited:
                to_visit.add(l)

        # הפסקה קטנה כדי לא להעמיס על השרת
        time.sleep(0.1)


if __name__ == '__main__':
    print(f"Starting crawl from {BASE_URL}")
    crawl(BASE_URL)
    print("Done.")
