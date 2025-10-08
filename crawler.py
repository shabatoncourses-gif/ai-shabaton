import requests, re, os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

BASE_URL = "https://www.shabaton.online/"
SAVE_DIR = "data/pages"
os.makedirs(SAVE_DIR, exist_ok=True)
visited = set()

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def crawl(url):
    if url in visited or not url.startswith(BASE_URL):
        return
    visited.add(url)
    print("📄", url)
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        text = clean_text(soup.get_text(separator=' '))
        filename = urlparse(url).path.replace("/", "_") or "index"
        with open(f"{SAVE_DIR}/{filename}.txt", "w", encoding="utf-8") as f:
            f.write(text)
        # חיפוש קישורים פנימיים
        for link in soup.find_all("a", href=True):
            next_url = urljoin(BASE_URL, link["href"])
            if BASE_URL in next_url and "#" not in next_url:
                crawl(next_url)
    except Exception as e:
        print("❌", e)

if __name__ == "__main__":
    crawl(BASE_URL)
