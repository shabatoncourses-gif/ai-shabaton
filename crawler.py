import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import time

BASE_URL = "https://example.com"
SAVE_DIR = "data"
visited = set()


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def fetch(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"fetch error {url}: {e}")
        return ""


def extract_links(html, base):
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
    soup = BeautifulSoup(html, 'html.parser')
    for s in soup(['script', 'style', 'noscript']):
        s.decompose()
    text = soup.get_text(separator='\n')
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def crawl(url):
    to_visit = set([url])
    while to_visit:
        u = to_visit.pop()
        if u in visited:
            continue
        visited.add(u)
        print('Crawling', u)
        html = fetch(u)
        if not html:
            continue
        text = text_from_html(html)
        filename = urlparse(u).path.strip('/') or 'index'
        filename = filename.replace('/', '_')
        os.makedirs(SAVE_DIR, exist_ok=True)
        with open(os.path.join(SAVE_DIR, filename + '.txt'), 'w', encoding='utf-8') as f:
            f.write(text)
        links = extract_links(html, u)
        for l in links:
            if l not in visited:
                to_visit.add(l)
        time.sleep(0.1)


if __name__ == '__main__':
    crawl(BASE_URL)
