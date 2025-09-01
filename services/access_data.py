import asyncio
import aiohttp
import async_timeout
import re
import json
import time
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Any
import string
from bs4 import BeautifulSoup

BASE = "https://medlineplus.gov"
A2Z = "https://medlineplus.gov/encyclopedia.html" 
HDRS = {"User-Agent":"Mozilla/5.0 (symptom-crawler; research/educational use)"} # 서버에 요청을 보내는 값. 근데 이제 mozilla 5.0 같은 경우 대부분 호환됨


SYMPTOM_REGEX = re.compile(
    r"(symptom|signs?\s+and\s+symptoms?|what\s+are\s+the\s+symptoms|clinical\s+features?)",
    re.I
)

CUE_WEIGHTS = [
    (re.compile(r"\bmost common\b", re.I), 1.5),
    (re.compile(r"\bmain symptoms?\b", re.I), 1.3),
    (re.compile(r"\bcan include\b", re.I), 0.7),
    (re.compile(r"\bmay\b", re.I), 0.6),
    (re.compile(r"\bother symptoms?\b", re.I), 0.6),
    (re.compile(r"\bmild\b|\bless common\b", re.I), 0.4),
    (re.compile(r"\bsevere\b|\bemergency\b|call 911|go to the er", re.I), 0.9),
    (re.compile(r"\bsymptoms", re.I), 0.8)
]

CONCURRENCY = 4

async def safe_get(session: aiohttp.ClientSession, url, retries = 2):

    for tries in range(retries+1):
        try:
            async with async_timeout.timeout(25):
                async with session.get(url, headers=HDRS) as r:
                    r.raise_for_status()
                    return await r.text()
        except Exception:
            if tries == retries:
                raise
            
            await asyncio.sleep(1.5 * (tries+1))

def letter_urls():

    urls = [urljoin(BASE,f"/ency/encyclopedia_{ch}.htm") for ch in string.ascii_uppercase]

    return urls

ARTICLE_PATH = re.compile(r"^/ency/article/\d{6}\.htm$")

def parse_article_links(letter_html, letter_url):
    soup = BeautifulSoup(letter_html, "html.parser")
    links = []
    for a in soup.select("li a[href]") :
        href = a.get("href", "")
        if not href:
            continue
        full = urljoin(letter_url, href)
        path = urlparse(full).path
        if ARTICLE_PATH.match(path):
            links.append(full)
    return list(dict.fromkeys(links))

def parse_symptom_blocks(article_html):

    soup = BeautifulSoup(article_html, "html.parser")

    title = soup.find("h1")
    title_text = title.get_text(" ", strip=True) if title else ""

    fulltext = soup.get_text(" ", strip=True)

    heads = soup.find_all(["h2","h3"])
    found_any = False
    symptom_items = []
    current_weight = 1.0

    for h in heads:
        head_text = h.get_text(" ", strip = True)
        if SYMPTOM_REGEX.search(head_text or ""):
            found_any = True

            for sib in h.find_all_next():
                if sib.name in ["h2","h3"]:
                    break

                if sib.name == "p":
                    p_txt = sib.get_text(" ", strip = True)
                    matched = False
                    for pat, wt in CUE_WEIGHTS:
                        if pat.search(p_txt or ""):
                            current_weight = wt
                            matched = True
                            break
                    if not matched:
                        current_weight = 1.0

                
                if sib.name in ["ul","ol"]:
                    for li in sib.find_all("li"):
                        t = li.get_text(" ", strip=True)
                        if t and len(t) > 1:
                            symptom_items.append((t, current_weight, "list"))
                elif  sib.name == "table":
                    for td in sib.find_all(["td","th"]):
                        t = td.get_text(" ",strip = True)
                        if t and len(t) > 1:
                            symptom_items.append((t,current_weight, "table"))
                elif sib.name == "dl":
                    for d in sib.find_all(["dt", "dd"]):
                        t = d.get_text(" ", strip=True)
                        if t and len(t) >1:
                            symptom_items.append((t, current_weight, "dl"))
                
            
    return {
        "title": title_text,
        "found_symptom_section": found_any,
        "symptoms": [
            {"text": s, "weight": w, "from": src} for (s, w, src) in symptom_items
        ],
        "fulltext_len": len(fulltext),
    }


async def crawl_all(limit_letters = None, limit_articles_per_letter = None):
    results = []
    seen_articles = set()
    sem = asyncio.Semaphore(CONCURRENCY)

    letter_links = letter_urls()

    if limit_letters is not None:
            letter_links = letter_links[:limit_letters]

    async with aiohttp.ClientSession() as session:
            async def handle_article(url):
                async with sem:
                    if url in seen_articles:
                        return
                    seen_articles.add(url)
                    try:
                        html = await safe_get(session, url)
                        parsed = parse_symptom_blocks(html)
                        parsed["url"] = url
                        parsed["level"] = "primary"
                        results.append(parsed)
                        print(f"[ok] {url} found_symptom_section={parsed['found_symptom_section']}")

                    except Exception as e:
                        print("article failed")

            for letter_url in letter_links:
                try: 
                    letter_html = await safe_get(session, letter_url)
                except Exception as e:
                    print(f"[letter fectch failed]")   
                    continue
                
                article_links = parse_article_links(letter_html, letter_url)
                print(f"[letter] {letter_url} -> {len(article_links)} article links")

                unique_links = [u for u in dict.fromkeys(article_links) if u not in seen_articles]
                await asyncio.gather(*(handle_article(u) for u in unique_links))
            
    return results            

if __name__ == "__main__":
    t0 = time.time()
    # 테스트 시: 알파벳 2개, 각 10개 기사만 -> 전체로 돌릴 땐 None
    data = asyncio.run(crawl_all(limit_letters=None, limit_articles_per_letter=None))
    print(f"Collected {len(data)} pages in {time.time() - t0:.1f}s")

    # JSON 저장(각 페이지 기준)
    with open("ency_symptoms.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 질병명(title) 기준으로 합쳐보고 싶으면 아래처럼 후처리 가능
    by_title: Dict[str, Dict[str, float]] = {}
    for row in data:
        title = row.get("title") or "(untitled)"
        by_title.setdefault(title, {})
        for s in row.get("symptoms", []):
            txt = s["text"]
            w = float(s["weight"])
            # 같은 텍스트가 여러 곳에서 나오면 더 높은 가중치를 유지(또는 누적합으로 바꿔도 됨)
            by_title[title][txt] = max(by_title[title].get(txt, 0.0), w)

    with open("disease_symptoms_merged.json", "w", encoding="utf-8") as f:
        json.dump(by_title, f, ensure_ascii=False, indent=2)

    print("Saved: ency_symptoms.json, disease_symptoms_merged.json")
                    

