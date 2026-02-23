"""
collect_corpus_blog.py
======================
Step 1 of 2: Data Collection

Crawls "Around the World in 80 Cuisines" blog to:
  1. Auto-discover ALL cuisine category URLs from the homepage
  2. Filter to East Asian categories via keyword rules (no LLM needed here)
  3. Parse each article into section-level documents (Introduction + each <h3> section)
  4. Save raw sections to JSONL

Run:
    pip install requests beautifulsoup4
    python collect_corpus_blog.py

Output:
    data/raw_blog.jsonl
"""

import requests
from bs4 import BeautifulSoup
import json, re, time
from datetime import datetime, timezone
from pathlib import Path
import sys
import io

# ───────────────────────────── CONFIG ─────────────────────────────
BLOG_HOME   = "https://aroundtheworldin80cuisinesblog.wordpress.com/"
OUTPUT_FILE = Path("raw_blog.jsonl")
DELAY       = 1.0
HEADERS     = {"User-Agent": "Mozilla/5.0 (academic research bot)"}

# ─────────────── EAST ASIAN KEYWORD FILTER ────────────────────────
EA_INCLUDE = [
    "china", "chinese", "japan", "japanese",
    "korea", "korean", "taiwan", "taiwanese",
    "mongolia", "mongolian", "hong kong", "tibet",
]
EA_EXCLUDE = [
    "vietnam", "thai", "indonesia", "malaysia", "singapore",
    "philippines", "cambodia", "myanmar", "laos",
    "india", "pakistan", "bangladesh", "sri lanka",
    "central asia", "iran", "arab", "turkey",
    "africa", "europe", "america", "australia", "latin", "russia",
]

def is_east_asian(label: str) -> bool:
    t = label.lower()
    if any(kw in t for kw in EA_EXCLUDE):
        return False
    return any(kw in t for kw in EA_INCLUDE)

# ──────────────── DISCOVER CATEGORIES ─────────────────────────────

def discover_categories() -> list[dict]:
    """Scrape homepage sidebar for all cuisine category links."""
    print(f"[Discover] {BLOG_HOME}")
    resp = requests.get(BLOG_HOME, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    categories, seen = [], set()
    for a in soup.find_all("a", href=True):
        href  = a["href"]
        label = a.get_text(strip=True)
        if "/category/" in href and href not in seen and label:
            seen.add(href)
            categories.append({"label": label, "url": href})

    print(f"[Discover] {len(categories)} total categories found")
    return categories

# ──────────────── TEXT CLEANING ───────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ──────────────── FIND CANONICAL POST URL ─────────────────────────

POST_URL_RE = re.compile(
    r"https://aroundtheworldin80cuisinesblog\.wordpress\.com/\d{4}/\d{2}/\d{2}/[^/\"]+/"
)

def find_canonical_url(category_url: str) -> str:
    try:
        resp    = requests.get(category_url, headers=HEADERS, timeout=15)
        matches = POST_URL_RE.findall(resp.text)
        if matches:
            return matches[0]
    except Exception:
        pass
    return category_url

# ──────────────── PARSE ARTICLE INTO SECTIONS ─────────────────────

def parse_article(fetch_url: str, cuisine_tag: str) -> tuple[str, str, list[dict]]:
    """
    Fetch a category page and split its article into sections.

    Blog HTML structure:
      <p>          ← Introduction (text before first <h3>)
      <hr>         ← visual divider (ignored)
      <h3>         ← section heading
      <p>          ← section body
      ... repeats ...

    Returns (article_title, date_str, sections)
    """
    try:
        resp = requests.get(fetch_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [ERROR] {e}")
        return "", "", []

    soup = BeautifulSoup(resp.text, "html.parser")

    h1            = soup.find("h1", class_="entry-title") or soup.find("h1")
    article_title = h1.get_text(strip=True) if h1 else cuisine_tag

    date_str = ""
    t = soup.find("time", class_="entry-date") or soup.find("time")
    if t:
        date_str = t.get("datetime", t.get_text(strip=True))

    content = (
        soup.find("div", class_="entry-content") or
        soup.find("div", class_="post-content") or
        soup.find("article")
    )
    if not content:
        return article_title, date_str, []

    sections: list[dict] = []
    current_heading      = "Introduction"
    current_paras: list  = []

    for tag in content.children:
        if not hasattr(tag, "name"):
            continue
        if tag.name == "h3":
            body = "\n\n".join(
                p.get_text(strip=True) for p in current_paras
                if p.get_text(strip=True)
            )
            if body:
                sections.append({"section_title": current_heading,
                                  "text": clean_text(body)})
            current_heading = tag.get_text(strip=True)
            current_paras   = []
        elif tag.name in ("p", "ul", "ol", "blockquote"):
            current_paras.append(tag)

    # flush last section
    body = "\n\n".join(
        p.get_text(strip=True) for p in current_paras
        if p.get_text(strip=True)
    )
    if body:
        sections.append({"section_title": current_heading,
                         "text": clean_text(body)})

    return article_title, date_str, sections

# ──────────────── MAIN ────────────────────────────────────────────

def collect():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    collected_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    all_cats = discover_categories()
    ea_cats  = [c for c in all_cats if is_east_asian(c["label"])]

    print(f"\n[Filter]   Kept {len(ea_cats)} East Asian categories:")
    for c in ea_cats:
        print(f"             ✅  {c['label']:35s}  {c['url']}")

    counter = total_docs = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for cat in ea_cats:
            cuisine_name  = cat["label"]
            cat_url       = cat["url"]
            print(f"\n{'='*60}\n[Scrape] {cuisine_name}\n         {cat_url}")

            canonical_url = find_canonical_url(cat_url)
            time.sleep(DELAY)

            article_title, date_str, sections = parse_article(cat_url, cuisine_name)
            if not sections:
                print("  [SKIP] No sections found")
                continue

            print(f"  {len(sections)} sections in '{article_title}'")
            for sec in sections:
                if len(sec["text"]) < 80:
                    continue

                counter += 1
                doc = {
                    "doc_id":   f"blog_{counter:04d}",
                    "source":   "Around the World in 80 Cuisines Blog",
                    "title":    f"{article_title} > {sec['section_title']}",
                    "text":     sec["text"],
                    "url":      canonical_url,
                    "metadata": {
                        "article_title":  article_title,
                        "section_title":  sec["section_title"],
                        "cuisine":        cuisine_name,
                        "date_published": date_str,
                        "collected_at":   collected_at,
                    },
                }
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                print(f"    ✅  [{doc['doc_id']}] {sec['section_title'][:55]}")
                total_docs += 1

            time.sleep(DELAY)

    print(f"\n{'='*60}")
    print(f"✅  Done. {total_docs} section-documents → {OUTPUT_FILE}")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    collect()
