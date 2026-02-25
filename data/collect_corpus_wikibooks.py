import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

WIKI_API = "https://en.wikibooks.org/w/api.php"
BASE = "https://en.wikibooks.org/wiki/"
SEED = "Cookbook:East_Asian_cuisines"

OUT_JSONL = Path("corpus_wikibooks_east_asia.jsonl")
OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

DELAY = 0.4
HEADERS = {"User-Agent": "EastAsianRecipesCrawler/1.0 (contact: your_email@example.com)"}
session = requests.Session()
session.headers.update(HEADERS)


def clean_text(s: str) -> str:
    s = re.sub(r"\[\s*\d+\s*\]", "", s)  # remove [1]
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    #  只取正文区域
    content = soup.find("div", class_="mw-parser-output")
    if not content:
        content = soup

    #  删除常见非正文元素
    for tag in content.select(
        "table, style, script, sup, noscript, "
        ".navbox, .vertical-navbox, .toc, "
        ".metadata, .noprint, "
        ".mw-editsection"
    ):
        tag.decompose()

    text = content.get_text("\n")

    # 清洗
    text = re.sub(r"\[\s*\d+\s*\]", "", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()


def parse_page_html(page_title: str) -> str:
    """action=parse 渲染页面并返回 HTML"""
    params = {
        "action": "parse",
        "format": "json",
        "page": page_title,
        "prop": "text",
        "formatversion": 2,
    }
    r = session.get(WIKI_API, params=params, timeout=30)
    r.raise_for_status()
    return r.json()["parse"]["text"]


def list_category_members(category_title: str) -> list[str]:
    """Category -> all Cookbook:* members (do NOT restrict cmnamespace)"""
    titles = []
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": category_title,   # e.g. 'Category:Chinese recipes'
        "cmlimit": "max",
    }

    while True:
        r = session.get(WIKI_API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        for item in (data.get("query", {}).get("categorymembers") or []):
            t = item.get("title")
            if t and t.startswith("Cookbook:"):
                titles.append(t)

        cont = data.get("continue")
        if not cont:
            break
        params.update(cont)
        time.sleep(DELAY)

    # unique keep order
    seen, out = set(), []
    for t in titles:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def fetch_intro(title: str) -> str:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "titles": title,
    }
    r = session.get(WIKI_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        return (page.get("extract") or "").strip()
    return ""

import re

def strip_cookbook_header_line(text: str) -> str:
    """
    Cookbook | Recipes | Ingredients | Equipment | Techniques | ...
    """
    pattern = re.compile(
         r"^\s*Cookbook\s*\|\s*Recipes\s*\|\s*Ingredients\s*\|\s*Equipment\s*\|.*?(?:\n|$)",
        flags=re.IGNORECASE
    )
    text = re.sub(pattern, "", text, count=1)
    pattern2 = re.compile(
        r"^\s*Cookbook\s*\|.*?(?:\n|$)",
        flags=re.IGNORECASE
    )
    text = re.sub(pattern2, "", text, count=1)

    return text.lstrip()

def fetch_page_text(title: str) -> str:
    """用 extracts 拉全文纯文本（比 parse->html 更干净）"""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "redirects": 1,
        # exchars 不设，默认就是尽可能给全文
        "titles": title,
    }
    r = session.get(WIKI_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        text = (page.get("extract") or "").strip()
        return clean_text(text)
    return ""


def category_href_to_title(href: str) -> str | None:
    """
    /wiki/Category:Chinese_recipes  -> Category:Chinese recipes
    /wiki/Category:Chinese%20recipes -> Category:Chinese recipes
    """
    if not href.startswith("/wiki/Category:"):
        return None
    raw = href[len("/wiki/"):]  # Category:Chinese_recipes...
    raw = raw.split("#", 1)[0].split("?", 1)[0]
    # url underscore -> space
    raw = raw.replace("_", " ")
    # percent decode (minimal)
    raw = re.sub(r"%([0-9A-Fa-f]{2})", lambda m: bytes([int(m.group(1), 16)]).decode("utf-8", "ignore"), raw)
    if not raw.lower().endswith(" recipes"):
        return None
    return raw


def main():
    # Step 1: parse seed html
    seed_html = parse_page_html(SEED)
    soup = BeautifulSoup(seed_html, "html.parser")

    # Step 2: extract Category:*_recipes links from HTML
    cats = []
    seen = set()
    for a in soup.select("a[href^='/wiki/Category:']"):
        href = a.get("href", "")
        cat_title = category_href_to_title(href)
        if cat_title and cat_title not in seen:
            seen.add(cat_title)
            cats.append(cat_title)

    # 如果 seed 页确实没直接给 Category 链接（极少数情况），给一个兜底：从文本猜 4 大类
    if not cats:
        fallback = [
            "Category:Chinese recipes",
            "Category:Japanese recipes",
            "Category:Korean recipes",
            "Category:Taiwanese recipes",
        ]
        cats = fallback
        print("[WARN] Seed HTML did not expose Category:* recipes links; using fallback 4 categories.")

    print(f"[Seed] {SEED} => {BASE}{SEED.replace(' ', '_')}")
    print(f"[Found recipe categories] {len(cats)}")
    for c in cats:
        print("  -", c, "=>", BASE + c.replace(" ", "_"))

    # Step 3+4: category -> recipes -> fetch page -> write jsonl
    doc_id = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for category in cats:
            recipes = list_category_members(category)
            print(f"\n[Category] {category} -> {len(recipes)} Cookbook:* pages")

            for recipe_title in recipes:
                try:
                    intro = fetch_intro(recipe_title)
                    text = fetch_page_text(recipe_title)
                    text = strip_cookbook_header_line(text)
                    # 过滤太短的异常页
                    if len(text) < 200:
                        continue

                    doc_id += 1
                    row = {
                        "doc_id": f"wikibooks_{doc_id:04d}",
                        "source": "Wikibooks",
                        "title": recipe_title,
                        "text": text,
                        "url": BASE + recipe_title.replace(" ", "_"),
                        "metadata": {
                            "seed": SEED,
                            "category": category,
                            "category_url": BASE + category.replace(" ", "_"),
                            "intro": intro,
                        },
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"   {recipe_title}")

                except Exception as e:
                    print(f"  [SKIP] {recipe_title}: {e}")

                time.sleep(DELAY)

    print(f"\n Done. Wrote {doc_id} docs to {OUT_JSONL.resolve()}")


if __name__ == "__main__":
    main()