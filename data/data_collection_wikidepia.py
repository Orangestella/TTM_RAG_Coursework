"""
Wikipedia East Asian Cuisine Scraper for RAG Corpus
====================================================
Uses wikipedia-api to crawl East Asian cuisine pages from Wikipedia.

Features:
  - Automatic retry on timeout/network errors (max 3 retries with backoff)
  - Incremental save: writes each record to JSONL immediately (no data loss on crash)
  - Resume support: reads existing output file to skip already-crawled pages
  - Graceful Ctrl+C handling: prints summary before exiting

Usage:
    pip install wikipedia-api
    python scrape_east_asian_cuisine.py

    # If interrupted, just re-run — it will resume from where it left off.
"""

import json
import re
import time
import signal
import sys
import logging
from pathlib import Path
from datetime import datetime

import wikipediaapi
import requests.exceptions

# ============================================================
# CONFIG
# ============================================================

USER_AGENT = "EastAsianCuisineRAG/1.0 (university coursework; contact@example.com)"
OUTPUT_FILE = Path("data/corpus_wikipedia.jsonl")

# --- Retry config ---
MAX_RETRIES = 3               # retries per page
RETRY_BACKOFF = [5, 15, 30]   # seconds to wait on 1st, 2nd, 3rd retry
TIMEOUT = 30                  # request timeout in seconds

# --- Crawl config ---
CRAWL_DEPTH = 1
MAX_PAGES = 400
MAX_LINKS_PER_PAGE = 15
DELAY = 0.5                   # politeness delay between successful requests
MIN_TEXT_LENGTH = 500

# ---------- Seed pages ----------
SEED_PAGES = [
    # --- Overview ---
    "East Asian cuisine",
    "Asian cuisine",
    "List of Asian cuisines",
    # --- National / regional cuisines ---
    "Chinese cuisine",
    "Japanese cuisine",
    "Korean cuisine",
    "Taiwanese cuisine",
    "Mongolian cuisine",
    "Hong Kong cuisine",
    "Macanese cuisine",
    "Tibetan cuisine",
    # --- Chinese regional cuisines (Eight Great Traditions) ---
    "Shandong cuisine",
    "Cantonese cuisine",
    "Sichuan cuisine",
    "Hunan cuisine",
    "Fujian cuisine",
    "Zhejiang cuisine",
    "Jiangsu cuisine",
    "Anhui cuisine",
    # --- Japanese sub-cuisines ---
    "Kaiseki",
    "Izakaya",
    "Okinawan cuisine",
    "Japanese street food",
    # --- Korean sub-cuisines ---
    "Korean royal court cuisine",
    "Korean temple cuisine",
    "Korean street food",
    # --- Iconic dishes / food categories ---
    "Sushi",
    "Ramen",
    "Dim sum",
    "Dumpling",
    "Kimchi",
    "Tofu",
    "Soy sauce",
    "Miso",
    "Noodle",
    "Rice",
    "Wok",
    "Tempura",
    "Teriyaki",
    "Bibimbap",
    "Peking duck",
    "Kung Pao chicken",
    "Mapo tofu",
    "Hot pot",
    "Udon",
    "Soba",
    "Gyoza",
    "Bulgogi",
    "Tteok",
    "Japchae",
    "Char siu",
    "Congee",
    "Bao (food)",
    "Xiaolongbao",
    "Mooncake",
    "Matcha",
    "Sake",
    "Soju",
    "Baijiu",
    # --- Ingredients & techniques ---
    "Dashi",
    "Umami",
    "Fermented bean paste",
    "Sesame oil",
    "Rice vinegar",
    "Fish sauce",
    "Five-spice powder",
    "Star anise",
    "Szechuan pepper",
    "Wasabi",
    "Nori",
    "Shiitake",
    "Bok choy",
    "Daikon",
    "Edamame",
    "Mirin",
    "Gochujang",
    "Doenjang",
    # --- Dining culture ---
    "Chinese table manners",
    "Chopsticks",
    "Tea culture",
    "Chinese tea culture",
    "Japanese tea ceremony",
    "Korean tea",
]

# ---------- TWO-LAYER FILTER for depth-1 links ----------
#
# Logic:
#   1) BLACKLIST check first — reject anything about non-East-Asian regions
#   2) Must contain at least one FOOD keyword (so "Taiwanese indigenous peoples" is rejected)
#   3) If title has "cuisine" but no East Asian region qualifier, reject it
#      (so "Georgian cuisine" is rejected, but "Cantonese cuisine" passes)

# Layer 1: Hard blacklist — reject if title contains any of these
NON_EA_BLACKLIST = [
    # Other Asian regions
    "indian", "south asian", "southeast asian", "central asian", "west asian",
    "thai", "vietnamese", "indonesian", "malaysian", "filipino", "burmese",
    "cambodian", "laotian", "singaporean", "brunei",
    "afghan", "pakistani", "bangladeshi", "sri lankan", "nepali", "bhutanese",
    "persian", "iranian", "turkish", "arab", "lebanese", "israeli", "syrian",
    "iraqi", "saudi", "emirati", "uzbek", "kazakh", "tajik", "turkmen",
    # Non-Asian
    "georgian", "armenian", "azerbaijani",
    "european", "french", "italian", "spanish", "german", "british", "greek",
    "portuguese", "dutch", "belgian", "swiss", "austrian", "scandinavian",
    "russian", "polish", "hungarian", "czech", "romanian", "ukrainian",
    "african", "ethiopian", "moroccan", "egyptian", "nigerian", "south african",
    "american", "mexican", "brazilian", "caribbean", "peruvian", "colombian",
    "australian", "new zealand",
    # Non-food topics that slip through region keywords
    "indigenous peoples", "ethnic group", "language", "dialect", "province",
    "prefecture", "dynasty", "empire", "war", "battle", "military",
    "politics", "government", "election", "president", "minister",
    "geography", "mountain", "river", "island", "city", "population",
    "religion", "buddhis", "confuci", "shintoism", "taoism",
    "film", "movie", "television", "anime", "manga", "music", "sport",
    "football", "baseball", "basketball", "olympic",
    "university", "school", "education",
    "airport", "railway", "highway",
]

# Layer 2: Must contain at least one food-related keyword
FOOD_KEYWORDS = [
    # Food types
    "cuisine", "food", "dish", "recipe", "cooking", "cook", "culinary",
    "noodle", "rice", "soup", "broth", "stew", "porridge", "congee",
    "tea", "sauce", "paste", "spice", "seasoning", "condiment", "vinegar",
    "meat", "pork", "beef", "chicken", "lamb", "mutton", "poultry",
    "fish", "seafood", "shrimp", "crab", "squid", "octopus", "shellfish",
    "vegetable", "fruit", "bean", "legume", "grain", "wheat", "flour",
    "tofu", "soy", "miso", "nori", "seaweed", "mushroom", "shiitake",
    "dumpling", "bun", "bao", "bread", "cake", "pastry", "dessert", "sweet",
    "pickle", "ferment", "kimchi",
    # Cooking methods
    "grill", "fry", "stir-fry", "steam", "roast", "boil", "braise",
    "smoke", "cure", "dry", "bake",
    # Specific EA dishes/items
    "sushi", "sashimi", "ramen", "udon", "soba", "tempura", "teriyaki",
    "bibimbap", "bulgogi", "japchae", "tteok", "gochujang", "doenjang",
    "dim sum", "char siu", "wonton", "xiaolongbao", "peking duck",
    "mapo", "kung pao", "hot pot", "fondue",
    "dashi", "wasabi", "mirin", "sake", "soju", "baijiu", "matcha",
    "gyoza", "okonomiyaki", "takoyaki", "onigiri", "bento",
    # Dining culture
    "chopstick", "wok", "restaurant", "street food", "hawker",
    "banquet", "feast", "meal", "breakfast", "lunch", "dinner", "snack",
    "appetizer", "ingredient", "herb", "oil",
    "wine", "beer", "spirit", "liquor", "alcohol", "beverage", "drink",
    # General food terms
    "gastronom", "dietary", "nutrition", "delicacy", "staple food",
]

# Layer 3: "cuisine" in title requires an East Asian qualifier
# (prevents "Georgian cuisine", "Indian cuisine", etc. that survived blacklist)
EA_REGION_QUALIFIERS = [
    "east asian", "chinese", "japanese", "korean", "taiwanese",
    "mongolian", "tibetan", "hong kong", "macanese", "okinawan",
    # Chinese regional
    "cantonese", "sichuan", "szechuan", "shandong", "hunan",
    "fujian", "zhejiang", "jiangsu", "anhui", "yunnan", "guangdong",
    "hakka", "hokkien", "teochew", "shanghainese", "beijing",
    "dongbei", "xinjiang", "guizhou", "hubei", "henan", "shaanxi",
    "manchurian",
]

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ============================================================
# GRACEFUL SHUTDOWN on Ctrl+C
# ============================================================

_interrupted = False

def _signal_handler(sig, frame):
    global _interrupted
    _interrupted = True
    log.warning("\n⚠️  Ctrl+C detected — finishing current page then exiting...")

signal.signal(signal.SIGINT, _signal_handler)

# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(raw: str) -> str:
    """Clean Wikipedia extracted text for RAG corpus."""
    text = raw
    text = re.sub(r"\[(?:\d+|citation needed|clarification needed)\]", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(
        r"\n==\s*(References|External links|See also|Further reading|Notes|Bibliography)\s*==.*",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = text.replace("&amp;", "&").replace("&lt;", "<")
    text = text.replace("&gt;", ">").replace("&nbsp;", " ").replace("\xa0", " ")
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def is_cuisine_related(title: str) -> bool:
    """
    Two-layer filter for depth-1 linked pages:
      1. Blacklist: reject non-East-Asian or non-food topics
      2. Require at least one food keyword in title
      3. If "cuisine" is the only food keyword, require an EA region qualifier
    """
    t = title.lower()

    # Layer 1: blacklist
    if any(term in t for term in NON_EA_BLACKLIST):
        return False

    # Layer 2: must have a food keyword
    matched_food = [kw for kw in FOOD_KEYWORDS if kw in t]
    if not matched_food:
        return False

    # Layer 3: if the ONLY food match is "cuisine", require EA region qualifier
    # e.g. "Foobar cuisine" without a known EA region → reject
    if matched_food == ["cuisine"] or set(matched_food) == {"cuisine"}:
        if not any(region in t for region in EA_REGION_QUALIFIERS):
            return False

    return True


# ============================================================
# RETRY WRAPPER — catches ALL network-related exceptions
# ============================================================

NETWORK_ERRORS = (
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectTimeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
    ConnectionResetError,
    ConnectionAbortedError,
    TimeoutError,
    OSError,
)


def fetch_with_retry(page, attribute: str):
    """
    Safely access a page attribute (text, sections, categories, links, exists)
    with automatic retry on network errors.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            return getattr(page, attribute)
        except NETWORK_ERRORS as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                log.warning(
                    f"  ⚠️  {type(e).__name__} fetching '{attribute}' — "
                    f"retry {attempt + 1}/{MAX_RETRIES} in {wait}s..."
                )
                time.sleep(wait)
            else:
                log.error(f"  ❌ Failed after {MAX_RETRIES} retries: {type(e).__name__}")
                raise


# ============================================================
# RESUME SUPPORT — read existing output to skip done pages
# ============================================================

def load_existing(path: Path) -> tuple[set[str], int]:
    """
    Load already-crawled page titles and highest doc_id from existing JSONL.
    Returns (set_of_titles, max_doc_counter).
    """
    titles = set()
    max_id = 0
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    titles.add(rec["title"])
                    num = int(rec["doc_id"].split("_")[1])
                    max_id = max(max_id, num)
                except (json.JSONDecodeError, KeyError, ValueError, IndexError):
                    continue
    return titles, max_id


# ============================================================
# MAIN SCRAPER
# ============================================================

def scrape():
    global _interrupted

    # --- Resume check ---
    already_done, doc_counter = load_existing(OUTPUT_FILE)
    total_existing = len(already_done)

    if total_existing > 0:
        log.info(f"📂 Resuming: {total_existing} pages already in {OUTPUT_FILE}")
        log.info(f"   Will continue from wiki_{doc_counter + 1:04d}")
    else:
        log.info(f"🆕 Fresh crawl → {OUTPUT_FILE}")

    # --- Init wikipedia-api ---
    wiki = wikipediaapi.Wikipedia(
        user_agent=USER_AGENT,
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        timeout=TIMEOUT,
    )

    visited: set[str] = set(already_done)
    pages_this_run = 0
    failed_pages: list[str] = []

    # Open in APPEND mode — existing data preserved
    outfile = open(OUTPUT_FILE, "a", encoding="utf-8")

    def process_page(title: str, depth: int):
        nonlocal doc_counter, pages_this_run

        if _interrupted:
            return
        if title in visited:
            return
        if (total_existing + pages_this_run) >= MAX_PAGES:
            return

        visited.add(title)

        # 1) Check existence
        try:
            page = wiki.page(title)
            exists = fetch_with_retry(page, "exists")
        except Exception:
            log.error(f"  ❌ Skipping (network): {title}")
            failed_pages.append(title)
            return

        if not exists:
            log.info(f"  Page not found: {title}")
            return

        time.sleep(DELAY)

        # 2) Fetch text
        try:
            raw_text = fetch_with_retry(page, "text")
        except Exception:
            log.error(f"  ❌ Skipping (network on text): {title}")
            failed_pages.append(title)
            return

        if len(raw_text) < MIN_TEXT_LENGTH:
            log.info(f"  Skipping stub: {title} ({len(raw_text)} chars)")
            return

        cleaned = clean_text(raw_text)
        if len(cleaned) < MIN_TEXT_LENGTH:
            return

        doc_counter += 1
        pages_this_run += 1
        doc_id = f"wiki_{doc_counter:04d}"

        # 3) Sections (non-fatal if fails)
        section_titles = []
        try:
            sections = fetch_with_retry(page, "sections")
            def collect_sections(sects):
                for s in sects:
                    section_titles.append(s.title)
                    collect_sections(s.sections)
            collect_sections(sections)
        except Exception:
            log.warning(f"  ⚠️  Sections unavailable: {title}")

        # 4) Categories (non-fatal if fails)
        categories = []
        try:
            cats = fetch_with_retry(page, "categories")
            categories = [c.replace("Category:", "") for c in list(cats.keys())[:10]]
        except Exception:
            log.warning(f"  ⚠️  Categories unavailable: {title}")

        # 5) Build record & write immediately
        record = {
            "doc_id": doc_id,
            "source": "Wikipedia",
            "title": page.title,
            "text": cleaned,
            "url": page.fullurl,
            "metadata": {
                "sections": section_titles,
                "categories": categories,
                "text_length": len(cleaned),
                "word_count": len(cleaned.split()),
                "crawl_depth": depth,
                "date_collected": datetime.now().strftime("%Y-%m-%d"),
            },
        }
        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
        outfile.flush()  # <-- flush to disk immediately, crash-safe

        count = total_existing + pages_this_run
        log.info(
            f"  [{count:3d}/{MAX_PAGES}] depth={depth} "
            f"✅ {page.title} ({len(cleaned):,} chars, {len(cleaned.split()):,} words)"
        )

        # 6) Follow links (depth 0 → 1)
        if depth < CRAWL_DEPTH and not _interrupted:
            try:
                links = fetch_with_retry(page, "links")
            except Exception:
                log.warning(f"  ⚠️  Links unavailable: {title}")
                return

            followed = 0
            for link_title in sorted(links.keys()):
                if _interrupted:
                    break
                if followed >= MAX_LINKS_PER_PAGE:
                    break
                if link_title in visited:
                    continue
                if is_cuisine_related(link_title):
                    process_page(link_title, depth + 1)
                    followed += 1

    # =============================================
    # Main crawl loop
    # =============================================
    log.info("")
    log.info(f"=== Config: {len(SEED_PAGES)} seeds | depth={CRAWL_DEPTH} | max={MAX_PAGES} ===")
    log.info(f"    Timeout={TIMEOUT}s | Retries={MAX_RETRIES} | Delay={DELAY}s")
    log.info("")

    for i, title in enumerate(SEED_PAGES, 1):
        if _interrupted:
            break
        if (total_existing + pages_this_run) >= MAX_PAGES:
            log.info(f"✅ Reached MAX_PAGES={MAX_PAGES}, stopping.")
            break
        log.info(f"--- Seed {i}/{len(SEED_PAGES)}: {title} ---")
        process_page(title, depth=0)

    outfile.close()

    # =============================================
    # Final summary (read full file for stats)
    # =============================================
    all_records = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    total_chars = sum(r["metadata"]["text_length"] for r in all_records)
    total_words = sum(r["metadata"]["word_count"] for r in all_records)
    depth_0 = sum(1 for r in all_records if r["metadata"]["crawl_depth"] == 0)
    depth_1 = sum(1 for r in all_records if r["metadata"]["crawl_depth"] == 1)

    log.info("")
    log.info("=" * 60)
    log.info("CRAWL SUMMARY")
    log.info("=" * 60)
    log.info(f"  Total documents:    {len(all_records)}")
    log.info(f"    Seed (depth 0):   {depth_0}")
    log.info(f"    Linked (depth 1): {depth_1}")
    log.info(f"  Added this run:     {pages_this_run}")
    log.info(f"  Failed (network):   {len(failed_pages)}")
    log.info(f"  Total characters:   {total_chars:,}")
    log.info(f"  Total words:        {total_words:,}")
    log.info(f"  Avg words/doc:      {total_words // max(len(all_records), 1):,}")
    log.info(f"  Output file:        {OUTPUT_FILE}")
    if OUTPUT_FILE.exists():
        log.info(f"  File size:          {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    log.info(f"  Est. chunks (512t): ~{total_words // 400:,}")

    if failed_pages:
        log.info("")
        log.info(f"  ❌ {len(failed_pages)} failed pages (will be retried on re-run):")
        for p in failed_pages:
            log.info(f"     - {p}")

    if _interrupted:
        log.info("")
        log.info("  ⚠️  Crawl interrupted — re-run to continue.")

    log.info("")
    log.info("  💡 Safe to re-run: existing pages will be skipped automatically.")


if __name__ == "__main__":
    scrape()