"""
JSONL Corpus Cleaner — Gemini-powered East Asian Cuisine Filter
================================================================
Reads corpus_wikipedia.jsonl, batches all titles to Gemini API,
identifies non-East-Asian-cuisine entries, and lets you confirm
deletions interactively.

Usage:
    pip install google-genai
    export GEMINI_API_KEY="your-api-key"
    python clean_corpus_gemini.py

    # Or pass file path as argument:
    python clean_corpus_gemini.py path/to/corpus_wikipedia.jsonl
"""

import json
import sys
import os
import copy
from pathlib import Path

from google import genai

# ============================================================
# CONFIG
# ============================================================

MODEL = "gemini-3-flash-preview"  
BATCH_SIZE = 80                 

DEFAULT_INPUT = Path("corpus_wikipedia.jsonl")

# ============================================================
# GEMINI PROMPT
# ============================================================

SYSTEM_PROMPT = """\
You are a culinary taxonomy expert. Your task is to judge whether each \
Wikipedia article title is relevant to an **East Asian cuisine** corpus.

East Asian cuisine covers: Chinese, Japanese, Korean, Taiwanese, Mongolian, \
Tibetan, Hong Kong, and Macanese food cultures — including their regional \
sub-cuisines, specific dishes, ingredients, cooking techniques, utensils, \
dining customs, and beverages.

Relevant examples: "Sichuan cuisine", "Sushi", "Kimchi", "Soy sauce", \
"Chopsticks", "Japanese tea ceremony", "Wok", "Peking duck", "Dashi", "Sake".

NOT relevant: cuisines from other regions (Georgian, Indian, Thai, French...), \
non-food topics (geography, politics, languages, ethnic groups, history \
unrelated to food), and overly generic articles (e.g. "Agriculture", \
"International trade") unless they are specifically about East Asian food.
"""

USER_PROMPT_TEMPLATE = """\
Below is a numbered list of Wikipedia article titles from a corpus.
For each title, judge: is it relevant to East Asian cuisine?

Reply ONLY with a JSON array of objects, one per title that is NOT relevant.
Each object: {{"index": <number>, "title": "<title>", "reason": "<brief reason>"}}

If ALL titles are relevant, reply with an empty array: []

Titles:
{titles_list}
"""


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def save_jsonl(records: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def format_titles_list(titles: list[tuple[int, str]]) -> str:
    """Format [(index, title), ...] into numbered list string."""
    return "\n".join(f"{idx}. {title}" for idx, title in titles)


def call_gemini(client: genai.Client, titles_batch: list[tuple[int, str]]) -> list[dict]:
    """
    Send a batch of titles to Gemini, return list of flagged entries.
    Each entry: {"index": int, "title": str, "reason": str}
    """
    prompt = USER_PROMPT_TEMPLATE.format(
        titles_list=format_titles_list(titles_batch)
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0.1,  # low randomness for classification
            "response_mime_type": "application/json",
        },
    )

    # Parse response
    text = response.text.strip()
    # Handle markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        flagged = json.loads(text)
        if isinstance(flagged, list):
            return flagged
        else:
            print(f"  ⚠️  Unexpected response format, skipping batch")
            return []
    except json.JSONDecodeError:
        print(f"  ⚠️  Failed to parse Gemini response, skipping batch")
        print(f"      Raw: {text[:200]}...")
        return []


# ============================================================
# MAIN
# ============================================================

def main():
    # --- Resolve input file ---
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        input_path = DEFAULT_INPUT

    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        sys.exit(1)

    # --- Load records ---
    records = load_jsonl(input_path)
    print(f"📂 Loaded {len(records)} records from {input_path}")
    print()

    # --- Init Gemini ---
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Enter your Gemini API key: ").strip()
        if not api_key:
            print("❌ No API key provided.")
            sys.exit(1)

    client = genai.Client(api_key=api_key)

    # --- Build title list with original indices ---
    indexed_titles = [(i, r["title"]) for i, r in enumerate(records)]

    # --- Send to Gemini in batches ---
    all_flagged: list[dict] = []

    total_batches = (len(indexed_titles) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"🔍 Sending {len(indexed_titles)} titles to {MODEL} in {total_batches} batch(es)...")
    print()

    for batch_num in range(total_batches):
        start = batch_num * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(indexed_titles))
        batch = indexed_titles[start:end]

        print(f"  Batch {batch_num + 1}/{total_batches} (titles {start + 1}-{end})...", end=" ")

        try:
            flagged = call_gemini(client, batch)
            print(f"→ {len(flagged)} flagged")
            all_flagged.extend(flagged)
        except Exception as e:
            print(f"→ ❌ Error: {e}")
            continue

    if not all_flagged:
        print()
        print("✅ Gemini says all titles are East Asian cuisine relevant. Nothing to remove!")
        return

    # --- Display flagged entries for review ---
    print()
    print("=" * 70)
    print(f"🚩 Gemini flagged {len(all_flagged)} titles as NOT East Asian cuisine related:")
    print("=" * 70)
    print()

    # Build a lookup: index in the flagged list -> record index
    # Gemini returns the index we sent (which is the position in indexed_titles)
    # We need to map back to original record index
    flagged_record_indices = {}
    for item in all_flagged:
        gemini_idx = item.get("index")
        title = item.get("title", "")
        reason = item.get("reason", "")

        # Try to find the actual record by matching title (more robust)
        matched_idx = None
        for rec_idx, rec in enumerate(records):
            if rec["title"] == title:
                matched_idx = rec_idx
                break

        # Fallback: use the index Gemini returned directly
        if matched_idx is None and isinstance(gemini_idx, int) and 0 <= gemini_idx < len(records):
            matched_idx = gemini_idx

        if matched_idx is not None:
            flagged_record_indices[matched_idx] = {
                "title": records[matched_idx]["title"],
                "reason": reason,
                "doc_id": records[matched_idx]["doc_id"],
            }

    # Print flagged items
    for i, (rec_idx, info) in enumerate(sorted(flagged_record_indices.items()), 1):
        print(f"  {i:3d}. [{info['doc_id']}] {info['title']}")
        print(f"       Reason: {info['reason']}")
        print()

    # --- Interactive confirmation ---
    print("-" * 70)
    print()
    print("Options:")
    print("  [a] Delete ALL flagged entries")
    print("  [n] Delete NONE (keep all)")
    print("  [s] Select which to delete (enter numbers to KEEP)")
    print()

    choice = input("Your choice [a/n/s]: ").strip().lower()

    if choice == "n":
        print("\n✅ No changes made.")
        return

    to_delete_indices: set[int] = set()

    if choice == "a":
        to_delete_indices = set(flagged_record_indices.keys())

    elif choice == "s":
        print()
        print("Enter the numbers (from the list above) you want to KEEP,")
        print("separated by commas. All others will be deleted.")
        print("Example: 1,3,5")
        print()
        keep_input = input("Numbers to KEEP (empty = delete all): ").strip()

        sorted_flagged = sorted(flagged_record_indices.items())

        if keep_input:
            try:
                keep_nums = {int(x.strip()) for x in keep_input.split(",")}
            except ValueError:
                print("❌ Invalid input. No changes made.")
                return

            for i, (rec_idx, _) in enumerate(sorted_flagged, 1):
                if i not in keep_nums:
                    to_delete_indices.add(rec_idx)
        else:
            to_delete_indices = set(flagged_record_indices.keys())

    else:
        print("❌ Invalid choice. No changes made.")
        return

    if not to_delete_indices:
        print("\n✅ Nothing to delete.")
        return

    # --- Perform deletion ---
    deleted_titles = [records[i]["title"] for i in sorted(to_delete_indices)]
    new_records = [r for i, r in enumerate(records) if i not in to_delete_indices]

    # Backup original file
    backup_path = input_path.with_suffix(".jsonl.bak")
    save_jsonl(records, backup_path)
    print(f"\n💾 Backup saved to {backup_path}")

    # Save cleaned file
    save_jsonl(new_records, input_path)

    print(f"🗑️  Deleted {len(to_delete_indices)} entries:")
    for t in deleted_titles:
        print(f"     - {t}")
    print(f"\n✅ Cleaned corpus: {len(new_records)} records → {input_path}")
    print(f"   (was {len(records)}, removed {len(to_delete_indices)})")


if __name__ == "__main__":
    main()