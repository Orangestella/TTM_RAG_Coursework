"""
clean_corpus_blog.py
====================
Step 2 of 2: Data Cleaning 

Reads raw_blog.jsonl, sends all titles+section names to Gemini API
in batches, flags entries NOT relevant to East Asian cuisine, then
lets you confirm deletions interactively.

This handles edge cases missed by keyword rules, e.g.:
  - A "Japan" article section that's really about WWII history
  - A "Taiwan" section about politics rather than food

Run:
    pip install google-genai
    export GEMINI_API_KEY="your-api-key"
    python clean_corpus_blog.py

    # Or pass input file as argument:
    python clean_corpus_blog.py data/raw_blog.jsonl

Output:
    data/raw_blog.jsonl      ← overwritten with cleaned version
    data/raw_blog.jsonl.bak  ← backup of original
    data/corpus_blog.jsonl   ← final clean copy (renamed)
"""

import json, sys, os
from pathlib import Path
from google import genai
import sys
import io

# ───────────────────────────── CONFIG ─────────────────────────────
MODEL        = "gemini-2.0-flash"
BATCH_SIZE   = 80
DEFAULT_IN   = Path("raw_blog.jsonl")
FINAL_OUT    = Path("corpus_blog.jsonl")

# ───────────────────────── GEMINI PROMPTS ─────────────────────────
SYSTEM_PROMPT = """\
You are a culinary taxonomy expert. Your task is to judge whether each
entry (identified by its article title and section title) is relevant
to an **East Asian cuisine** corpus.

East Asian cuisine covers: Chinese (including all regional sub-cuisines),
Japanese, Korean, Taiwanese, Mongolian, Tibetan, Hong Kong, and Macanese
food cultures — including specific dishes, ingredients, cooking techniques,
utensils, dining customs, and food-related beverages.

An entry is RELEVANT if its section text is primarily about:
  - Food, recipes, cooking methods, or ingredients
  - Dining culture, food history, or food-related customs
  - Food-related items (utensils, vessels, fermentation, etc.)

An entry is NOT RELEVANT if its section text is primarily about:
  - Non-food history (wars, politics, dynasties)
  - Geography, demographics, language
  - Non-East-Asian cuisines
  - Completely off-topic subjects (even if the article is East Asian)
"""

USER_PROMPT_TEMPLATE = """\
Below is a numbered list of entries from a corpus.
Each entry shows: index. [doc_id] Article Title > Section Title

For each entry, judge: is it relevant to East Asian cuisine?

Reply ONLY with a JSON array of objects for entries that are NOT relevant.
Each object: {{"index": <number>, "doc_id": "<doc_id>", "reason": "<brief reason>"}}

If ALL entries are relevant, reply with an empty array: []

Entries:
{entries_list}
"""

# ─────────────────────────── HELPERS ──────────────────────────────

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


def call_gemini(client, batch: list[tuple[int, dict]]) -> list[dict]:
    """Send a batch to Gemini. Returns list of flagged entries."""
    entries_list = "\n".join(
        f"{i}. [{r['doc_id']}] {r['title']}"
        for i, r in batch
    )
    prompt = USER_PROMPT_TEMPLATE.format(entries_list=entries_list)

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0.1,
            "response_mime_type": "application/json",
        },
    )

    text = response.text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0].strip()

    try:
        flagged = json.loads(text)
        return flagged if isinstance(flagged, list) else []
    except json.JSONDecodeError:
        print(f"  Could not parse response, skipping batch. Raw: {text[:200]}")
        return []

# ─────────────────────────── MAIN ─────────────────────────────────

def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN

    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} records from {input_path}")

    # ── Init Gemini ──
    api_key = os.environ.get("GEMINI_API_KEY") or input("Enter Gemini API key: ").strip()
    if not api_key:
        print("No API key provided.")
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    # ── Send to Gemini in batches ──
    indexed   = list(enumerate(records))
    n_batches = (len(indexed) + BATCH_SIZE - 1) // BATCH_SIZE
    all_flagged: list[dict] = []

    print(f"\nSending {len(records)} entries to {MODEL} in {n_batches} batch(es)...")
    for b in range(n_batches):
        batch = indexed[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        print(f"  Batch {b+1}/{n_batches} (entries {batch[0][0]+1}–{batch[-1][0]+1})...", end=" ")
        try:
            flagged = call_gemini(client, batch)
            print(f"→ {len(flagged)} flagged")
            all_flagged.extend(flagged)
        except Exception as e:
            print(f"→ Error: {e}")

    if not all_flagged:
        print("\nGemini found nothing irrelevant. Saving final corpus unchanged.")
        save_jsonl(records, FINAL_OUT)
        print(f"   {FINAL_OUT}  ({len(records)} records)")
        return

    # ── Build index of flagged records ──
    # Match by doc_id (most reliable)
    doc_id_to_idx = {r["doc_id"]: i for i, r in enumerate(records)}
    flagged_map: dict[int, dict] = {}   # record_index → flag info

    for item in all_flagged:
        doc_id = item.get("doc_id", "")
        reason = item.get("reason", "")
        if doc_id in doc_id_to_idx:
            idx = doc_id_to_idx[doc_id]
            flagged_map[idx] = {"title": records[idx]["title"],
                                 "doc_id": doc_id, "reason": reason}

    # ── Display for review ──
    print(f"\n{'='*70}")
    print(f"Gemini flagged {len(flagged_map)} entries as NOT East Asian cuisine:")
    print(f"{'='*70}\n")
    sorted_flagged = sorted(flagged_map.items())
    for num, (rec_idx, info) in enumerate(sorted_flagged, 1):
        print(f"  {num:3d}. [{info['doc_id']}] {info['title']}")
        print(f"       Reason: {info['reason']}\n")

    # ── Interactive confirmation ──
    print("-" * 70)
    print("Options:")
    print("  [a] Delete ALL flagged entries")
    print("  [n] Keep all (no changes)")
    print("  [s] Select which to KEEP (enter their numbers, rest will be deleted)")
    print()
    choice = input("Your choice [a/n/s]: ").strip().lower()

    to_delete: set[int] = set()

    if choice == "n":
        print("\nNo changes made.")
        save_jsonl(records, FINAL_OUT)
        print(f"   Final corpus (unchanged): {FINAL_OUT}")
        return

    elif choice == "a":
        to_delete = set(flagged_map.keys())

    elif choice == "s":
        keep_input = input(
            "Enter numbers to KEEP (comma-separated, empty = delete all): "
        ).strip()
        if keep_input:
            try:
                keep_nums = {int(x.strip()) for x in keep_input.split(",")}
            except ValueError:
                print("Invalid input. No changes made.")
                return
            for num, (rec_idx, _) in enumerate(sorted_flagged, 1):
                if num not in keep_nums:
                    to_delete.add(rec_idx)
        else:
            to_delete = set(flagged_map.keys())
    else:
        print("Invalid choice. No changes made.")
        return

    if not to_delete:
        print("\nNothing to delete.")
        save_jsonl(records, FINAL_OUT)
        return

    # ── Perform deletion ──
    deleted_titles = [records[i]["title"] for i in sorted(to_delete)]
    new_records    = [r for i, r in enumerate(records) if i not in to_delete]

    backup_path = input_path.with_suffix(".jsonl.bak")
    save_jsonl(records, backup_path)
    print(f"\nBackup saved → {backup_path}")

    save_jsonl(new_records, input_path)    # overwrite raw file
    save_jsonl(new_records, FINAL_OUT)     # save final clean copy

    print(f"\nDeleted {len(to_delete)} entries:")
    for t in deleted_titles:
        print(f"      - {t}")

    print(f"\nClean corpus: {len(new_records)} records")
    print(f"   Overwritten : {input_path}")
    print(f"   Final output: {FINAL_OUT}")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main()
