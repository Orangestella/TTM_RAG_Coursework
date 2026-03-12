import json
from pathlib import Path

INPUT_FILES = [
    Path("../data/benchmark_dataset.jsonl"),
    Path("../data/benchmark_dataset_2.jsonl"),
]

OUTPUT_JSONL = Path("../data/benchmark_unified.jsonl")


def load_messy_json_objects(path: Path):
    """
    Supports either:
    1) proper JSONL: one JSON object per line
    2) a file containing multiple JSON objects separated by commas/newlines
    3) a JSON array
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Case 1: JSON array
    if text.startswith("["):
        data = json.loads(text)
        return data if isinstance(data, list) else []

    # Case 2: try standard JSONL first
    records = []
    ok = True
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            ok = False
            break
    if ok and records:
        return records

    # Case 3: wrap as array and parse
    wrapped = "[" + text.rstrip(",") + "]"
    data = json.loads(wrapped)
    return data if isinstance(data, list) else []


def normalize_record(record: dict, idx: int, source_name: str) -> dict:
    query_id = record.get("id") or f"{source_name}_{idx:04d}"
    question = (record.get("question") or "").strip()
    gold_answer = (record.get("answer") or "").strip()
    gold_doc_id = (record.get("doc_id") or "").strip()

    return {
        "query_id": str(query_id),
        "question": question,
        "gold_answer": gold_answer,
        "gold_doc_id": gold_doc_id,
        "cuisine": record.get("cuisine", ""),
        "difficulty": record.get("difficulty", ""),
        "type": record.get("type", ""),
        "source_file": source_name,
    }


def main():
    all_records = []
    seen = set()

    for input_path in INPUT_FILES:
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path.resolve()}")

        raw_records = load_messy_json_objects(input_path)

        for i, rec in enumerate(raw_records, start=1):
            norm = normalize_record(rec, i, input_path.stem)

            # skip empty questions
            if not norm["question"]:
                continue

            # deduplicate by (question, gold_doc_id)
            key = (norm["question"], norm["gold_doc_id"])
            if key in seen:
                continue
            seen.add(key)

            all_records.append(norm)

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(all_records)} records to {OUTPUT_JSONL.resolve()}")


if __name__ == "__main__":
    main()