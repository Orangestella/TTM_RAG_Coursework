import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
INPUT_FILES = [
    DATA_DIR / "benchmark_dataset.json",
    # Add more benchmark files here if needed, e.g.:
    # DATA_DIR / "benchmark_dataset_2.json",
]

OUTPUT_JSONL = DATA_DIR / "benchmark_unified.jsonl"


def load_benchmark_json(path: Path):
    """
    Load benchmark JSON in the nested source-grouped format:
      {
        "sources": [
          {
            "source": "Wikipedia",
            "questions": [
              {"question": "...", "answer": "..."},
              ...
            ]
          },
          ...
        ]
      }
    Returns a flat list of dicts, each with: question, answer, source.
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict) or not isinstance(data.get("sources"), list):
        raise TypeError(
            f"Unsupported benchmark format in {path}: "
            "expected a dict with 'sources' -> list of source blocks."
        )

    records = []
    for source_block in data["sources"]:
        source_name = source_block.get("source", "")
        for q in source_block.get("questions", []):
            if isinstance(q, dict) and q.get("question", "").strip():
                records.append({
                    "question": q["question"].strip(),
                    "answer":   q.get("answer", "").strip(),
                    "source":   source_name,
                })
    return records


def normalize_record(record: dict, idx: int, source_file: str) -> dict:
    """
    Map a flat question record to the unified output schema.
    Output fields mirror the original benchmark_unified.jsonl schema:
      query_id, question, gold_answer, gold_doc_id,
      cuisine, difficulty, type, source_file, source
    Fields not present in the new benchmark format are left empty.
    """
    return {
        "query_id":    f"q_{idx:03d}",
        "question":    record["question"],
        "gold_answer": record["answer"],
        "source_file": source_file,
        "source":      record["source"],
    }


def main():
    all_records = []
    seen = set()
    global_idx = 1

    for input_path in INPUT_FILES:
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path.resolve()}")

        raw_records = load_benchmark_json(input_path)
        source_file = input_path.stem      # e.g. "benchmark_dataset"

        for rec in raw_records:
            # deduplicate by question text
            key = rec["question"]
            if key in seen:
                continue
            seen.add(key)

            norm = normalize_record(rec, global_idx, source_file)
            all_records.append(norm)
            global_idx += 1

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_records)} records to {OUTPUT_JSONL.resolve()}")


if __name__ == "__main__":
    main()