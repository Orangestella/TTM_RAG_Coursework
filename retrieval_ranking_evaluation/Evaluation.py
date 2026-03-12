import json
from pathlib import Path
from collections import defaultdict

# =========================================================
# 1. Config
# =========================================================
DATA_DIR = Path("../data")
RESULTS_DIR = Path("./results")
OUTPUT_PATH = RESULTS_DIR / "evaluation_summary_all.json"

BENCHMARK_FILES = {
    "benchmark_dataset": DATA_DIR / "benchmark_dataset.json",
    "benchmark_dataset_2": DATA_DIR / "benchmark_dataset_2.json",
}

FILE_NAMES = [
    "chunks_fixed300_overlap0",
    "chunks_fixed300_overlap20",
    "chunks_sentence220_overlap1",
    "chunks_sentence220_overlap2",
    "chunks_structure_aware_coarse",
    "chunks_structure_aware_fine",
]

RETRIEVAL_TYPES = [
    "biencoder_top20",
    "biencoder_top5",
    "reranked_top5",
]


# =========================================================
# 2. IO helpers
# =========================================================
def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# =========================================================
# 3. Normalization helpers
# =========================================================
def normalize_doc_id(chunk_id: str) -> str:
    """
    Example:
    wiki_0018_chunk_0 -> wiki_0018
    """
    if "_chunk_" in chunk_id:
        return chunk_id.split("_chunk_")[0]
    return chunk_id


def build_retrieval_path(results_dir: Path, retrieval_type: str, file_name: str, benchmark_name: str) -> Path:
    """
    Examples:
    retrieval_biencoder_top20_chunks_fixed300_overlap0_benchmark_dataset.jsonl
    retrieval_biencoder_top5_chunks_fixed300_overlap0_benchmark_dataset_2.jsonl
    retrieval_reranked_top5_chunks_structure_aware_fine_benchmark_dataset.jsonl
    """
    return results_dir / f"retrieval_{retrieval_type}_{file_name}_{benchmark_name}.jsonl"


# =========================================================
# 4. Load benchmark / retrieval
# =========================================================
def load_benchmark(path: Path):
    """
    benchmark json format:
    [
      {
        "id": "...",
        "question": "...",
        "answer": "...",
        "doc_id": "...",
        "cuisine": "...",
        "difficulty": "...",
        "type": "..."
      },
      ...
    ]
    """
    benchmark = {}
    data = read_json(path)
    for rec in data:
        benchmark[rec["id"]] = rec
    return benchmark


def load_retrieval_as_dict(path: Path):
    """
    retrieval jsonl format:
    {
      "query_id": "...",
      "query": "...",
      "retrieved": [
         {"rank": 1, "chunk_id": "...", "score": ..., "text": "..."},
         ...
      ]
    }
    """
    data = {}
    for rec in read_jsonl(path):
        data[rec["query_id"]] = rec
    return data


# =========================================================
# 5. Evaluation core
# =========================================================
def empty_group_stats():
    return {"total": 0, "hit1": 0, "hit5": 0, "hit20": 0, "mrr": 0.0}


def evaluate_retrieval(benchmark, retrieval_dict):
    total = 0
    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_20 = 0
    mrr = 0.0

    by_cuisine = defaultdict(empty_group_stats)
    by_difficulty = defaultdict(empty_group_stats)
    by_type = defaultdict(empty_group_stats)

    missing_queries = []

    for qid, gold in benchmark.items():
        if qid not in retrieval_dict:
            missing_queries.append(qid)
            continue

        gold_doc_id = gold["doc_id"]
        cuisine = gold.get("cuisine", "UNKNOWN")
        difficulty = gold.get("difficulty", "UNKNOWN")
        qtype = gold.get("type", "UNKNOWN")

        retrieved = retrieval_dict[qid].get("retrieved", [])
        retrieved_doc_ids = [normalize_doc_id(x["chunk_id"]) for x in retrieved]

        total += 1

        h1 = len(retrieved_doc_ids) >= 1 and retrieved_doc_ids[0] == gold_doc_id
        h5 = gold_doc_id in retrieved_doc_ids[:5]
        h20 = gold_doc_id in retrieved_doc_ids[:20]

        rr = 0.0
        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id == gold_doc_id:
                rr = 1.0 / rank
                break

        hit_at_1 += int(h1)
        hit_at_5 += int(h5)
        hit_at_20 += int(h20)
        mrr += rr

        for key, group_stats in [
            (cuisine, by_cuisine),
            (difficulty, by_difficulty),
            (qtype, by_type),
        ]:
            group_stats[key]["total"] += 1
            group_stats[key]["hit1"] += int(h1)
            group_stats[key]["hit5"] += int(h5)
            group_stats[key]["hit20"] += int(h20)
            group_stats[key]["mrr"] += rr

    result = {
        "n_queries_evaluated": total,
        "n_queries_in_benchmark": len(benchmark),
        "n_missing_queries": len(missing_queries),
        "missing_queries": missing_queries,
        "Hit@1": hit_at_1 / total if total else 0.0,
        "Hit@5": hit_at_5 / total if total else 0.0,
        "Recall@20": hit_at_20 / total if total else 0.0,
        "MRR": mrr / total if total else 0.0,
        "by_cuisine": convert_group_stats(by_cuisine),
        "by_difficulty": convert_group_stats(by_difficulty),
        "by_type": convert_group_stats(by_type),
    }
    return result


def convert_group_stats(group_stats):
    out = {}
    for key, stats in group_stats.items():
        total = stats["total"]
        out[key] = {
            "n": total,
            "Hit@1": stats["hit1"] / total if total else 0.0,
            "Hit@5": stats["hit5"] / total if total else 0.0,
            "Recall@20": stats["hit20"] / total if total else 0.0,
            "MRR": stats["mrr"] / total if total else 0.0,
        }
    return dict(sorted(out.items(), key=lambda x: x[0]))


# =========================================================
# 6. Pretty print
# =========================================================
def print_overall_result(tag, result):
    print(f"\n{'=' * 100}")
    print(tag)
    print(f"{'=' * 100}")
    print(f"Queries evaluated : {result['n_queries_evaluated']} / {result['n_queries_in_benchmark']}")
    print(f"Missing queries   : {result['n_missing_queries']}")
    print(f"Hit@1             : {result['Hit@1']:.4f}")
    print(f"Hit@5             : {result['Hit@5']:.4f}")
    print(f"Recall@20         : {result['Recall@20']:.4f}")
    print(f"MRR               : {result['MRR']:.4f}")


def print_group_result(title, group_result):
    print(f"\n{title}")
    print("-" * len(title))
    for key, stats in group_result.items():
        print(
            f"{str(key):20s} | "
            f"n={stats['n']:3d} | "
            f"Hit@1={stats['Hit@1']:.4f} | "
            f"Hit@5={stats['Hit@5']:.4f} | "
            f"Recall@20={stats['Recall@20']:.4f} | "
            f"MRR={stats['MRR']:.4f}"
        )


# =========================================================
# 7. Run all 36 evaluations
# =========================================================
def main():
    all_results = {}

    # preload benchmarks
    benchmark_cache = {}
    for benchmark_name, benchmark_path in BENCHMARK_FILES.items():
        if not benchmark_path.exists():
            print(f"[WARN] Benchmark file not found: {benchmark_path}")
            continue
        benchmark_cache[benchmark_name] = load_benchmark(benchmark_path)

    for benchmark_name, benchmark in benchmark_cache.items():
        for file_name in FILE_NAMES:
            for retrieval_type in RETRIEVAL_TYPES:
                retrieval_path = build_retrieval_path(
                    RESULTS_DIR, retrieval_type, file_name, benchmark_name
                )

                tag = f"{benchmark_name} | {file_name} | {retrieval_type}"

                if not retrieval_path.exists():
                    print(f"[WARN] Missing retrieval file: {retrieval_path}")
                    continue

                retrieval_dict = load_retrieval_as_dict(retrieval_path)
                result = evaluate_retrieval(benchmark, retrieval_dict)

                all_results[tag] = {
                    "benchmark": benchmark_name,
                    "chunking": file_name,
                    "retrieval_type": retrieval_type,
                    "retrieval_path": str(retrieval_path),
                    "metrics": result,
                }

                print_overall_result(tag, result)

    # save all results
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved evaluation summary to: {OUTPUT_PATH}")

    # print leaderboard
    leaderboard = []
    for tag, info in all_results.items():
        metrics = info["metrics"]
        leaderboard.append({
            "tag": tag,
            "benchmark": info["benchmark"],
            "chunking": info["chunking"],
            "retrieval_type": info["retrieval_type"],
            "Hit@1": metrics["Hit@1"],
            "Hit@5": metrics["Hit@5"],
            "Recall@20": metrics["Recall@20"],
            "MRR": metrics["MRR"],
        })

    leaderboard = sorted(leaderboard, key=lambda x: x["MRR"], reverse=True)

    print(f"\n{'#' * 100}")
    print("Leaderboard by MRR")
    print(f"{'#' * 100}")
    for row in leaderboard:
        print(
            f"{row['benchmark']:20s} | "
            f"{row['chunking']:32s} | "
            f"{row['retrieval_type']:15s} | "
            f"Hit@1={row['Hit@1']:.4f} | "
            f"Hit@5={row['Hit@5']:.4f} | "
            f"Recall@20={row['Recall@20']:.4f} | "
            f"MRR={row['MRR']:.4f}"
        )


if __name__ == "__main__":
    main()