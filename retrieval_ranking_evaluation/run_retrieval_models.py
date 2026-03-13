import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
EMBEDDING_DIR = ROOT_DIR / "embedding" / "output"
RESULTS_DIR = ROOT_DIR / "retrieval_ranking_evaluation" / "results"

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

MODEL_CONFIGS = {
    "minilm": {
        "embedding_file": "embedded_minilm",
        "query_encoder": "sentence_transformers",
        "model_name": "all-MiniLM-L6-v2",
        "query_prefix": "",
    },
    "nomic": {
        "embedding_file": "embedded_nomic",
        "query_encoder": "sentence_transformers",
        "model_name": "nomic-ai/nomic-embed-text-v1",
        "query_prefix": "search_query: ",
        "trust_remote_code": True,
    },
    "openai_small": {
        "embedding_file": "embedded_openai_small",
        "query_encoder": "openai",
        "model_name": "text-embedding-3-small",
        "openai_batch_size": 100,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run retrieval and reranking for MiniLM, nomic, and OpenAI small."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_CONFIGS.keys()),
        default=["nomic", "openai_small"],
        help="Embedding model aliases to run.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=sorted(BENCHMARK_FILES.keys()),
        default=sorted(BENCHMARK_FILES.keys()),
        help="Benchmark datasets to run.",
    )
    parser.add_argument(
        "--file-names",
        nargs="+",
        default=FILE_NAMES,
        help="Chunking output folders to run.",
    )
    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        help="Skip cross-encoder reranking and only write bi-encoder outputs.",
    )
    parser.add_argument(
        "--cross-encoder-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model used for reranking top-20 into top-5.",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="Optional OpenAI API key. If omitted, OPENAI_API_KEY is used.",
    )
    return parser.parse_args()


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_benchmark(path: Path) -> List[dict]:
    data = read_json(path)
    return [
        {
            "query_id": rec["id"],
            "query": rec["question"],
            "answer": rec.get("answer", ""),
            "doc_id": rec["doc_id"],
        }
        for rec in data
    ]


def load_chunks(embedded_path: Path) -> List[dict]:
    return list(read_jsonl(embedded_path))


def get_query_encoder(model_alias: str, openai_api_key: Optional[str] = None):
    config = MODEL_CONFIGS[model_alias]
    if config["query_encoder"] == "sentence_transformers":
        return SentenceTransformer(
            config["model_name"],
            trust_remote_code=config.get("trust_remote_code", False),
        )
    if config["query_encoder"] == "openai":
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY") or"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"
        if not api_key:
            raise RuntimeError(
                "openai_small requires an OpenAI API key. Set OPENAI_API_KEY "
                "or pass --openai-api-key when running the script."
            )
        return OpenAI(api_key=api_key)
    raise ValueError(f"Unsupported query encoder for model alias: {model_alias}")


def encode_queries(encoder, model_alias: str, queries: List[str]) -> np.ndarray:
    config = MODEL_CONFIGS[model_alias]
    if config["query_encoder"] == "sentence_transformers":
        prefixed = [config.get("query_prefix", "") + q for q in queries]
        embs = encoder.encode(prefixed, normalize_embeddings=True, show_progress_bar=True)
        return np.asarray(embs, dtype=np.float32)

    if config["query_encoder"] == "openai":
        batch_size = config.get("openai_batch_size", 100)
        all_embs = []
        for start in range(0, len(queries), batch_size):
            batch = queries[start : start + batch_size]
            resp = encoder.embeddings.create(model=config["model_name"], input=batch)
            all_embs.extend(item.embedding for item in resp.data)
        embs = np.asarray(all_embs, dtype=np.float32)
        faiss.normalize_L2(embs)
        return embs

    raise ValueError(f"Unsupported query encoder for model alias: {model_alias}")


def retrieve_top_k(
    index: faiss.Index,
    chunks: List[dict],
    query_ids: List[str],
    queries: List[str],
    query_embs: np.ndarray,
    top_k: int,
) -> List[dict]:
    scores, indices = index.search(query_embs, top_k)
    results = []
    for i, query_id in enumerate(query_ids):
        retrieved = []
        for rank, (chunk_idx, score) in enumerate(zip(indices[i], scores[i]), start=1):
            if chunk_idx < 0:
                continue
            chunk = chunks[int(chunk_idx)]
            retrieved.append(
                {
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "score": float(score),
                    "text": chunk["text"],
                }
            )
        results.append({"query_id": query_id, "query": queries[i], "retrieved": retrieved})
    return results


def rerank_results(
    cross_encoder: CrossEncoder,
    top20_records: List[dict],
    top_k: int = 5,
) -> List[dict]:
    reranked = []
    for rec in top20_records:
        pairs = [(rec["query"], item["text"]) for item in rec["retrieved"]]
        if not pairs:
            reranked.append({"query_id": rec["query_id"], "query": rec["query"], "retrieved": []})
            continue
        scores = cross_encoder.predict(pairs, show_progress_bar=False)
        ordered = sorted(
            zip(rec["retrieved"], scores),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )[:top_k]
        reranked.append(
            {
                "query_id": rec["query_id"],
                "query": rec["query"],
                "retrieved": [
                    {
                        "rank": rank,
                        "chunk_id": item["chunk_id"],
                        "score": float(score),
                        "text": item["text"],
                    }
                    for rank, (item, score) in enumerate(ordered, start=1)
                ],
            }
        )
    return reranked


def output_path(retrieval_type: str, file_name: str, benchmark_name: str, model_alias: str) -> Path:
    return RESULTS_DIR / f"retrieval_{retrieval_type}_{file_name}_{benchmark_name}_{model_alias}.jsonl"


def run_one(
    model_alias: str,
    benchmark_name: str,
    file_name: str,
    cross_encoder: Optional[CrossEncoder],
    openai_api_key: Optional[str],
):
    config = MODEL_CONFIGS[model_alias]
    embedded_base = config["embedding_file"]
    embedded_path = EMBEDDING_DIR / file_name / f"{embedded_base}.jsonl"
    faiss_path = EMBEDDING_DIR / file_name / f"{embedded_base}.faiss"
    benchmark_path = BENCHMARK_FILES[benchmark_name]

    if not embedded_path.exists():
        raise FileNotFoundError(f"Embedded JSONL not found: {embedded_path}")
    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    print(f"\nRunning {model_alias} | {benchmark_name} | {file_name}")
    chunks = load_chunks(embedded_path)
    index = faiss.read_index(str(faiss_path))
    benchmark = load_benchmark(benchmark_path)

    query_ids = [item["query_id"] for item in benchmark]
    queries = [item["query"] for item in benchmark]

    encoder = get_query_encoder(model_alias, openai_api_key=openai_api_key)
    query_embs = encode_queries(encoder, model_alias, queries)

    top20_records = retrieve_top_k(index, chunks, query_ids, queries, query_embs, top_k=20)
    top5_records = [
        {"query_id": rec["query_id"], "query": rec["query"], "retrieved": rec["retrieved"][:5]}
        for rec in top20_records
    ]

    top20_path = output_path("biencoder_top20", file_name, benchmark_name, model_alias)
    top5_path = output_path("biencoder_top5", file_name, benchmark_name, model_alias)
    write_jsonl(top20_path, top20_records)
    write_jsonl(top5_path, top5_records)
    print(f"Saved top-20 -> {top20_path}")
    print(f"Saved top-5  -> {top5_path}")

    if cross_encoder is not None:
        reranked_records = rerank_results(cross_encoder, top20_records, top_k=5)
        reranked_path = output_path("reranked_top5", file_name, benchmark_name, model_alias)
        write_jsonl(reranked_path, reranked_records)
        print(f"Saved reranked top-5 -> {reranked_path}")


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cross_encoder = None
    if not args.skip_rerank:
        cross_encoder = CrossEncoder(args.cross_encoder_model)

    for model_alias in args.models:
        for benchmark_name in args.benchmarks:
            for file_name in args.file_names:
                run_one(
                    model_alias,
                    benchmark_name,
                    file_name,
                    cross_encoder,
                    args.openai_api_key,
                )


if __name__ == "__main__":
    main()
