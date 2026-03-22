# RAG Culinary QA System — East Asian Cuisine

## Quick Start

#### Using Conda
```bash
conda env create -f environment.yml
conda activate COMP64702_COURSEWORK
```

#### Using pip
```bash
pip install -r requirements.txt
```

### Inference Pipeline
1. Place the test query file (e.g. `input_payload.json`) in the root directory
2. Open `inference_pipeline.ipynb`
3. Change `INPUT_JSON` to point to the test file
4. Run All → output saved to `results/output_results.json`

### Evaluation Pipeline
1. Open `evaluation_pipeline.ipynb`
2. Run All → prints retrieval and generation metrics

## Directory Structure

```
submission/
│
├── inference_pipeline.ipynb        # Main inference pipeline (Run All)
├── evaluation_pipeline.ipynb       # Evaluation pipeline (Run All)
│
├── data/                           # Pre-built retrieval index
│   └── chunks_fixed300_overlap0/
│       ├── embedded_nomic.faiss    # FAISS index (nomic-embed-text-v1.5)
│       └── embedded_nomic.jsonl    # Chunk texts + embeddings
│
├── examples/                       # Sample / reference files
│
└── results/                        # Pipeline outputs
    ├── output_results.json         # Generated answers from inference
    └── evaluation_summary.json     # Evaluation metrics
```

## Pipeline Configuration

| Component | Choice |
|-----------|--------|
| Chunking | Fixed 300 tokens, 0% overlap |
| Embedding | nomic-ai/nomic-embed-text-v1.5 (768d) |
| Retrieval | Bi-encoder + cross-encoder reranker (top-5) |
| Generation | Qwen/Qwen2.5-0.5B-Instruct |
| Prompt | Basic template, greedy decoding |

## Input / Output Format

**Input** (`input_payload.json`):
```json
{
  "queries": [
    {"query_id": "0", "query": "What is sushi made of?"}
  ]
}
```

**Output** (`results/output_results.json`):
```json
{
  "results": [
    {
      "query_id": "0",
      "query": "What is sushi made of?",
      "response": "...",
      "retrieved_context": [
        {"doc_id": "wiki_0001_chunk_0", "text": "..."}
      ]
    }
  ]
}
```

