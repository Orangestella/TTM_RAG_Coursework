import json
import re
from pathlib import Path
from typing import List, Dict, Any, Iterable

# =========================
# Config
# =========================

MAX_WORDS_PER_CHUNK = 220      # 可以调大/调小（ 180~300）
MIN_WORDS_PER_CHUNK = 40       # 太短的 chunk 可丢弃（可设 0 表示不丢）
SENT_OVERLAP = 2               # chunk 之间重叠的“句子数”（0/1/2 常用）

INPUT_JSONLS = [
    Path("../data/corpus_wikibooks_east_asia.jsonl"),
    Path("../data/corpus_blog.jsonl"),
    Path("../data/corpus_wikipedia.jsonl"),
]
OUTPUT_JSONL = Path(f"data/chunks_sentence{MAX_WORDS_PER_CHUNK}_overlap{SENT_OVERLAP}.jsonl")



# 额外给每个 chunk 加上 title 前缀，设为 True
PREFIX_TITLE = False

EMPTY_EMBEDDING = []

# =========================
# Sentence splitter
# =========================

def _try_nltk_sent_tokenize(text: str) -> List[str] | None:
    """Try NLTK punkt; return None if unavailable."""
    try:
        import nltk
        try:
            from nltk.tokenize import sent_tokenize
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except LookupError:
            # punkt not downloaded
            return None
    except Exception:
        return None


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(\[])")


def regex_sentence_split(text: str) -> List[str]:
    """
    Simple English sentence splitter (fallback).
    Works ok for Wikibooks recipe prose.
    """
    text = text.strip()
    if not text:
        return []
    # normalize whitespace a bit
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # split on punctuation boundaries
    parts = _SENT_SPLIT_RE.split(text)
    # further split long lines with newlines (recipes often have short lines)
    sentences: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # break on double newlines as pseudo-sentences (sections/steps)
        subparts = [x.strip() for x in re.split(r"\n{2,}", p) if x.strip()]
        sentences.extend(subparts)
    return sentences


def split_into_sentences(text: str) -> List[str]:
    sents = _try_nltk_sent_tokenize(text)
    if sents is not None and len(sents) > 0:
        return sents
    return regex_sentence_split(text)


# =========================
# Chunker
# =========================

def count_words(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))


def sentence_based_chunking(
    sentences: List[str],
    max_words: int,
    min_words: int,
    sent_overlap: int
) -> List[str]:
    chunks: List[str] = []
    cur: List[str] = []
    cur_words = 0

    for sent in sentences:
        w = count_words(sent)

        # if single sentence is too large, still keep it as a chunk
        if not cur and w >= max_words:
            if w >= min_words:
                chunks.append(sent.strip())
            continue

        if cur_words + w <= max_words:
            cur.append(sent)
            cur_words += w
        else:
            # flush current
            chunk_text = "\n".join(cur).strip()
            if count_words(chunk_text) >= min_words:
                chunks.append(chunk_text)

            # start new chunk with overlap
            if sent_overlap > 0:
                overlap_sents = cur[-sent_overlap:] if len(cur) >= sent_overlap else cur[:]
            else:
                overlap_sents = []

            cur = overlap_sents.copy()
            cur_words = sum(count_words(x) for x in cur)

            # add current sentence
            cur.append(sent)
            cur_words += w

    # last flush
    if cur:
        chunk_text = "\n".join(cur).strip()
        if count_words(chunk_text) >= min_words:
            chunks.append(chunk_text)

    return chunks


# =========================
# IO helpers
# =========================

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =========================
# Main
# =========================

def main():
    out_records: List[Dict[str, Any]] = []
    n_docs = 0
    n_chunks = 0
    for input_path in INPUT_JSONLS:
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path.resolve()}")

        for doc in read_jsonl(input_path):
            # doc_id, source, title, text, url, metadata
            doc_id = doc.get("doc_id") or ""
            source = doc.get("source") or ("Blog" if "blog" in input_path.name else "Wikipedia" if "wikipedia" in input_path.name else "Wikibooks")
            raw_title = doc.get("title") or ""

            if source.lower() == "wikibooks":
                title = re.sub(r"^Cookbook:\s*", "", raw_title)
            else:
                title = raw_title
            text = doc.get("text") or ""

            if not doc_id or not text.strip():
                continue

            n_docs += 1


            if PREFIX_TITLE and title:
                text_for_split = f"{title}\n\n{text}"
            else:
                text_for_split = text

            sentences = split_into_sentences(text_for_split)
            chunks = sentence_based_chunking(
                sentences,
                max_words=MAX_WORDS_PER_CHUNK,
                min_words=MIN_WORDS_PER_CHUNK,
                sent_overlap=SENT_OVERLAP
            )

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"

                out_records.append({
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "embedding": EMPTY_EMBEDDING,
                    "metadata": {
                        "doc_id": doc_id,
                        "source": source,
                        "title": title,
                    }
                })
                n_chunks += 1

    write_jsonl(OUTPUT_JSONL, out_records)
    print(f" Done. Docs processed: {n_docs}, chunks written: {n_chunks}")
    print(f" Output: {OUTPUT_JSONL.resolve()}")


if __name__ == "__main__":
    main()