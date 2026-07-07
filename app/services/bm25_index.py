"""
BM25 index management.
- build_bm25   : build a BM25Okapi from a list of Documents
- save_bm25    : persist to disk (pickle)
- load_bm25    : load from disk
- bm25_search  : search and return (chunk_idx, score) pairs
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.services.arabic_utils import normalize_arabic

logger = logging.getLogger(__name__)


def build_bm25(chunks: list[Document]) -> BM25Okapi:
    corpus = [normalize_arabic(c.page_content).split() for c in chunks]
    index = BM25Okapi(corpus)
    logger.info("BM25 index built over %d chunks.", len(corpus))
    return index


def save_bm25(index: BM25Okapi, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f)
    logger.info("BM25 index saved to %s", path)


def load_bm25(path: Path) -> BM25Okapi:
    if not path.exists():
        raise FileNotFoundError(f"BM25 index not found at {path}. Run scripts/init_bm25.py first.")
    with open(path, "rb") as f:
        index = pickle.load(f)
    logger.info("BM25 index loaded from %s", path)
    return index


def bm25_search(
    index: BM25Okapi,
    query: str,
    k: int,
) -> list[tuple[int, float]]:
    """Return top-k (chunk_idx, score) pairs (score > 0 only)."""
    tokens = normalize_arabic(query).split()
    scores = index.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [(i, float(scores[i])) for i in ranked[:k] if scores[i] > 0]
