"""
One-time BM25 index build script.
Loads child chunks from disk (written by init_index.py),
builds BM25Okapi index, and saves bm25_index.pkl.

Run AFTER init_index.py:
    python scripts/init_bm25.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from app.core.config import settings
from app.core.logging import setup_logging

setup_logging("INFO")

import logging
logger = logging.getLogger("init_bm25")


def main() -> None:
    logger.info("=== init_bm25.py starting ===")

    chunks_path = settings.models_dir / "chunks.pkl"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"chunks.pkl not found at {chunks_path}. Run init_index.py first."
        )

    with open(chunks_path, "rb") as f:
        child_chunks = pickle.load(f)
    logger.info("Loaded %d child chunks.", len(child_chunks))

    from app.services.bm25_index import build_bm25, save_bm25
    bm25 = build_bm25(child_chunks)
    save_bm25(bm25, settings.models_dir / "bm25_index.pkl")

    logger.info("=== init_bm25.py complete ===")


if __name__ == "__main__":
    main()
