"""
One-time script to build and save the hadith-level BM25 index.
Loads hadiths directly from the source CSV, builds BM25 index over `hadith_clean`,
and saves hadith_bm25_index.pkl.

Usage:
    python scripts/init_hadith_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from app.core.config import settings
from app.core.logging import setup_logging

setup_logging("INFO")

import logging
logger = logging.getLogger("init_hadith_index")


def main() -> None:
    logger.info("=== init_hadith_index.py starting ===")

    csv_path = settings.data_csv
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found at {csv_path}. Please place it under data/."
        )

    from app.services.hadith_search import build_hadith_bm25, save_hadith_index
    
    # We use settings.max_rows if defined (to allow quick test runs on small data)
    index, records = build_hadith_bm25(csv_path, max_rows=settings.max_rows)
    
    output_path = settings.models_dir / "hadith_bm25_index.pkl"
    save_hadith_index(index, records, output_path)

    logger.info("=== init_hadith_index.py complete ===")


if __name__ == "__main__":
    main()
