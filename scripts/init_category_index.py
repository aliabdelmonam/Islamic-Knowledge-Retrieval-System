"""
One-time category index build script.
Loads category descriptions CSV → embeds → upserts to Qdrant.

Usage:
    python scripts/init_category_index.py
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
logger = logging.getLogger("init_category_index")


def main() -> None:
    import pandas as pd

    logger.info("=== init_category_index.py starting ===")

    # 1. Load category descriptions
    csv_path = settings.category_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"Category CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded %d categories from %s", len(df), csv_path.name)

    # Expect columns: high_level_category, description
    category_names = df["high_level_category"].tolist()
    descriptions = df["description"].fillna("").tolist()

    # 2. Connect to Qdrant
    from app.services.vector_store import get_qdrant_client
    client = get_qdrant_client(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        prefer_grpc=settings.qdrant_prefer_grpc,
        timeout=settings.qdrant_timeout,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    # 3. Build category index
    from app.services.category_retriever import build_category_index
    build_category_index(
        client=client,
        collection_name=settings.category_collection_name,
        category_names=category_names,
        category_descriptions=descriptions,
        model_name=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
    )

    logger.info("=== init_category_index.py complete ===")


if __name__ == "__main__":
    main()
