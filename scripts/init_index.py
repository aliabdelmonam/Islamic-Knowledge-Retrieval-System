"""
One-time indexing script.
Loads CSV → chunks → embeds → upserts to Qdrant.
Also persists parent_store.pkl and chunks.pkl to disk for API startup.

Usage:
    python scripts/init_index.py
"""
from __future__ import annotations

import os
import pickle
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
logger = logging.getLogger("init_index")


def main() -> None:
    logger.info("=== init_index.py starting ===")

    # 1. Load documents
    from app.services.data_loader import load_hadith_documents
    docs = load_hadith_documents(
        csv_path=settings.data_csv,
        embed_column=settings.embed_column,
        metadata_columns=settings.metadata_columns,
        max_rows=settings.max_rows,
    )
    logger.info("Loaded %d parent documents.", len(docs))

    # 2. Build child chunks
    from app.services.chunker import build_chunks
    child_chunks = build_chunks(
        raw_docs=docs,
        model_name=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    logger.info("Built %d child chunks from %d parents.", len(child_chunks), len(docs))

    # 3. Save chunks to disk
    models_dir = settings.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = models_dir / "chunks.pkl"

    with open(chunks_path, "wb") as f:
        pickle.dump(child_chunks, f)
    logger.info("Saved chunks (%d docs) → %s", len(child_chunks), chunks_path)

    # 4. Connect to Qdrant and index
    from app.services.vector_store import get_qdrant_client, index_chunks
    client = get_qdrant_client(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        prefer_grpc=settings.qdrant_prefer_grpc,
        timeout=settings.qdrant_timeout,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    index_chunks(
        chunks=child_chunks,
        client=client,
        collection_name=settings.collection_name,
        model_name=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        encode_batch_size=settings.embedding_batch_size,
    )

    logger.info("=== init_index.py complete ===")


if __name__ == "__main__":
    main()
