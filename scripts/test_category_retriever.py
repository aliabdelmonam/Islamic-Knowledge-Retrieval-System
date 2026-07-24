"""
Standalone test script for the multi-stage category retriever.

Run from the project root:
    python scripts/test_category_retriever.py

Tests:
1. Category matching: shows which categories match each query.
2. Filtered retrieval: runs full retrieve_with_category_filter pipeline.
"""
from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

# ── Make sure project root is on the path ──────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from app.core.config import settings
from app.core.logging import setup_logging

setup_logging("INFO")

logger = logging.getLogger("test_category_retriever")

# ── Test queries ───────────────────────────────────────────────────────────────

TEST_QUERIES = [
    "ما هي العبادة",
    "بكم يجوز أن يتصدق الأنسان من ماله قبل موته؟",
    "ينفع اقعد مع بنت الجيران لوحدنا ؟"
]

def load_components():
    """Load all required ML components."""
    logger.info("Loading embeddings [%s] …", settings.embedding_model)
    from app.services.embeddings import build_embeddings
    embeddings = build_embeddings(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
        openai_model=settings.openai_embedding_model,
        batch_size=settings.embedding_batch_size,
    )

    logger.info("Connecting to Qdrant …")
    from app.services.vector_store import get_qdrant_client, get_vectorstore
    client = get_qdrant_client(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        prefer_grpc=settings.qdrant_prefer_grpc,
        timeout=settings.qdrant_timeout,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    vectorstore = get_vectorstore(client, embeddings, settings.collection_name)

    logger.info("Loading chunks and parent store …")
    with open(settings.models_dir / "chunks.pkl", "rb") as f:
        all_chunks = pickle.load(f)
    with open(settings.models_dir / "parent_store.pkl", "rb") as f:
        parent_store = pickle.load(f)
    logger.info("  %d chunks, %d parents loaded.", len(all_chunks), len(parent_store))

    logger.info("Loading BM25 index …")
    from app.services.bm25_index import load_bm25
    bm25_index = load_bm25(settings.models_dir / "bm25_index.pkl")

    logger.info("Loading SentenceTransformer for hadith similarity …")
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(settings.embedding_model, device=device)
    if device == "cuda":
        embedding_model.half()

    return dict(
        embeddings=embeddings,
        qdrant_client=client,
        vectorstore=vectorstore,
        bm25_index=bm25_index,
        all_chunks=all_chunks,
        parent_store=parent_store,
        embedding_model=embedding_model,
    )


def main():
    separator = "=" * 70

    logger.info("=== Category Retriever Test ===")

    components = load_components()

    # ── Test 1: Category Matching ──────────────────────────────────────────
    print(f"\n{separator}")
    print("TEST 1: Category Matching (Stage 1 only)")
    print(separator)

    from app.services.category_retriever import retrieve_categories

    for query in TEST_QUERIES:
        categories = retrieve_categories(
            query=query,
            client=components["qdrant_client"],
            model_name=settings.embedding_model,
            collection_name=settings.category_collection_name,
            top_k=settings.category_top_k,
        )
        print(f"\n  Query: {query}")
        for cat_name, score in categories:
            print(f"    → {cat_name} ({score:.4f})")

    # ── Test 2: Full Category-Filtered Retrieval ───────────────────────────
    print(f"\n{separator}")
    print("TEST 2: Full Category-Filtered Retrieval")
    print(separator)

    from app.services.retriever import retrieve_with_category_filter

    for query in TEST_QUERIES:
        print(f"\n  Query: {query}")
        results = retrieve_with_category_filter(
            query=query,
            vectorstore=components["vectorstore"],
            bm25_index=components["bm25_index"],
            all_chunks=components["all_chunks"],
            parent_store=components["parent_store"],
            embedding_model=components["embedding_model"],
            qdrant_client=components["qdrant_client"],
            embedding_model_name=settings.embedding_model,
            category_collection_name=settings.category_collection_name,
            category_top_k=settings.category_top_k,
            k=settings.retriever_k,
            fetch_k=settings.retriever_fetch_k,
        )
        print(f"  Results: {len(results)}")
        for j, r in enumerate(results[:3], 1):
            hadith_preview = r.hadith[:120] if r.hadith else "(no hadith)"
            print(f"    [{j}] {hadith_preview}…")
            print(f"        Source: {r.source}  |  Similarity: {r.similarity_score:.4f}")

    print(f"\n{separator}")
    logger.info("=== Test complete ===")


if __name__ == "__main__":
    main()
