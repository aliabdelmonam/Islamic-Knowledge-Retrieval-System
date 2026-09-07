"""
Category retriever: semantic search over high-level hadith category
descriptions to identify the most relevant categories for a user query.

Used as Stage 1 in the multi-stage retrieval pipeline.
"""
from __future__ import annotations

import logging
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)


def build_category_index(
    client: QdrantClient,
    collection_name: str,
    category_names: list[str],
    category_descriptions: list[str],
    embedding_model,
    embedding_dim: int = 768,
) -> None:
    """
    Build a Qdrant collection of category description embeddings.

    Parameters
    ----------
    client : QdrantClient
        Connected Qdrant client.
    collection_name : str
        Name for the category collection (e.g. ``hadith_categories``).
    category_names : list[str]
        High-level category names (e.g. ``['صلاة', 'صيام', ...]``).
    category_descriptions : list[str]
        Corresponding descriptions for each category.
    embedding_model
        Configured Hugging Face model for encoding.
    embedding_dim : int
        Dimension of the embedding vectors.
    """
    collections = [c.name for c in client.get_collections().collections]

    if collection_name in collections:
        info = client.get_collection(collection_name)
        existing = info.points_count or 0
        if existing == len(category_names):
            logger.info(
                "Category collection '%s' already has %d points — skipping.",
                collection_name, existing,
            )
            return
        logger.warning(
            "Category collection has %d points (expected %d). Recreating…",
            existing, len(category_names),
        )
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )

    logger.info("Encoding %d category descriptions…", len(category_descriptions))

    embs = embedding_model.encode(
        category_descriptions,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    points = [
        PointStruct(
            id=i,
            vector=embs[i].tolist(),
            payload={
                "category_name": category_names[i],
                "description": category_descriptions[i],
            },
        )
        for i in range(len(category_names))
    ]

    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=64,
        wait=True,
    )
    logger.info("Indexed %d categories into '%s'.", len(points), collection_name)


def retrieve_categories(
    query: str,
    client: QdrantClient,
    embedding_model,
    collection_name: str = "hadith_categories",
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    Find the top-k most relevant categories for *query* using cosine
    similarity against category description embeddings.

    Parameters
    ----------
    query : str
        User query in Arabic.
    client : QdrantClient
        Connected Qdrant client.
    embedding_model
        Configured Hugging Face model (must match the indexing model).
    collection_name : str
        Qdrant collection holding category vectors.
    top_k : int
        Number of categories to return.

    Returns
    -------
    list[tuple[str, float]]
        List of ``(category_name, similarity_score)`` sorted by score desc.
    """
    query_vec = embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    results = client.query_points(
        collection_name=collection_name,
        query=query_vec,
        limit=top_k,
        with_payload=True,
    )

    categories: list[tuple[str, float]] = []
    for point in results.points:
        cat_name = point.payload.get("category_name", "")
        score = point.score
        categories.append((cat_name, score))
        logger.debug("Category match: %s (%.4f)", cat_name, score)

    logger.info(
        "Query %r → top-%d categories: %s",
        query[:60], top_k,
        [c[0] for c in categories],
    )
    return categories
