"""
Qdrant vector store integration.
- get_qdrant_client  : returns a connected QdrantClient
- get_vectorstore    : returns a LangChain QdrantVectorStore wrapper
- index_chunks       : used only by the init script (one-time indexing)
"""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_qdrant_client(
    host: str = "localhost",
    port: int = 6333,
    prefer_grpc: bool = True,
    timeout: int = 600,
    url: str | None = None,
    api_key: str | None = None,
) -> QdrantClient:
    """Return a connected QdrantClient."""
    if url:
        logger.info("Connecting to Qdrant at %s", url)
        return QdrantClient(url=url, api_key=api_key, timeout=timeout)
    logger.info("Connecting to Qdrant at %s:%s", host, port)
    return QdrantClient(
        host=host,
        port=port,
        prefer_grpc=prefer_grpc,
        timeout=timeout,
    )


def get_vectorstore(client: QdrantClient, embeddings, collection_name: str):
    """Wrap an existing Qdrant collection as a LangChain vector store."""
    from langchain_qdrant import QdrantVectorStore

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


def index_chunks(
    chunks: list[Document],
    client: QdrantClient,
    collection_name: str,
    embedding_model,
    embedding_dim: int = 768,
    batch_size: int = 100,
    encode_batch_size: int = 256,
) -> None:
    """
    Embed *chunks* and upsert them into Qdrant.
    Creates the collection if it doesn't exist.
    Skips indexing if point count already matches len(chunks).
    """
    collections = [c.name for c in client.get_collections().collections]

    if collection_name not in collections:
        logger.info("Creating Qdrant collection '%s'", collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )

    info = client.get_collection(collection_name)
    existing = info.points_count or 0
    expected = len(chunks)

    if existing == expected:
        logger.info(
            "Collection '%s' already has %d points — skipping indexing.",
            collection_name,
            existing,
        )
        return

    if existing > 0:
        logger.warning(
            "Collection has %d points (expected %d). Recreating…",
            existing,
            expected,
        )
        client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )

    logger.info("Encoding %d chunks with the configured Hugging Face model…", expected)

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    embs = embedding_model.encode(
        texts,
        batch_size=encode_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    logger.info("Uploading to Qdrant…")
    for start in tqdm(range(0, len(texts), batch_size), desc="Uploading"):
        end = min(start + batch_size, len(texts))
        points = [
            PointStruct(
                id=chunks[i].metadata["idx"],
                vector=embs[i].tolist(),
                payload={
                    "page_content": texts[i],
                    "metadata": metadatas[i],
                },
            )
            for i in range(start, end)
        ]
        client.upload_points(
            collection_name=collection_name,
            points=points,
            batch_size=64,
            parallel=1,
            max_retries=100,
            wait=True,
        )

    logger.info("Indexed %d chunks into '%s'.", len(texts), collection_name)
