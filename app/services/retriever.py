"""
Hybrid retriever: dense (Qdrant) + BM25 → hadith-level embedding similarity.

Flow
----
1. Dense search (Qdrant) on sharh index → fetch_k chunks
2. BM25 search on sharh index → fetch_k chunks
3. Merge candidates (deduplicate by chunk idx)
4. Extract ALL hadiths from chunk metadata (each chunk may have N hadiths)
5. Embed query + all hadiths → cosine similarity
6. Return top-k RetrievedResult (correct hadith↔sharh pairing)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.services.arabic_utils import normalize_arabic

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class RetrievedResult:
    """One retrieved candidate, ready for the prompt."""
    hadith: str
    sharh: str
    rawy: str
    source: str
    hokm: str
    page_id: str = ""
    chunk_text: str = ""          # sharh chunk text
    similarity_score: float = 0.0  # hadith ↔ query cosine similarity

    @classmethod
    def from_parent(
        cls,
        parent: Document,
        hadith_idx: int,
        similarity_score: float = 0.0,
    ) -> "RetrievedResult":
        """Build a result from a parent doc, selecting a specific hadith by index."""
        meta = parent.metadata
        return cls(
            hadith=_at(meta.get("hadith", [""]), hadith_idx),
            sharh=_at(meta.get("sharh", [""]), hadith_idx),
            rawy=_at(meta.get("rawy", [""]), hadith_idx),
            source=_at(meta.get("source", [""]), hadith_idx),
            hokm=_at(meta.get("hokm", [""]), hadith_idx),
            page_id=str(_at(meta.get("page_id", [""]), hadith_idx)),
            chunk_text=parent.page_content,
            similarity_score=similarity_score,
        )


def _at(v, idx: int):
    """Return element at *idx* if list, else value itself."""
    if isinstance(v, list):
        if idx < len(v):
            return v[idx]
        return v[0] if v else ""
    return v


# ── Dense search ───────────────────────────────────────────────────────────────

def _dense_search(vectorstore, query: str, k: int) -> list[tuple[Document, float]]:
    try:
        normalized_query = normalize_arabic(query)
        return vectorstore.similarity_search_with_score(normalized_query, k=k)
    except Exception as exc:
        logger.warning("Dense search failed: %s", exc)
        return []


# ── BM25 search ────────────────────────────────────────────────────────────────

def _bm25_search_docs(
    bm25_index: BM25Okapi,
    all_chunks: list[Document],
    query: str,
    k: int,
) -> list[tuple[Document, float]]:
    from app.services.bm25_index import bm25_search
    hits = bm25_search(bm25_index, query, k)
    results = []
    for idx, score in hits:
        if 0 <= idx < len(all_chunks):
            results.append((all_chunks[idx], score))
    return results


# ── Hadith-level embedding similarity ─────────────────────────────────────────

def _hadith_similarity(
    embedding_model: SentenceTransformer,
    query: str,
    chunks: list[Document],
    top_k: int,
) -> list[RetrievedResult]:
    """
    Extract all hadiths from chunks (or documents), embed them + query,
    rank by cosine similarity, return top-k RetrievedResult.
    """
    # Collect all (hadith_text, chunk_doc, hadith_index) tuples
    candidates: list[tuple[str, Document, int]] = []
    for chunk in chunks:
        hadiths = chunk.metadata.get("hadith", [])
        if isinstance(hadiths, list):
            for i, h in enumerate(hadiths):
                h_text = str(h).strip()
                if h_text:
                    candidates.append((h_text, chunk, i))
        else:
            # Single hadith (not a list)
            h_text = str(hadiths).strip()
            if h_text:
                candidates.append((h_text, chunk, 0))

    if not candidates:
        logger.warning("No hadiths found in %d chunk documents", len(chunks))
        return []

    hadith_texts = [c[0] for c in candidates]

    # Normalize query before generating its embedding
    normalized_query = normalize_arabic(query)

    # Embed query + all hadiths in one batch
    all_texts = [normalized_query] + hadith_texts
    embeddings = embedding_model.encode(
        all_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    query_emb = embeddings[0]  # shape: (dim,)
    hadith_embs = embeddings[1:]  # shape: (N, dim)

    # Cosine similarity (embeddings are already normalized)
    scores = np.dot(hadith_embs, query_emb)

    # Sort by score descending
    ranked_indices = np.argsort(scores)[::-1]

    # Deduplicate by hadith text and take top_k
    seen_hadiths: set[str] = set()
    results: list[RetrievedResult] = []
    for idx in ranked_indices:
        hadith_text, parent_doc, hadith_idx = candidates[idx]
        if hadith_text in seen_hadiths:
            continue
        seen_hadiths.add(hadith_text)
        results.append(
            RetrievedResult.from_parent(
                parent=parent_doc,
                hadith_idx=hadith_idx,
                similarity_score=float(scores[idx]),
            )
        )
        if len(results) >= top_k:
            break

    logger.info(
        "Hadith similarity: %d candidates → %d unique results (top-%d)",
        len(candidates), len(results), top_k,
    )
    return results


# ── Main retrieve ──────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    vectorstore,
    bm25_index: BM25Okapi,
    all_chunks: list[Document],
    embedding_model: SentenceTransformer,
    k: int = 5,
    fetch_k: int = 25,
) -> list[RetrievedResult]:
    """
    Full hybrid retrieve pipeline:
    1. Dense search (Qdrant) on sharh
    2. BM25 search on sharh
    3. Merge candidates (deduplicate)
    4. Hadith-level embedding similarity → top-k
    """
    # 1 & 2: search
    dense_results = _dense_search(vectorstore, query, fetch_k)
    bm25_results = _bm25_search_docs(bm25_index, all_chunks, query, fetch_k)

    # 3: merge (deduplicate by chunk idx)
    seen_ids: set[int] = set()
    merged: list[Document] = []

    for doc, _score in dense_results:
        idx = doc.metadata.get("idx", -1)
        if idx not in seen_ids:
            seen_ids.add(idx)
            merged.append(doc)

    for doc, _score in bm25_results:
        idx = doc.metadata.get("idx", -1)
        if idx not in seen_ids:
            seen_ids.add(idx)
            merged.append(doc)

    if not merged:
        logger.warning("No candidates found for query: %r", query)
        return []

    # 4: hadith-level embedding similarity
    results = _hadith_similarity(embedding_model, query, merged, top_k=k)

    logger.info("Retrieved %d results for query: %r", len(results), query[:60])
    return results


# ── Category-filtered retrieve ─────────────────────────────────────────────────

def _dense_search_filtered(
    vectorstore,
    query: str,
    k: int,
    category_names: list[str],
) -> list[tuple[Document, float]]:
    """Dense search with Qdrant payload filter on high_level_categories."""
    from qdrant_client.http.models import FieldCondition, Filter, MatchAny

    qdrant_filter = Filter(
        should=[
            FieldCondition(
                key="metadata.high_level_categories",
                match=MatchAny(any=category_names),
            )
        ]
    )

    try:
        return vectorstore.similarity_search_with_score(
            query, k=k, filter=qdrant_filter,
        )
    except Exception as exc:
        logger.warning("Filtered dense search failed: %s — falling back to unfiltered", exc)
        return _dense_search(vectorstore, query, k)


def _bm25_search_filtered(
    bm25_index: BM25Okapi,
    all_chunks: list[Document],
    query: str,
    k: int,
    category_names: set[str],
) -> list[tuple[Document, float]]:
    """BM25 search, then filter results to only chunks matching categories."""
    from app.services.bm25_index import bm25_search
    hits = bm25_search(bm25_index, query, k * 3)  # fetch extra, then filter
    results = []
    for idx, score in hits:
        if 0 <= idx < len(all_chunks):
            chunk = all_chunks[idx]
            chunk_cats = set(chunk.metadata.get("high_level_categories", []))
            if chunk_cats & category_names:
                results.append((chunk, score))
                if len(results) >= k:
                    break
    return results


def retrieve_with_category_filter(
    query: str,
    vectorstore,
    bm25_index: BM25Okapi,
    all_chunks: list[Document],
    embedding_model: SentenceTransformer,
    qdrant_client,
    embedding_model_name: str,
    category_collection_name: str = "hadith_categories",
    category_top_k: int = 5,
    k: int = 5,
    fetch_k: int = 25,
) -> list[RetrievedResult]:
    """
    Multi-stage retrieval:
    1. Retrieve top category_top_k categories via semantic search.
    2. Filter dense and BM25 searches to matching categories.
    3. Merge → hadith embedding similarity → top-k.
    """
    from app.services.category_retriever import retrieve_categories

    # Stage 1: category matching
    matched = retrieve_categories(
        query=query,
        client=qdrant_client,
        model_name=embedding_model_name,
        collection_name=category_collection_name,
        top_k=category_top_k,
    )
    category_names = [name for name, _score in matched]

    if not category_names:
        logger.warning("No categories matched — falling back to unfiltered retrieval")
        return retrieve(
            query, vectorstore, bm25_index, all_chunks,
            embedding_model, k=k, fetch_k=fetch_k,
        )

    logger.info("Stage 1 categories: %s", category_names)

    # Stage 2: filtered hybrid search
    dense_results = _dense_search_filtered(vectorstore, query, fetch_k, category_names)
    bm25_results = _bm25_search_filtered(
        bm25_index, all_chunks, query, fetch_k, set(category_names),
    )

    # Merge
    seen_ids: set[int] = set()
    merged: list[Document] = []

    for doc, _score in dense_results:
        idx = doc.metadata.get("idx", -1)
        if idx not in seen_ids:
            seen_ids.add(idx)
            merged.append(doc)

    for doc, _score in bm25_results:
        idx = doc.metadata.get("idx", -1)
        if idx not in seen_ids:
            seen_ids.add(idx)
            merged.append(doc)

    if not merged:
        logger.warning("No candidates after category filter — falling back to unfiltered")
        return retrieve(
            query, vectorstore, bm25_index, all_chunks,
            embedding_model, k=k, fetch_k=fetch_k,
        )

    # Hadith-level embedding similarity
    results = _hadith_similarity(embedding_model, query, merged, top_k=k)

    logger.info(
        "Retrieved %d results (category-filtered) for query: %r",
        len(results), query[:60],
    )
    return results
