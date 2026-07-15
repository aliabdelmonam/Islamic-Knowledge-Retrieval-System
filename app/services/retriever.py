"""
Hybrid retriever: dense (Qdrant) + BM25 + cross-encoder reranking.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.services.arabic_utils import normalize_arabic

logger = logging.getLogger(__name__)


@dataclass
class RetrievedResult:
    """One retrieved + reranked candidate, ready for the prompt."""
    hadith: str
    sharh: str
    rawy: str
    source: str
    hokm: str
    page_id: str = ""
    chunk_text: str = ""
    rerank_score: float = 0.0
    dense_score: float = 0.0
    bm25_score: float = 0.0

    @classmethod
    def from_chunk(cls, chunk: Document, rerank_score: float = 0.0) -> "RetrievedResult":
        meta = chunk.metadata
        return cls(
            hadith=_first(meta.get("hadith", [""])),
            sharh=_first(meta.get("sharh", [""])),
            rawy=_first(meta.get("rawy", [""])),
            source=_first(meta.get("source", [""])),
            hokm=_first(meta.get("hokm", [""])),
            page_id=str(_first(meta.get("page_id", [""]))),
            chunk_text=chunk.page_content,
            rerank_score=rerank_score,
        )


def _first(v):
    """Return first element if list, else value itself."""
    if isinstance(v, list):
        return v[0] if v else ""
    return v


# ── Dense search ───────────────────────────────────────────────────────────────

def _dense_search(vectorstore, query: str, k: int) -> list[tuple[Document, float]]:
    try:
        return vectorstore.similarity_search_with_score(query, k=k)
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


# ── Parent expansion ───────────────────────────────────────────────────────────

def _expand_to_parent(
    chunks: list[Document],
    parent_store: list[Document],
) -> list[Document]:
    """Replace child chunks by their parent documents (deduplicated)."""
    seen: set[int] = set()
    parents: list[Document] = []
    for chunk in chunks:
        pid = chunk.metadata.get("parent_id")
        if pid is not None and pid not in seen:
            seen.add(pid)
            if 0 <= pid < len(parent_store):
                parents.append(parent_store[pid])
            else:
                parents.append(chunk)  # fallback: keep chunk as-is
        elif pid is None and id(chunk) not in seen:
            seen.add(id(chunk))
            parents.append(chunk)
    return parents


# ── Rerank ─────────────────────────────────────────────────────────────────────

def _rerank(
    reranker,
    query: str,
    candidates: list[Document],
    top_k: int,
    batch_size: int = 128,
) -> list[tuple[Document, float]]:
    from app.services.reranker import rerank

    passages = [c.page_content for c in candidates]
    scores = rerank(reranker, query, passages, batch_size=batch_size)
    paired = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return paired[:top_k]


# ── Main retrieve ──────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    vectorstore,
    bm25_index: BM25Okapi,
    all_chunks: list[Document],
    parent_store: list[Document],
    reranker,
    k: int = 5,
    fetch_k: int = 25,
    reranker_batch_size: int = 128,
) -> list[RetrievedResult]:
    """
    Full hybrid retrieve pipeline:
    1. Dense search (Qdrant)
    2. BM25 search
    3. Merge candidates
    4. Parent expansion
    5. Cross-encoder rerank
    6. Return top-k RetrievedResult objects
    """
    # 1 & 2: search
    dense_results = _dense_search(vectorstore, query, fetch_k)
    bm25_results = _bm25_search_docs(bm25_index, all_chunks, query, fetch_k)

    # 3: merge (dense takes priority; add BM25 extras)
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

    # 4: expand to parents
    expanded = _expand_to_parent(merged, parent_store)

    # 5: rerank
    reranked = _rerank(reranker, query, expanded, top_k=k, batch_size=reranker_batch_size)

    # 6: build results
    results: list[RetrievedResult] = []
    for doc, score in reranked:
        r = RetrievedResult.from_chunk(doc, rerank_score=score)
        results.append(r)

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
    parent_store: list[Document],
    reranker,
    qdrant_client,
    embedding_model_name: str,
    category_collection_name: str = "hadith_categories",
    category_top_k: int = 5,
    k: int = 5,
    fetch_k: int = 25,
    reranker_batch_size: int = 128,
) -> list[RetrievedResult]:
    """
    Multi-stage retrieval:
    1. Retrieve top category_top_k categories via semantic search.
    2. Filter dense and BM25 searches to matching categories.
    3. Merge → parent expansion → rerank → return top-k.
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
            query, vectorstore, bm25_index, all_chunks, parent_store, reranker,
            k=k, fetch_k=fetch_k, reranker_batch_size=reranker_batch_size,
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
            query, vectorstore, bm25_index, all_chunks, parent_store, reranker,
            k=k, fetch_k=fetch_k, reranker_batch_size=reranker_batch_size,
        )

    # Parent expansion
    expanded = _expand_to_parent(merged, parent_store)

    # Rerank
    reranked = _rerank(reranker, query, expanded, top_k=k, batch_size=reranker_batch_size)

    # Build results
    results: list[RetrievedResult] = []
    for doc, score in reranked:
        r = RetrievedResult.from_chunk(doc, rerank_score=score)
        results.append(r)

    logger.info(
        "Retrieved %d results (category-filtered) for query: %r",
        len(results), query[:60],
    )
    return results
