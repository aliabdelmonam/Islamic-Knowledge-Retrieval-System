"""POST /api/v1/ask — full RAG: retrieve + rerank + LLM generation."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.core.config import settings
from app.core.exceptions import LLMError, PipelineNotReadyError, RetrievalError
from app.schemas.request import AskRequest
from app.schemas.response import AskResponse, RetrievedItem
from app.services import chain as chain_svc
from app.services import retriever as retriever_svc

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_endpoint(body: AskRequest, request: Request) -> AskResponse:
    state = request.app.state

    # Guard: all components must be ready
    for attr in ("vectorstore", "bm25_index", "all_chunks", "parent_store", "reranker", "rag_chain"):
        if getattr(state, attr, None) is None:
            raise PipelineNotReadyError(attr)

    question = body.question
    query_rewritten: str | None = None

    # Optional query rewriting
    if body.rewrite and settings.groq_api_key:
        from app.services.query_rewriter import rewrite_query
        rewritten = rewrite_query(question, settings.groq_api_key, settings.query_rewrite_model)
        if rewritten != question:
            query_rewritten = rewritten
            question = rewritten

    # Retrieve
    try:
        results = retriever_svc.retrieve(
            query=question,
            vectorstore=state.vectorstore,
            bm25_index=state.bm25_index,
            all_chunks=state.all_chunks,
            parent_store=state.parent_store,
            reranker=state.reranker,
            k=body.k,
        )
    except Exception as exc:
        logger.exception("Retrieval error: %s", exc)
        raise RetrievalError(str(exc))

    # Generate
    try:
        answer = chain_svc.ask(
            question=body.question,   # original for generation
            chain=state.rag_chain,
            results=results,
        )
    except Exception as exc:
        logger.exception("LLM error: %s", exc)
        raise LLMError(str(exc))

    sources = [
        RetrievedItem(
            hadith=r.hadith,
            sharh=r.sharh,
            rawy=r.rawy,
            source=r.source,
            hokm=r.hokm,
            page_id=r.page_id,
            rerank_score=r.rerank_score,
        )
        for r in results
    ]

    return AskResponse(
        answer=answer,
        sources=sources,
        query_rewritten=query_rewritten,
    )
