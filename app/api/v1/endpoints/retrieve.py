"""POST /api/v1/retrieve — retrieve hadith candidates without LLM."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.core.exceptions import PipelineNotReadyError, RetrievalError
from app.schemas.request import RetrieveRequest
from app.schemas.response import RetrievedItem, RetrieveResponse
from app.services import retriever as retriever_svc

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_endpoint(body: RetrieveRequest, request: Request) -> RetrieveResponse:
    state = request.app.state

    # Guard: all retrieval components must be ready
    for attr in ("vectorstore", "bm25_index", "all_chunks", "parent_store", "embedding_model"):
        if getattr(state, attr, None) is None:
            raise PipelineNotReadyError(attr)

    query = body.query
    try:
        results = retriever_svc.retrieve(
            query=query,
            vectorstore=state.vectorstore,
            bm25_index=state.bm25_index,
            all_chunks=state.all_chunks,
            parent_store=state.parent_store,
            embedding_model=state.embedding_model,
            k=body.k,
        )
    except Exception as exc:
        logger.exception("Retrieval error: %s", exc)
        raise RetrievalError(str(exc))

    items = [
        RetrievedItem(
            hadith=r.hadith,
            sharh=r.sharh,
            rawy=r.rawy,
            source=r.source,
            hokm=r.hokm,
            page_id=r.page_id,
            similarity_score=r.similarity_score,
        )
        for r in results
    ]
    return RetrieveResponse(results=items, query_used=query)
