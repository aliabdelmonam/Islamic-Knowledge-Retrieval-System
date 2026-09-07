"""POST /api/v1/retrieve — triage -> multi-source retrieval (general/hadith/quran), no LLM generation."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.core.exceptions import LLMError, PipelineNotReadyError, RetrievalError
from app.schemas.request import RetrieveRequest
from app.schemas.response import RetrieveResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_endpoint(body: RetrieveRequest, request: Request) -> RetrieveResponse:
    state = request.app.state
    for attr in ("triage_agent", "retrieval_agent"):
        if getattr(state, attr, None) is None:
            raise PipelineNotReadyError(attr)

    try:
        triage = await state.triage_agent.classify(body.query)
    except Exception as exc:
        logger.exception("Triage error: %s", exc)
        raise LLMError(str(exc)) from exc

    if not triage.has_actionable_request or triage.needs_clarification:
        return RetrieveResponse(
            query=body.query,
            categories=triage.categories,
            needs_clarification=triage.needs_clarification,
            results=[],
        )

    try:
        retrieval = await state.retrieval_agent.run(body.query, triage)
    except Exception as exc:
        logger.exception("Retrieval error: %s", exc)
        raise RetrievalError(str(exc)) from exc

    results = retrieval.flattened()[: body.top_k]

    return RetrieveResponse(
        query=body.query,
        categories=triage.categories,
        needs_clarification=False,
        results=results,
    )