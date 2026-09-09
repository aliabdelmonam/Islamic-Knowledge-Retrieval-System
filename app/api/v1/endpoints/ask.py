"""POST /api/v1/ask — triage -> multi-source retrieval (general/hadith/quran) -> LLM generation.

Falls back to whitelist-restricted web search (SearchAgent) if the internal
sources aren't enough to answer; see app.agents.helper.answer_generation.
"""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Request

from app.agents.helper.answer_generation import generate_answer
from app.core.exceptions import LLMError, PipelineNotReadyError, RetrievalError
from app.schemas.request import AskRequest
from app.schemas.response import AskResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_endpoint(body: AskRequest, request: Request) -> AskResponse:
    state = request.app.state
    for attr in ("response_llm","task_llm", "triage_agent", "retrieval_agent"):
        if getattr(state, attr, None) is None:
            raise PipelineNotReadyError(attr)

    session_id = body.session_id or str(uuid.uuid4())

    try:
        triage = await state.triage_agent.classify(body.question)
    except Exception as exc:
        logger.exception("Triage error: %s", exc)
        raise LLMError(str(exc)) from exc

    if not triage.has_actionable_request:
        return AskResponse(
            answer=triage.canned_response(),
            sources=[],
            categories=[],
            chitchat_type=triage.chitchat_type,
            needs_clarification=False,
            session_id=session_id,
        )

    if triage.needs_clarification:
        return AskResponse(
            answer="سؤالك يحتاج إلى توضيح أكثر حتى أستطيع تحديد المصدر المناسب للإجابة — "
                   "هل يمكنك تفصيل سؤالك أكثر؟",
            sources=[],
            categories=[],
            chitchat_type=triage.chitchat_type,
            needs_clarification=True,
            session_id=session_id,
        )

    try:
        retrieval = await state.retrieval_agent.run(body.question, triage)
    except Exception as exc:
        logger.exception("Retrieval error: %s", exc)
        raise RetrievalError(str(exc)) from exc

    docs = retrieval.flattened()[: body.top_k]

    try:
        answer, used_sources = await generate_answer(
            llm=state.response_llm,
            search_agent=getattr(state, "search_agent", None),
            query=body.question,
            docs=docs,
        )
    except Exception as exc:
        logger.exception("LLM generation error: %s", exc)
        raise LLMError(str(exc)) from exc

    return AskResponse(
        answer=answer,
        sources=used_sources,
        categories=triage.categories,
        chitchat_type=triage.chitchat_type,
        needs_clarification=False,
        session_id=session_id,
    )