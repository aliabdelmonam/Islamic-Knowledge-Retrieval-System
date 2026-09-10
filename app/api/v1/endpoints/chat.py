"""POST /api/v1/chat — multi-turn conversation: triage -> retrieval -> LLM generation, with history.

Falls back to whitelist-restricted web search (SearchAgent) if the internal
sources aren't enough to answer; see app.agents.helper.answer_generation.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.agents.helper.answer_generation import generate_answer
from app.core.exceptions import LLMError, PipelineNotReadyError, RetrievalError
from app.providers import Message
from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest, request: Request) -> ChatResponse:
    state = request.app.state
    for attr in ("response_llm","task_llm", "triage_agent", "retrieval_agent", "session_store"):
        if getattr(state, attr, None) is None:
            raise PipelineNotReadyError(attr)

    session_id = body.session_id or state.session_store.new_session_id()
    history = await state.session_store.get_last_k(session_id, k=10)

    try:
        triage = await state.triage_agent.classify(body.message, conversation_history=history)
    except Exception as exc:
        logger.exception("Triage error: %s", exc)
        raise LLMError(str(exc)) from exc

    if not triage.has_actionable_request:
        answer = triage.canned_response()
        await state.session_store.append_turn(
            session_id,
            Message(role="user", content=body.message),
            Message(role="assistant", content=answer),
        )
        return ChatResponse(
            answer=answer,
            sources=[],
            categories=[],
            chitchat_type=triage.chitchat_type,
            needs_clarification=False,
            session_id=session_id,
        )

    if triage.needs_clarification:
        answer = (
            "سؤالك يحتاج إلى توضيح أكثر حتى أستطيع تحديد المصدر المناسب للإجابة — "
            "هل يمكنك تفصيل سؤالك أكثر؟"
        )
        await state.session_store.append_turn(
            session_id,
            Message(role="user", content=body.message),
            Message(role="assistant", content=answer),
        )
        return ChatResponse(
            answer=answer,
            sources=[],
            categories=[],
            chitchat_type=triage.chitchat_type,
            needs_clarification=True,
            session_id=session_id,
        )

    try:
        retrieval = await state.retrieval_agent.run(body.message, triage)
    except Exception as exc:
        logger.exception("Retrieval error: %s", exc)
        raise RetrievalError(str(exc)) from exc

    docs = retrieval.flattened()[: body.top_k]

    try:
        answer, used_sources = await generate_answer(
            llm=state.response_llm,
            search_agent=getattr(state, "search_agent", None),
            query=body.message,
            docs=docs,
            history=history,
        )
    except Exception as exc:
        logger.exception("LLM generation error: %s", exc)
        raise LLMError(str(exc)) from exc

    await state.session_store.append_turn(
    session_id=session_id,
    user_message=Message(role="user", content=body.question),
    assistant_message=Message(role="assistant", content=answer),
)

    return ChatResponse(
        answer=answer,
        sources=used_sources,
        categories=triage.categories,
        chitchat_type=triage.chitchat_type,
        needs_clarification=False,
        session_id=session_id,
    )