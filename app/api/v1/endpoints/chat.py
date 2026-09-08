"""POST /api/v1/chat — multi-turn call into the agentic answer pipeline, with session history."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.core.exceptions import LLMError, PipelineNotReadyError
from app.providers import Message
from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest, request: Request) -> ChatResponse:
    state = request.app.state
    for attr in ("answer_agent", "session_store"):
        if getattr(state, attr, None) is None:
            raise PipelineNotReadyError(attr)

    session_id = body.session_id or state.session_store.new_session_id()
    history = await state.session_store.get_history(session_id)

    try:
        result = await state.answer_agent.answer(
            body.message,
            conversation_history=history,
            top_k=body.top_k,
        )
    except Exception as exc:
        logger.exception("Answer agent error: %s", exc)
        raise LLMError(str(exc)) from exc

    await state.session_store.append_turn(
        session_id,
        Message(role="user", content=body.message),
        Message(role="assistant", content=result.answer),
    )

    return ChatResponse(
        answer=result.answer,
        sources=result.sources,
        categories=result.categories,
        chitchat_type=result.chitchat_type,
        needs_clarification=result.needs_clarification,
        resolved_query=result.resolved_query,
        tool_calls_made=result.tool_calls_made,
        is_fallback=result.is_fallback,
        session_id=session_id,
    )