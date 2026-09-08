"""POST /api/v1/ask — single-turn call into the agentic answer pipeline."""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Request

from app.core.exceptions import LLMError, PipelineNotReadyError
from app.schemas.request import AskRequest
from app.schemas.response import AskResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_endpoint(body: AskRequest, request: Request) -> AskResponse:
    state = request.app.state
    if getattr(state, "answer_agent", None) is None:
        raise PipelineNotReadyError("answer_agent")

    session_id = body.session_id or str(uuid.uuid4())

    try:
        result = await state.answer_agent.answer(
            body.question,
            conversation_history=[],  # /ask is stateless by design — use /chat for history
            top_k=body.top_k,
        )
    except Exception as exc:
        logger.exception("Answer agent error: %s", exc)
        raise LLMError(str(exc)) from exc

    return AskResponse(
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