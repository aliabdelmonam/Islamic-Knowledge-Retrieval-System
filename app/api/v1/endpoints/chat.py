"""POST /api/v1/chat — multi-turn conversation: triage -> retrieval -> LLM generation, with history."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.agents.retrieval_agent import RetrievedDocument
from app.agents.helper.retrieval_agent_system_prompt import RETRIEVAL_SYSTEM_PROMPT
from app.core.exceptions import LLMError, PipelineNotReadyError, RetrievalError
from app.providers import Message
from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _build_context(docs: list[RetrievedDocument]) -> str:
    blocks = []
    for i, doc in enumerate(docs, 1):
        ref = doc.source_ref or doc.metadata.get("title", "")
        blocks.append(f"[{i}] (الفئة: {doc.category.value}, المصدر: {ref})\n{doc.text}")
    return "\n\n".join(blocks)


async def _generate_answer(
    llm,
    message: str,
    history: list[Message],
    docs: list[RetrievedDocument],
) -> str:
    context_note = (
        f"المصادر:\n{_build_context(docs)}" if docs else "لا توجد مصادر مسترجعة لهذه الرسالة."
    )
    prompt = f"{context_note}\n\nرسالة المستخدم الحالية: {message}"

    response = await llm.generate(
        messages=[
            Message(role="system", content=ANSWER_SYSTEM_PROMPT),
            *history,
            Message(role="user", content=prompt),
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return response.text


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest, request: Request) -> ChatResponse:
    state = request.app.state
    for attr in ("llm", "triage_agent", "retrieval_agent", "session_store"):
        if getattr(state, attr, None) is None:
            raise PipelineNotReadyError(attr)

    session_id = body.session_id or state.session_store.new_session_id()
    history = await state.session_store.get_history(session_id)

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
        answer = await _generate_answer(state.response_llm, body.message, history, docs)
    except Exception as exc:
        logger.exception("LLM generation error: %s", exc)
        raise LLMError(str(exc)) from exc

    await state.session_store.append_turn(
        session_id,
        Message(role="user", content=body.message),
        Message(role="assistant", content=answer),
    )

    return ChatResponse(
        answer=answer,
        sources=docs,
        categories=triage.categories,
        chitchat_type=triage.chitchat_type,
        needs_clarification=False,
        session_id=session_id,
    )