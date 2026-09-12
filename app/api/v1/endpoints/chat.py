"""POST /api/v1/chat — multi-turn conversation: query rewrite/decomposition
-> batched triage -> parallel retrieval -> single pooled generation, with
history.

See app.services.query_rewriter, app.agents.triage_agent.classify_batch,
app.agents.retrieval_agent.retrieve_all, and
app.agents.helper.answer_generation.generate_multi_answer.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from langsmith import traceable

from app.agents.helper.answer_generation import generate_multi_answer, INSUFFICIENT_EVIDENCE_MESSAGE
from app.agents.triage_agent import _NON_ISLAMIC_RESPONSE
from app.core.exceptions import LLMError, PipelineNotReadyError, RetrievalError
from app.providers import Message
from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse
from app.services.query_rewriter import rewrite_query

logger = logging.getLogger(__name__)
router = APIRouter()

_CLARIFICATION_MESSAGE = (
    "سؤالك يحتاج إلى توضيح أكثر حتى أستطيع تحديد المصدر المناسب للإجابة — "
    "هل يمكنك تفصيل سؤالك أكثر؟"
)


@router.post("/chat", response_model=ChatResponse)
@traceable(name="chat_request")
async def chat_endpoint(body: ChatRequest, request: Request) -> ChatResponse:
    state = request.app.state
    for attr in ("response_llm", "task_llm", "triage_agent", "retrieval_agent", "session_store"):
        if getattr(state, attr, None) is None:
            raise PipelineNotReadyError(attr)

    session_id = body.session_id or state.session_store.new_session_id()
    short_history = await state.session_store.get_last_k(session_id, k=2)
    long_history = await state.session_store.get_last_k(session_id, k=4)

    # --- Query rewrite + decomposition -----------------------------------
    try:
        questions = await rewrite_query(
            query=body.message, llm=state.task_llm, temperature=0.2, history=short_history
        )
    except Exception as exc:
        logger.exception("Query rewrite error: %s", exc)
        raise LLMError(str(exc)) from exc

    # --- Batched triage ----------------------------------------------------
    try:
        triage_results = await state.triage_agent.classify_batch(questions, conversation_history=[])
    except Exception as exc:
        logger.exception("Triage error: %s", exc)
        raise LLMError(str(exc)) from exc

    # --- Handle the "nothing actionable at all" cases up front -----------
    # If every sub-question is non-Islamic / chitchat / needs clarification,
    # there's nothing to retrieve or generate — respond immediately.
    actionable_idxs = [
        i for i, t in enumerate(triage_results)
        if t.has_actionable_request and not t.needs_clarification
    ]

    if not actionable_idxs:
        answer_parts = []
        any_needs_clarification = False
        for q, t in zip(questions, triage_results):
            if t.is_non_islamic:
                answer_parts.append(_NON_ISLAMIC_RESPONSE)
            elif t.needs_clarification:
                answer_parts.append(_CLARIFICATION_MESSAGE)
                any_needs_clarification = True
            else:
                answer_parts.append(t.canned_response())

        answer = answer_parts[0] if len(answer_parts) == 1 else "\n\n".join(answer_parts)

        await state.session_store.append_turn(
            session_id=session_id,
            user_message=Message(role="user", content=body.message),
            assistant_message=Message(role="assistant", content=answer),
        )

        return ChatResponse(
            answer=answer,
            sources=[],
            categories=[],
            chitchat_type=triage_results[0].chitchat_type if len(triage_results) == 1 else "none",
            needs_clarification=any_needs_clarification,
            session_id=session_id,
        )

    # --- Parallel retrieval, only for actionable sub-questions -----------
    try:
        docs_per_question = await state.retrieval_agent.retrieve_all(questions, triage_results)
    except Exception as exc:
        logger.exception("Retrieval error: %s", exc)
        raise RetrievalError(str(exc)) from exc

    # retrieve_all slices with the agent's configured top_k, not body.top_k;
    # re-slice here so per-request top_k is actually respected.
    docs_per_question = [docs[: body.top_k] for docs in docs_per_question]

    # --- Single pooled generation over all actionable sub-questions ------
    try:
        answer, used_sources = await generate_multi_answer(
            llm=state.response_llm,
            questions=questions,
            docs_per_question=docs_per_question,
            history=long_history,
        )
    except Exception as exc:
        logger.exception("LLM generation error: %s", exc)
        raise LLMError(str(exc)) from exc

    # If some sub-questions were non-Islamic/chitchat/clarification-needed
    # alongside actionable ones, prepend/append their responses so nothing
    # gets silently dropped.
    extra_parts = []
    for i, t in enumerate(triage_results):
        if i in actionable_idxs:
            continue
        if t.is_non_islamic:
            extra_parts.append(_NON_ISLAMIC_RESPONSE)
        elif t.needs_clarification:
            extra_parts.append(_CLARIFICATION_MESSAGE)
        elif not t.has_actionable_request:
            extra_parts.append(t.canned_response())

    if extra_parts:
        answer = answer + "\n\n" + "\n\n".join(extra_parts)

    all_categories = list({
        c for t in triage_results if t.has_actionable_request for c in t.categories
    })
    any_needs_clarification = any(t.needs_clarification for t in triage_results)

    await state.session_store.append_turn(
        session_id=session_id,
        user_message=Message(role="user", content=body.message),
        assistant_message=Message(role="assistant", content=answer),
    )

    return ChatResponse(
        answer=answer,
        sources=used_sources,
        categories=all_categories,
        chitchat_type=triage_results[0].chitchat_type if len(triage_results) == 1 else "none",
        needs_clarification=any_needs_clarification,
        session_id=session_id,
    )