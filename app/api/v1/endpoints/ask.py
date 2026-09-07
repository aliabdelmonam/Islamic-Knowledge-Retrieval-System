"""POST /api/v1/ask — triage -> multi-source retrieval (general/hadith/quran) -> LLM generation."""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Request

from app.agents.retrieval_agent import RetrievedDocument
from app.agents.helper.retrieval_agent_system_prompt import RETRIEVAL_SYSTEM_PROMPT
from app.core.exceptions import LLMError, PipelineNotReadyError, RetrievalError
from app.providers import Message
from app.schemas.request import AskRequest
from app.schemas.response import AskResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _build_context(docs: list[RetrievedDocument]) -> str:
    blocks = []
    for i, doc in enumerate(docs, 1):
        ref = doc.source_ref or doc.metadata.get("title", "")
        blocks.append(
            f"[{i}] (الفئة: {doc.category.value}, المصدر: {ref})\n{doc.text}"
        )
    return "\n\n".join(blocks)


async def _generate_answer(llm, question: str, docs: list[RetrievedDocument]) -> str:
    if not docs:
        return "لم أجد مصادر كافية للإجابة على هذا السؤال بدقة."

    prompt = f"السؤال: {question}\n\nالمصادر:\n{_build_context(docs)}\n\nالإجابة:"

    response = await llm.generate(
        messages=[
            Message(role="system", content=RETRIEVAL_SYSTEM_PROMPT),
            Message(role="user", content=prompt),
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return response.text


@router.post("/ask", response_model=AskResponse)
async def ask_endpoint(body: AskRequest, request: Request) -> AskResponse:
    state = request.app.state
    for attr in ("llm", "triage_agent", "retrieval_agent"):
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
        answer = await _generate_answer(state.llm, body.question, docs)
    except Exception as exc:
        logger.exception("LLM generation error: %s", exc)
        raise LLMError(str(exc)) from exc

    return AskResponse(
        answer=answer,
        sources=docs,
        categories=triage.categories,
        chitchat_type=triage.chitchat_type,
        needs_clarification=False,
        session_id=session_id,
    )