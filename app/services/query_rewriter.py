"""
LLM query rewriting: converts Egyptian/colloquial Arabic to MSA, and splits
a message into multiple standalone questions if the user asked more than one
unrelated question at once.
Falls back to [original query] if rewriting/decomposition fails.
"""
from __future__ import annotations

from typing import Any, Optional
import logging
from pydantic import BaseModel, Field
from app.providers import GenerationClient, Message, GenerationResponse

logger = logging.getLogger(__name__)

_REWRITE_PROMPT_WITH_HISTORY = """\
أنت خبير لغوي عربي متخصص في النصوص الإسلامية والفقهية.

مهمتك جزءان:
١. إذا كانت رسالة المستخدم تحتوي على أكثر من سؤال منفصل وغير مرتبط بالموضوع
   (مثل سؤالين عن حكمين شرعيين مختلفين تمامًا)، افصلها إلى أسئلة مستقلة.
   لا تفصل الرسالة إذا كانت سؤالاً واحدًا حتى لو احتوت على حرف "و" أو فاصلة؛
   الأصل عدم الفصل إلا إذا كان هناك موضوعان مختلفان بوضوح تام.
٢. أعد صياغة كل سؤال (سواء كان واحدًا أو أكثر) بالفصحى (اللغة العربية الفصحى
   المعيارية)، مع إزالة العامية والتعبيرات الدارجة، مع الحفاظ التام على المعنى الأصلي.

استخدم سجل المحادثة أدناه فقط لحل أي مرجع غامض في السؤال (مثل "الحديث الثاني"
أو "ذلك الحكم") لجعل السؤال المعاد صياغته مكتملاً بذاته. لا تُجب عن السؤال، ولا
تُضف معلومات لم يطلبها المستخدم.

سجل المحادثة:
{history}

رسالة المستخدم: {question}

أخرج فقط قائمة بالأسئلة المعاد صياغتها بالفصحى.
"""

_REWRITE_PROMPT = """\
أنت خبير لغوي عربي متخصص في النصوص الإسلامية والفقهية.

مهمتك جزءان:
١. إذا كانت رسالة المستخدم تحتوي على أكثر من سؤال منفصل وغير مرتبط بالموضوع
   (مثل سؤالين عن حكمين شرعيين مختلفين تمامًا)، افصلها إلى أسئلة مستقلة.
   لا تفصل الرسالة إذا كانت سؤالاً واحدًا حتى لو احتوت على حرف "و" أو فاصلة؛
   الأصل عدم الفصل إلا إذا كان هناك موضوعان مختلفان بوضوح تام.
٢. أعد صياغة كل سؤال (سواء كان واحدًا أو أكثر) بالفصحى (اللغة العربية الفصحى
   المعيارية)، مع إزالة العامية والتعبيرات الدارجة، مع الحفاظ التام على المعنى الأصلي.

رسالة المستخدم: {question}

أخرج فقط قائمة بالأسئلة المعاد صياغتها بالفصحى.
"""


class RewrittenQuestions(BaseModel):
    questions: list[str] = Field(
        description=(
            "One or more standalone MSA-rewritten questions extracted from "
            "the user's message. If the message contains only one question, "
            "return a list with exactly that one rewritten question."
        ),
        min_length=1,
    )


def _extract_text(content: Any) -> str:
    """response.text can be a plain string or (via LangChain) a list of
    content blocks. Normalize to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def _format_history(history: list[Message]) -> str:
    lines = []
    for m in history:
        speaker = "المستخدم" if m.role == "user" else "المساعد"
        lines.append(f"{speaker}: {m.content}")
    return "\n".join(lines)


async def rewrite_query(
    query: str,
    llm: GenerationClient,
    temperature: float = 0.2,
    history: Optional[list[Message]] = None,
    **kwargs: Any,
) -> list[str]:
    """
    Rewrite *query* from colloquial Arabic to MSA, splitting it into multiple
    standalone questions if the user asked more than one unrelated question.
    Returns [query] unchanged on any error or empty result. The common case
    (a single question) returns a list of length 1.
    """
    try:
        if history:
            prompt = _REWRITE_PROMPT_WITH_HISTORY.format(
                history=_format_history(history),
                question=query,
            )
        else:
            prompt = _REWRITE_PROMPT.format(question=query)

        messages = [Message(role="user", content=prompt)]

        response: GenerationResponse = await llm.generate(
            messages=messages,
            temperature=temperature,
            output_schema=RewrittenQuestions,
            **kwargs,
        )

        raw_text = _extract_text(response.text)
        parsed = RewrittenQuestions.model_validate_json(raw_text)

        cleaned = [q.strip().strip('"').strip("'") for q in parsed.questions if q.strip()]

        if cleaned:
            if len(cleaned) > 1:
                logger.info("Query decomposed into %d sub-questions: %r", len(cleaned), cleaned)
            else:
                logger.info("Query rewritten: %r → %r", query, cleaned[0])
            return cleaned
        else:
            logger.warning("Query rewriting returned empty list, using original query.")
            return [query]

    except Exception as exc:
        logger.warning("Query rewriting failed (%s), using original query.", exc)
        return [query]