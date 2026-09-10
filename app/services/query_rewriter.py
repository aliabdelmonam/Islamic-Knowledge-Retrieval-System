"""
LLM query rewriting: converts Egyptian/colloquial Arabic to MSA.
Falls back to original query if rewriting fails.
"""
from __future__ import annotations

from typing import Any,Optional
import logging
import re
from app.providers import GenerationClient,Message,GenerationResponse

logger = logging.getLogger(__name__)

_REWRITE_PROMPT_WITH_HISTORY = """\
You are an expert Arabic linguist specializing in Islamic texts.
Your task is to rewrite the given Arabic question into formal Modern Standard Arabic (MSA),
preserving the original meaning accurately. Remove any colloquial expressions,
slang, or dialect-specific phrases while maintaining the semantic content.

Use the conversation history below only to resolve pronouns or implicit
references in the question (e.g. "the second one", "that hadith") so the
rewritten question is self-contained. Do not answer the question, and do not
add information the user didn't ask for.

Conversation history:
{history}

Question: {question}

Provide ONLY the rewritten question in MSA, nothing else.
"""

_REWRITE_PROMPT = """\
You are an expert Arabic linguist specializing in Islamic texts.
Your task is to rewrite the given Arabic question into formal Modern Standard Arabic (MSA),
preserving the original meaning accurately. Remove any colloquial expressions,
slang, or dialect-specific phrases while maintaining the semantic content.

If the conversation history below contains context needed to resolve pronouns
or implicit references in the question (e.g. "the second one", "that hadith"),
use it to make the rewritten question self-contained and unambiguous — but do
not answer the question, and do not add information the user didn't ask for.

Question: {question}

Provide ONLY the rewritten question in MSA, nothing else.
"""


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


async def rewrite_query(query: str,
                   llm: GenerationClient,
                    temperature: float = 0.2,
                    history: Optional[list[Message]] = None,
                    **kwargs: Any) -> str:
    """
    Rewrite *query* from colloquial Arabic to MSA via Groq.
    Returns the original query on any error.
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

        response = await llm.generate(
            messages=messages,
            temperature=temperature,
            output_schema=None,
            **kwargs
        )

        rewritten = _extract_text(response.text).strip().strip('"').strip("'")

        if rewritten:
            logger.info("Query rewritten: %r → %r", query, rewritten)
            return rewritten
        else:
            logger.warning("Query rewriting returned empty, using original query.")
            return query

    except Exception as exc:
        logger.warning("Query rewriting failed (%s), using original query.", exc)
        return query

