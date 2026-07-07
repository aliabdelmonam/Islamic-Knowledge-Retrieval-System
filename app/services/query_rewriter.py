"""
LLM query rewriting: converts Egyptian/colloquial Arabic to MSA.
Falls back to original query if rewriting fails.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """\
You are an expert Arabic linguist specializing in Islamic texts.
Your task is to rewrite the given Arabic question into formal Modern Standard Arabic (MSA),
preserving the original meaning accurately. Remove any colloquial expressions,
slang, or dialect-specific phrases while maintaining the semantic content.

Question: {question}

Provide ONLY the rewritten question in MSA, nothing else.
"""


def rewrite_query(question: str, groq_api_key: str, model: str = "llama-3.3-70b-versatile") -> str:
    """
    Rewrite *question* from colloquial Arabic to MSA via Groq.
    Returns the original question on any error.
    """
    try:
        from groq import Groq

        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": _REWRITE_PROMPT.format(question=question)},
            ],
            temperature=0.1,
            max_tokens=256,
        )
        rewritten = response.choices[0].message.content or ""
        rewritten = rewritten.strip().strip('"').strip("'")
        if rewritten:
            logger.info("Query rewritten: %r → %r", question, rewritten)
            return rewritten
    except Exception as exc:
        logger.warning("Query rewriting failed (%s), using original query.", exc)
    return question
