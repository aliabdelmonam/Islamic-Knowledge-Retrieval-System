"""
RAG chain: builds the prompt, calls the LLM, and returns (answer, sources).
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from app.services.retriever import RetrievedResult

logger = logging.getLogger(__name__)


# ── Prompt ─────────────────────────────────────────────────────────────────────

def build_rag_prompt(system_role: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_role),
            HumanMessagePromptTemplate.from_template(
                "السياق:\n{context}\n\nالسؤال: {question}"
            ),
        ]
    )


def format_docs(results: list[RetrievedResult]) -> str:
    """Serialize retrieved results into the prompt context block."""
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[{i}] الحديث: {r.hadith}\n"
            f"    الشرح: {r.sharh}\n"
            f"    الراوي: {r.rawy}\n"
            f"    المصدر: {r.source}\n"
            f"    الحكم: {r.hokm}"
        )
    return "\n\n".join(parts)


# ── Chain ──────────────────────────────────────────────────────────────────────

def build_rag_chain(llm, system_role: str):
    """Return an LCEL chain: context+question → answer string."""
    prompt = build_rag_prompt(system_role)
    return prompt | llm | StrOutputParser()


# ── Ask ────────────────────────────────────────────────────────────────────────

def ask(
    question: str,
    chain,
    results: list[RetrievedResult],
) -> str:
    """Invoke the RAG chain and return the answer string."""
    if not results:
        logger.warning("No retrieved results to pass to LLM.")
        return "لم أجد معلومات كافية للإجابة على هذا السؤال."

    context = format_docs(results)
    logger.info("Calling LLM for question: %r", question[:60])
    answer: str = chain.invoke({"context": context, "question": question})
    return answer.strip()
