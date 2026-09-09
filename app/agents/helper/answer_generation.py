"""
answer_generation — shared two-stage answer generation used by /ask and /chat.

Stage 1: generate an answer from the internally retrieved docs (general /
hadith / quran). The retrieval system prompt instructs the model to output
the literal string "None" when the supplied sources don't actually answer
the question.

Stage 2: if stage 1 comes back as "None" (or there were no internal docs to
begin with), fall back to SearchAgent — a whitelist-restricted web search
(see islamic_search_domains.DOMAINS) — and retry generation against those
results.

If stage 2 also comes back "None" (or no web results were found, or no
SearchAgent is configured), return INSUFFICIENT_EVIDENCE_MESSAGE and stop.
No further fallback beyond that.
"""
from __future__ import annotations

import logging

from app.agents.retrieval_agent import RetrievedDocument
from app.agents.search_agent import SearchAgent
from app.agents.helper.retrieval_agent_system_prompt import RETRIEVAL_SYSTEM_PROMPT
from app.providers import Message

logger = logging.getLogger(__name__)

# Sentinel the retrieval-answering LLM is instructed (via
# RETRIEVAL_SYSTEM_PROMPT) to emit when the supplied sources don't answer
# the question. Compared case-sensitively after stripping whitespace/periods.
NO_ANSWER_MARKER = "None"

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "لا أملك حاليًا معلومات كافية وموثوقة من المصادر المتاحة للإجابة على هذا السؤال بدقة. "
    "يُرجى مراجعة مصادر موثوقة أخرى (مثل مواقع الإفتاء الرسمية أو كتب أهل العلم المعتمدة) "
    "للتأكد من الإجابة الصحيحة. سنعمل على تحسين قدرتنا على البحث عبر مصادر إضافية قريبًا."
)


def _build_context(docs: list[RetrievedDocument]) -> str:
    blocks = []
    for i, doc in enumerate(docs, 1):
        ref = doc.source_ref or doc.metadata.get("title", "")
        blocks.append(
            f"[{i}] (الفئة: {doc.category.value}, المصدر: {ref})\n{doc.text}"
        )
    return "\n\n".join(blocks)


def _is_no_answer(text: str) -> bool:
    return text.strip().rstrip(".") == NO_ANSWER_MARKER


async def _call_llm(
    llm,
    query: str,
    docs: list[RetrievedDocument],
    history: list[Message] | None,
) -> str:
    prompt = f"السؤال: {query}\n\nالمصادر:\n{_build_context(docs)}\n\nالإجابة:"

    response = await llm.generate(
        messages=[
            Message(role="system", content=RETRIEVAL_SYSTEM_PROMPT),
            *(history or []),
            Message(role="user", content=prompt),
        ],
        temperature=0.2,
        max_tokens=800,
    )
    text = response.text
    if isinstance(text, list):
        text = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in text
        )
    return text


async def generate_answer(
    *,
    llm,
    search_agent: SearchAgent | None,
    query: str,
    docs: list[RetrievedDocument],
    history: list[Message] | None = None,
) -> tuple[str, list[RetrievedDocument]]:
    """Two-stage answer generation with a whitelisted-web-search fallback.

    Returns (answer_text, sources_actually_used_for_that_answer). When the
    canned insufficient-evidence message is returned, sources is [].
    """
    if not docs:
        # Nothing to even ask the LLM about — treat exactly like a "None"
        # verdict and go straight to the web fallback.
        logger.info("No internal docs retrieved — skipping straight to web fallback")
        stage1_text = NO_ANSWER_MARKER
    else:
        stage1_text = await _call_llm(llm, query, docs, history)

    if not _is_no_answer(stage1_text):
        return stage1_text, docs

    logger.info("Internal sources insufficient (LLM returned None) — trying web search")

    if search_agent is None:
        logger.warning("No search_agent configured — cannot fall back to web search")
        return INSUFFICIENT_EVIDENCE_MESSAGE, []

    try:
        web_docs = await search_agent.search(query)
    except Exception:
        logger.exception("SearchAgent web fallback failed")
        return INSUFFICIENT_EVIDENCE_MESSAGE, []

    if not web_docs:
        logger.info("Web fallback returned no results")
        return INSUFFICIENT_EVIDENCE_MESSAGE, []

    stage2_text = await _call_llm(llm, query, web_docs, history)

    if _is_no_answer(stage2_text):
        logger.info("Web sources also insufficient (LLM returned None) — giving up")
        return INSUFFICIENT_EVIDENCE_MESSAGE, []

    return stage2_text, web_docs