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

generate_multi_answer handles the multi-question case: one pooled LLM call
across all questions and their independently retrieved documents, producing
a single coherent answer. It does not get the two-stage web-search fallback
— see its docstring for why.
"""
from __future__ import annotations

from typing import Any
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


def _extract_text(content: Any) -> str:
    """response.text can be a plain string or (via LangChain) a list of
    content blocks. Normalize to a plain string. Used everywhere in this
    file that reads response.text, so there's exactly one implementation."""
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


def _build_context(docs: list[RetrievedDocument]) -> str:
    blocks = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "")
        question = doc.metadata.get("question", "")
        answer = doc.text
        blocks.append(
            f"العنوان: {title}\n"
            f"السؤال: {question}\n"
            f"الإجابة: {answer}\n"
        )
    return "\n\n".join(blocks)


def _is_no_answer(text: str) -> bool:
    return text.strip().rstrip(".") == NO_ANSWER_MARKER


def _build_multi_context(questions: list[str], docs_per_question: list[list[RetrievedDocument]]) -> str:
    blocks = []
    for i, (question, docs) in enumerate(zip(questions, docs_per_question), 1):
        if not docs:
            blocks.append(f"### السؤال {i}: {question}\n(لا توجد مصادر متاحة لهذا السؤال)")
            continue

        doc_blocks = []
        for j, doc in enumerate(docs, 1):
            ref = doc.source_ref or doc.metadata.get("title", "")

            meta_line = ""
            if doc.metadata:
                meta_parts = [
                    f"{key}: {value}"
                    for key, value in doc.metadata.items()
                    if key != "title" and value not in (None, "", [])
                ]
                if meta_parts:
                    meta_line = f"\n({', '.join(meta_parts)})"

            doc_blocks.append(
                f"[{i}.{j}] (الفئة: {doc.category.value}, المصدر: {ref}){meta_line}\n{doc.text}"
            )

        blocks.append(f"### السؤال {i}: {question}\n\n" + "\n\n".join(doc_blocks))

    return "\n\n---\n\n".join(blocks)


async def generate_multi_answer(
    *,
    llm,
    questions: list[str],
    docs_per_question: list[list[RetrievedDocument]],
    history: list[Message] | None = None,
) -> tuple[str, list[RetrievedDocument]]:
    """
    Single-call generation over multiple questions and their independently
    retrieved documents. Produces one coherent answer addressing every
    question. Unlike generate_answer, there is no per-question web-search
    fallback here — if the model outputs the None sentinel, the ENTIRE
    response is treated as insufficient, discarding any questions that were
    actually answered well. This is a real limitation of pooling into one
    call; acceptable for now, but worth revisiting if it shows up often in
    practice with mixed-sufficiency batches.
    """
    if len(questions) == 1:
        # No batching benefit for a single question — reuse the existing,
        # already-tested single-question path with its full fallback logic.
        docs = docs_per_question[0] if docs_per_question else []
        return await generate_answer(
            llm=llm, search_agent=None, query=questions[0], docs=docs, history=history,
        )

    context = _build_multi_context(questions, docs_per_question)
    prompt = (
        f"فيما يلي عدة أسئلة، كل سؤال مصحوب بمصادره الخاصة به.\n"
        f"أجب عن كل سؤال بالاعتماد فقط على مصادره الخاصة، ثم اجمع الإجابات "
        f"في رد واحد متكامل ومترابط، دون الخلط بين مصادر الأسئلة المختلفة.\n\n"
        f"{context}\n\nالإجابة:"
    )

    response = await llm.generate(
        messages=[
            Message(role="system", content=RETRIEVAL_SYSTEM_PROMPT),
            *(history or []),
            Message(role="user", content=prompt),
        ],
        temperature=0.2,
        max_tokens=800 * len(questions),
    )

    text = _extract_text(response.text)

    if _is_no_answer(text):
        return INSUFFICIENT_EVIDENCE_MESSAGE, []

    used_sources = [doc for docs in docs_per_question for doc in docs]
    return text, used_sources


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
        max_tokens=12000,
    )
    return _extract_text(response.text)


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