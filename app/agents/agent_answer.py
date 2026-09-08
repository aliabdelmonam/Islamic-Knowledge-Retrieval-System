"""
AnswerAgent — the agentic layer sitting on top of TriageAgent and RetrievalAgent.

Flow per call:
  1. Query rewrite (upfront, single pass) — resolves coreference/context against
     conversation history. Skipped entirely when there's no history, since a
     standalone message needs no resolving. This is NOT a retry mechanism —
     it runs exactly once, before anything else.
  2. Triage — classify the resolved query (chitchat / needs-clarification /
     categories), same TriageAgent used elsewhere.
  3. Tool-calling loop — one retriever ("tool") call per iteration, ordered by
     triage's categories first, then any remaining categories. After EVERY
     tool call, an LLM sufficiency check re-judges the *entire accumulated*
     pool of chunks (not just the newest batch) for actual relevance, not
     just presence. If sufficient, break out. If not, the ONLY recovery lever
     is switching to the next tool — the resolved query itself never changes
     mid-loop. Hard-capped by max_tool_calls regardless of how many
     categories exist.
  4. Generation — runs once, over only the chunks the sufficiency check
     marked relevant. The model returns the answer plus the indices of the
     chunks it actually used (inline, no second call). The final "sources"
     shown to the caller are built from THIS output, not from the broader
     sufficiency-filtered set — a chunk can be relevant but never cited, and
     it won't appear in sources.
  5. Fallback — a fixed, non-LLM-generated message is returned whenever the
     loop exhausts its tool budget without sufficient evidence, or when
     generation itself comes back empty as a last line of defense. This is
     an interim placeholder until an online-search fallback tool exists.
"""
from __future__ import annotations

import time
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.agents.helper.agent_messages import (
    CLARIFICATION_NEEDED_MESSAGE,
    INSUFFICIENT_EVIDENCE_MESSAGE,
)
from app.agents.helper.answer_agent_system_prompt import ANSWER_SYSTEM_PROMPT
from app.agents.helper.query_rewrite_system_prompt import QUERY_REWRITE_SYSTEM_PROMPT
from app.agents.helper.evidence_check_system_prompt import SUFFICIENCY_CHECK_SYSTEM_PROMPT
from app.agents.retrieval_agent import RetrievalAgent, RetrievedDocument
from app.agents.triage_agent import ChitchatType, IslamicCategory, TriageAgent, TriageResult
from app.providers import GenerationClient, Message

from app.core import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Structured outputs for each LLM step
# ---------------------------------------------------------------------------

class RewrittenQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolved_query: str = Field(min_length=1)


class SufficiencyCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_sufficient: bool
    relevant_chunk_indices: list[int] = Field(default_factory=list)
    reasoning: str = Field(min_length=1, max_length=500)


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(default="")
    used_chunk_indices: list[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

class AgentAnswer(BaseModel):
    answer: str
    sources: list[RetrievedDocument] = Field(default_factory=list)
    categories: list[IslamicCategory] = Field(default_factory=list)
    chitchat_type: ChitchatType = ChitchatType.NONE
    needs_clarification: bool = False
    resolved_query: str = ""
    tool_calls_made: list[IslamicCategory] = Field(default_factory=list)
    is_fallback: bool = False


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AnswerAgent:

    def __init__(
        self,
        llm: GenerationClient,
        triage_agent: TriageAgent,
        retrieval_agent: RetrievalAgent,
        *,
        max_tool_calls: int = 3,
        top_k: int = 5,
    ) -> None:
        self.llm = llm
        self.triage_agent = triage_agent
        self.retrieval_agent = retrieval_agent
        self.max_tool_calls = max_tool_calls
        self.top_k = top_k
        logger.info(
            f"AnswerAgent initialized (max_tool_calls={max_tool_calls}, top_k={top_k}, "
            f"tools={[c.value for c in retrieval_agent.retrievers]})"
        )

    async def answer(
        self,
        message: str,
        conversation_history: Optional[list[Message]] = None,
        top_k: Optional[int] = None,
    ) -> AgentAnswer:
        conversation_history = conversation_history or []
        effective_top_k = top_k or self.top_k

        logger.info(
            f"AnswerAgent.answer started (message_length={len(message) if message else 0}, "
            f"history_messages={len(conversation_history)}, preview={(message or '')[:120]!r})"
        )

        resolved_query = await self._rewrite_query(message, conversation_history)

        triage = await self.triage_agent.classify(resolved_query, conversation_history)

        if not triage.has_actionable_request:
            return AgentAnswer(
                answer=triage.canned_response(),
                categories=[],
                chitchat_type=triage.chitchat_type,
                needs_clarification=False,
                resolved_query=resolved_query,
            )

        if triage.needs_clarification:
            return AgentAnswer(
                answer=CLARIFICATION_NEEDED_MESSAGE,
                categories=[],
                chitchat_type=triage.chitchat_type,
                needs_clarification=True,
                resolved_query=resolved_query,
            )

        return await self._run_tool_loop(resolved_query, triage, effective_top_k)

    # -----------------------------------------------------------------
    # Step 1: upfront query rewrite (single pass, not a retry mechanism)
    # -----------------------------------------------------------------

    async def _rewrite_query(self, message: str, history: list[Message]) -> str:
        message = (message or "").strip()

        if not history:
            # Nothing to resolve against — skip the LLM call entirely.
            return message

        try:
            response = await self.llm.generate(
                messages=[
                    Message(role="system", content=QUERY_REWRITE_SYSTEM_PROMPT),
                    *history,
                    Message(role="user", content=message),
                ],
                temperature=0.0,
                max_tokens=200,
                output_schema=RewrittenQuery,
            )
            rewritten = RewrittenQuery.model_validate_json(response.text)
            logger.debug(f"Query rewritten: {message!r} -> {rewritten.resolved_query!r}")
            return rewritten.resolved_query or message
        except Exception:
            logger.exception("Query rewrite failed; falling back to the raw message")
            return message

    # -----------------------------------------------------------------
    # Step 2: tool-calling loop with sufficiency check as the stop condition
    # -----------------------------------------------------------------

    def _tool_queue(self, triage: TriageResult) -> list[IslamicCategory]:
        ordered = list(triage.categories)
        for category in self.retrieval_agent.retrievers:
            if category not in ordered:
                ordered.append(category)
        return ordered[: self.max_tool_calls]

    async def _check_sufficiency(
        self,
        query: str,
        pool: list[RetrievedDocument],
    ) -> SufficiencyCheck:
        context = "\n\n".join(
            f"[{i}] (الفئة: {doc.category.value})\n{doc.text}"
            for i, doc in enumerate(pool)
        )
        prompt = f"السؤال: {query}\n\nالمقاطع المسترجعة:\n{context}"

        response = await self.llm.generate(
            messages=[
                Message(role="system", content=SUFFICIENCY_CHECK_SYSTEM_PROMPT),
                Message(role="user", content=prompt),
            ],
            temperature=0.0,
            max_tokens=32000,
            output_schema=SufficiencyCheck,
        )
        return SufficiencyCheck.model_validate_json(response.text)

    async def _run_tool_loop(
        self,
        query: str,
        triage: TriageResult,
        top_k: int,
    ) -> AgentAnswer:
        pool: list[RetrievedDocument] = []
        tools_called: list[IslamicCategory] = []
        queue = self._tool_queue(triage)

        logger.info(
            f"Tool loop starting (queue={[c.value for c in queue]}, "
            f"max_tool_calls={self.max_tool_calls})"
        )

        for step, category in enumerate(queue, 1):
            retriever = self.retrieval_agent.retrievers.get(category)
            if retriever is None:
                continue

            start = time.perf_counter()
            new_docs = await retriever.retrieve(query, top_k)
            tools_called.append(category)
            pool.extend(new_docs)

            logger.info(
                f"Tool call {step}/{len(queue)} (category={category.value}) fetched "
                f"{len(new_docs)} doc(s) in {time.perf_counter() - start:.2f}s "
                f"(pool_size={len(pool)})"
            )

            if not pool:
                continue  # nothing to judge yet, switch to the next tool

            try:
                check = await self._check_sufficiency(query, pool)
            except Exception:
                logger.exception("Sufficiency check failed; treating as insufficient")
                continue

            logger.info(
                f"Sufficiency check after tool {step}: is_sufficient={check.is_sufficient}, "
                f"relevant_indices={check.relevant_chunk_indices}"
            )

            if check.is_sufficient and check.relevant_chunk_indices:
                relevant_docs = [
                    pool[i] for i in check.relevant_chunk_indices if 0 <= i < len(pool)
                ]
                if relevant_docs:
                    return await self._finalize(query, triage, tools_called, relevant_docs)

            if step < len(queue):
                logger.info(f"Evidence insufficient — switching to next tool (not rewriting)")

        logger.info(
            f"Tool budget exhausted without sufficient evidence "
            f"(tools_called={[c.value for c in tools_called]}) — returning fallback"
        )
        return AgentAnswer(
            answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            sources=[],
            categories=triage.categories,
            chitchat_type=triage.chitchat_type,
            needs_clarification=False,
            resolved_query=query,
            tool_calls_made=tools_called,
            is_fallback=True,
        )

    # -----------------------------------------------------------------
    # Step 3: generation, grounded only in the sufficiency-filtered chunks
    # -----------------------------------------------------------------

    async def _generate(
        self,
        query: str,
        relevant_docs: list[RetrievedDocument],
    ) -> GeneratedAnswer:
        context = "\n\n".join(
            f"[{i}] (الفئة: {doc.category.value}, المصدر: {doc.source_ref})\n{doc.text}"
            for i, doc in enumerate(relevant_docs)
        )
        prompt = f"السؤال: {query}\n\nالمقاطع ذات الصلة:\n{context}"

        response = await self.llm.generate(
            messages=[
                Message(role="system", content=ANSWER_SYSTEM_PROMPT),
                Message(role="user", content=prompt),
            ],
            temperature=0.2,
            max_tokens=900,
            output_schema=GeneratedAnswer,
        )
        return GeneratedAnswer.model_validate_json(response.text)

    async def _finalize(
        self,
        query: str,
        triage: TriageResult,
        tools_called: list[IslamicCategory],
        relevant_docs: list[RetrievedDocument],
    ) -> AgentAnswer:
        try:
            generated = await self._generate(query, relevant_docs)
        except Exception:
            logger.exception("Generation failed after sufficient evidence was found")
            return AgentAnswer(
                answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                sources=[],
                categories=triage.categories,
                chitchat_type=triage.chitchat_type,
                needs_clarification=False,
                resolved_query=query,
                tool_calls_made=tools_called,
                is_fallback=True,
            )

        # Sources come from the generation step's own output, not from the
        # broader sufficiency-filtered `relevant_docs` set — only chunks the
        # model actually cited make it into the final answer's sources.
        used_sources = [
            relevant_docs[i]
            for i in generated.used_chunk_indices
            if 0 <= i < len(relevant_docs)
        ]

        if not generated.answer.strip() or not used_sources:
            # Defense in depth: even though the sufficiency check passed
            # upstream, generation itself found nothing it could ground an
            # answer in — don't let a hallucinated or citation-less answer
            # through.
            logger.warning(
                "Generation returned no usable answer/citations despite passing "
                "sufficiency check — returning fallback"
            )
            return AgentAnswer(
                answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                sources=[],
                categories=triage.categories,
                chitchat_type=triage.chitchat_type,
                needs_clarification=False,
                resolved_query=query,
                tool_calls_made=tools_called,
                is_fallback=True,
            )

        return AgentAnswer(
            answer=generated.answer,
            sources=used_sources,
            categories=triage.categories,
            chitchat_type=triage.chitchat_type,
            needs_clarification=False,
            resolved_query=query,
            tool_calls_made=tools_called,
            is_fallback=False,
        )