"""
RelevanceCheckAgent — sits between RetrievalAgent and generation in the plain
RAG pipeline.

Takes the full pool of retrieved chunks (RetrievalResponse.flattened(), i.e.
everything RetrievalAgent.run() pulled back across whichever categories
triage selected) plus the original query, and makes ONE LLM call that judges
relevance per chunk. Chunks the model doesn't flag as relevant are dropped
before generation ever sees them.

If the filtered result is empty, that's the caller's signal to fall back
(e.g. to a future web-search step) — this class only does the filtering, it
doesn't decide what happens on empty.
"""
import time

from pydantic import BaseModel, ConfigDict, Field

from app.agents.helper.check_relevance_system_prompt import RELEVANCE_CHECK_SYSTEM_PROMPT
from app.agents.retrieval_agent import RetrievedDocument
from app.providers import GenerationClient, Message

from app.core import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------

class RelevantChunks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevant_chunk_indices: list[int] = Field(default_factory=list)
    reasoning: str = Field(default="", max_length=500)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RelevanceCheckAgent:

    def __init__(
        self,
        llm: GenerationClient,
        *,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(
            f"RelevanceCheckAgent initialized (llm={type(llm).__name__}, "
            f"temperature={temperature}, max_tokens={max_tokens})"
        )

    @staticmethod
    def _build_context(docs: list[RetrievedDocument]) -> str:
        blocks = []
        for i, doc in enumerate(docs):
            ref = doc.source_ref or doc.metadata.get("title", "")
            blocks.append(
                f"[{i}] (الفئة: {doc.category.value}, المصدر: {ref})\n{doc.text}"
            )
        return "\n\n".join(blocks)

    async def filter(
        self,
        query: str,
        docs: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """
        Takes the flattened chunk pool from RetrievalAgent.run(...).flattened()
        and returns only the chunks judged actually relevant to `query`.
        Returns an empty list if nothing qualifies (or the check itself fails —
        fail closed, not open, since passing unfiltered chunks through would
        defeat the point of this step).
        """
        logger.debug(
            f"RelevanceCheckAgent.filter started (query_length={len(query) if query else 0}, "
            f"doc_count={len(docs)}, preview={(query or '')[:120]!r})"
        )

        if not docs:
            logger.info("No chunks to filter — retrieval returned nothing")
            return []

        prompt = f"السؤال: {query}\n\nالمقاطع المسترجعة:\n{self._build_context(docs)}"

        start = time.perf_counter()
        try:
            response = await self.llm.generate(
                messages=[
                    Message(role="system", content=RELEVANCE_CHECK_SYSTEM_PROMPT),
                    Message(role="user", content=prompt),
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                output_schema=RelevantChunks,
            )
        except Exception:
            logger.exception(
                f"Relevance check LLM call failed (doc_count={len(docs)}, "
                f"elapsed={time.perf_counter() - start:.2f}s) — failing closed, "
                f"returning no chunks rather than passing them through unfiltered"
            )
            return []

        try:
            result = RelevantChunks.model_validate_json(response.text)
        except Exception:
            logger.exception(
                f"Relevance check response failed schema validation. "
                f"Raw response: {response.text!r}"
            )
            return []

        relevant_docs = [
            docs[i] for i in result.relevant_chunk_indices if 0 <= i < len(docs)
        ]

        logger.info(
            f"Relevance check completed (elapsed={time.perf_counter() - start:.2f}s, "
            f"input_count={len(docs)}, relevant_count={len(relevant_docs)}, "
            f"relevant_indices={result.relevant_chunk_indices})"
        )
        logger.debug(f"Relevance check reasoning: {result.reasoning}")

        return relevant_docs


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

async def example():
    from app.agents.triage_agent import TriageAgent
    from app.agents.retrieval_agent import RetrievalAgent
    from app.providers import Provider, ProviderFactory

    llm = ProviderFactory.create(Provider.GEMINI, model="gemini-3.1-flash-lite")

    triage_agent = TriageAgent(llm=llm)
    retrieval_agent = RetrievalAgent(top_k=5)
    relevance_agent = RelevanceCheckAgent(llm=llm)

    query = "ما هي اركان الوضوء"

    triage = await triage_agent.classify(query)
    retrieval = await retrieval_agent.run(query, triage)
    pool = retrieval.flattened()

    print(f"Retrieved {len(pool)} chunk(s) before filtering.")

    relevant_docs = await relevance_agent.filter(query, pool)

    if not relevant_docs:
        print("No relevant chunks found — handle fallback here.")
        return

    print(f"{len(relevant_docs)} chunk(s) judged relevant:")
    for doc in relevant_docs:
        print(f"ID: {doc.id}")
        print(f"Category: {doc.category.value}")
        print(f"Score: {doc.score}")
        print(f"Source: {doc.source_ref}")
        print(f"Text: {doc.text}")
        print("-" * 40)


if __name__ == "__main__":
    import asyncio
    asyncio.run(example())