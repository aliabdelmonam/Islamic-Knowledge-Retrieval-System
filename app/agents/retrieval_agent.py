import asyncio
import time
from typing import Protocol

from pydantic import BaseModel, Field

from app.agents.helper.general_knowledge_retrieval import create_retriever
from app.agents.triage_agent import IslamicCategory

from app.core import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RetrievedDocument(BaseModel):
    id: str
    text: str
    category: IslamicCategory
    score: float
    source_ref: str = ""
    metadata: dict = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    query: str
    results_by_category: dict[IslamicCategory, list[RetrievedDocument]] = Field(
        default_factory=dict
    )

    def flattened(self) -> list[RetrievedDocument]:
        all_docs = sorted(
            [
                doc
                for docs in self.results_by_category.values()
                for doc in docs
            ],
            key=lambda x: x.score,
            reverse=True,
        )
        logger.debug(
            f"Flattened {len(all_docs)} document(s) across "
            f"{len(self.results_by_category)} category/categories "
            f"(best_score={all_docs[0].score if all_docs else 0.0})"
        )
        return all_docs


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------

class GeneralQuestionRetriever:

    def __init__(self):
        self.retriever = create_retriever()
        logger.info(
            f"GeneralQuestionRetriever initialized "
            f"(underlying_retriever={type(self.retriever).__name__})"
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        logger.debug(
            f"GeneralQuestionRetriever.retrieve started "
            f"(query_length={len(query) if query else 0}, top_k={top_k}, "
            f"preview={(query or '')[:120]!r})"
        )
        start = time.perf_counter()

        chunks = await self.retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        logger.debug(
            f"GeneralQuestionRetriever raw chunks fetched "
            f"(count={len(chunks)}, elapsed={time.perf_counter() - start:.2f}s, "
            f"top_score={max((c.score for c in chunks), default=0.0):.4f})"
        )

        retrieved_docs = [
            RetrievedDocument(
                id=chunk.id,
                text=chunk.answer,
                category=IslamicCategory.GENERAL_QUESTION,
                score=chunk.score,
                source_ref=chunk.fatwa_url,
                metadata={
                    "title": chunk.title,
                    "question": chunk.question,
                    **chunk.metadata,
                },
            )
            for chunk in chunks
        ]

        logger.debug(
            f"Mapped {len(retrieved_docs)} chunk(s) to RetrievedDocument "
            f"(category={IslamicCategory.GENERAL_QUESTION.value})"
        )
        return retrieved_docs


class QuranRetriever:

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        logger.warning(
            f"QuranRetriever.retrieve called but not implemented yet "
            f"(query_length={len(query) if query else 0}, top_k={top_k})"
        )
        raise NotImplementedError("QuranRetriever is not implemented yet.")


class HadithRetriever:

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        logger.warning(
            f"HadithRetriever.retrieve called but not implemented yet "
            f"(query_length={len(query) if query else 0}, top_k={top_k})"
        )
        raise NotImplementedError("HadithRetriever is not implemented yet.")


# ---------------------------------------------------------------------------
# Retrieval Agent
# ---------------------------------------------------------------------------

class TriageLike(Protocol):
    has_actionable_request: bool
    needs_clarification: bool
    categories: list[IslamicCategory]


class RetrievalAgent:

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

        self.retrievers = {
            IslamicCategory.GENERAL_QUESTION: GeneralQuestionRetriever(),
            # IslamicCategory.QURAN: QuranRetriever(),
            # IslamicCategory.HADITH: HadithRetriever(),
        }
        logger.info(
            f"RetrievalAgent initialized (top_k={top_k}, "
            f"registered_retrievers={[c.value for c in self.retrievers]})"
        )

    async def run(
        self,
        query: str,
        triage: TriageLike,
    ) -> RetrievalResponse:
        logger.info(
            f"RetrievalAgent.run started (query_length={len(query) if query else 0}, "
            f"preview={(query or '')[:120]!r}, "
            f"actionable={triage.has_actionable_request}, "
            f"needs_clarification={triage.needs_clarification}, "
            f"triage_categories={[c.value for c in triage.categories]})"
        )

        if not triage.has_actionable_request:
            logger.info("Skipping retrieval: triage reports no actionable request")
            return RetrievalResponse(query=query)

        if triage.needs_clarification or not triage.categories:
            logger.info(
                f"Skipping retrieval: needs_clarification={triage.needs_clarification}, "
                f"triage_categories={[c.value for c in triage.categories]}"
            )
            return RetrievalResponse(query=query)

        # Filter down to categories we actually have a retriever for FIRST,
        # then keep this same filtered list in lockstep with `tasks` so the
        # zip below can't misalign category <-> result pairs.
        categories = [c for c in triage.categories if c in self.retrievers]

        if len(categories) != len(triage.categories):
            logger.warning(
                f"Dropping triage categories without a registered retriever: "
                f"requested={[c.value for c in triage.categories]}, "
                f"supported={[c.value for c in self.retrievers]}, "
                f"using={[c.value for c in categories]}"
            )

        if not categories:
            logger.info(
                "Skipping retrieval: none of the triage categories have a registered retriever "
                f"(requested={[c.value for c in triage.categories]}, "
                f"supported={[c.value for c in self.retrievers]})"
            )
            return RetrievalResponse(query=query)

        tasks = [
            self.retrievers[category].retrieve(query, self.top_k)
            for category in categories
        ]

        logger.debug(
            f"Dispatching {len(tasks)} parallel retrieval task(s) for "
            f"categories={[c.value for c in categories]} (top_k={self.top_k})"
        )
        start = time.perf_counter()
        try:
            results = await asyncio.gather(*tasks)
        except Exception:
            logger.exception(
                f"Retrieval failed during concurrent gather "
                f"(categories={[c.value for c in categories]}, "
                f"tasks={len(tasks)}, elapsed={time.perf_counter() - start:.2f}s)"
            )
            raise
        elapsed = time.perf_counter() - start

        per_category = {
            c.value: {
                "docs": len(docs),
                "best_score": round(max((d.score for d in docs), default=0.0), 4),
            }
            for c, docs in zip(categories, results)
        }
        logger.info(
            f"Retrieval completed (elapsed={elapsed:.2f}s, "
            f"total_docs={sum(len(docs) for docs in results)}, "
            f"per_category={per_category})"
        )

        return RetrievalResponse(
            query=query,
            results_by_category=dict(zip(categories, results)),
        )


async def example():
    from app.agents.triage_agent import TriageAgent, ChitchatType
    from app.providers import ProviderFactory, Provider

    retrieval_agent = RetrievalAgent(top_k=3)
    query = "ما هي اركان الوضوء"

    triage_agent = TriageAgent(llm=ProviderFactory.create(Provider.GEMINI, model="gemini-3.1-flash-lite"))
    triage = await triage_agent.classify(query)

    response = await retrieval_agent.run(query, triage)
    docs = response.flattened()

    if not docs:
        print("No results found.")
        return

    for doc in docs:
        print(f"ID: {doc.id}")
        print(f"Category: {doc.category.value}")
        print(f"Score: {doc.score}")
        print(f"Source: {doc.source_ref}")
        print(f"Text: {doc.text}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 40)


if __name__ == "__main__":
    asyncio.run(example())