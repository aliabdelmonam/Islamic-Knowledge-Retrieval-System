import asyncio
import time
from typing import Protocol

from pydantic import BaseModel, Field

from app.agents.helper.general_knowledge_retrieval import create_retriever
from app.agents.helper.hadith_retrieval import create_hadith_retriever
from app.agents.helper.quran_retreival import create_quran_retriever
from app.agents.triage_agent import IslamicCategory, TriageResult

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

    def __init__(self):
        self.retriever = create_quran_retriever()
        logger.info(
            f"QuranRetriever initialized "
            f"(underlying_retriever={type(self.retriever).__name__})"
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        logger.debug(
            f"QuranRetriever.retrieve started "
            f"(query_length={len(query) if query else 0}, top_k={top_k}, "
            f"preview={(query or '')[:120]!r})"
        )
        start = time.perf_counter()

        # The underlying Whoosh search is synchronous/blocking; run it off the
        # event loop so it doesn't stall sibling retrievers running concurrently.
        ayat = await asyncio.to_thread(self.retriever.retrieve, query, top_k)

        logger.debug(
            f"QuranRetriever raw results fetched "
            f"(count={len(ayat)}, elapsed={time.perf_counter() - start:.2f}s, "
            f"top_score={max((a.score for a in ayat), default=0.0):.4f})"
        )

        retrieved_docs = [
            RetrievedDocument(
                id=ayah.id,
                text=ayah.text,
                category=IslamicCategory.QURAN,
                score=ayah.score,
                source_ref=f"{ayah.surah_ar} ({ayah.surah_en})".strip(),
                metadata={
                    "tafsir": ayah.tafsir,
                    "surah_ar": ayah.surah_ar,
                    "surah_en": ayah.surah_en,
                    **ayah.metadata,
                },
            )
            for ayah in ayat
        ]

        logger.debug(
            f"Mapped {len(retrieved_docs)} ayah(s) to RetrievedDocument "
            f"(category={IslamicCategory.QURAN.value})"
        )
        return retrieved_docs


class HadithRetriever:

    def __init__(self):
        self.retriever = create_hadith_retriever()
        logger.info(
            f"HadithRetriever initialized "
            f"(underlying_retriever={type(self.retriever).__name__})"
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        logger.debug(
            f"HadithRetriever.retrieve started "
            f"(query_length={len(query) if query else 0}, top_k={top_k}, "
            f"preview={(query or '')[:120]!r})"
        )
        start = time.perf_counter()

        # search_hadith is synchronous/blocking (Whoosh); offload to a thread.
        hadiths = await asyncio.to_thread(self.retriever.retrieve, query, top_k)

        logger.debug(
            f"HadithRetriever raw results fetched "
            f"(count={len(hadiths)}, elapsed={time.perf_counter() - start:.2f}s, "
            f"top_score={max((h.score for h in hadiths), default=0.0):.4f})"
        )

        retrieved_docs = [
            RetrievedDocument(
                id=hadith.id,
                text=hadith.hadith,
                category=IslamicCategory.HADITH,
                score=hadith.score,
                source_ref=hadith.categories,
                metadata={
                    "clean_hadith": hadith.clean_hadith,
                    **hadith.metadata,
                },
            )
            for hadith in hadiths
        ]

        logger.debug(
            f"Mapped {len(retrieved_docs)} hadith(s) to RetrievedDocument "
            f"(category={IslamicCategory.HADITH.value})"
        )
        return retrieved_docs


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
            IslamicCategory.QURAN: QuranRetriever(),
            IslamicCategory.HADITH: HadithRetriever(),
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
    
    async def retrieve_all(
        self,
        questions: list[str],
        triage_results: list[TriageResult],
    ) -> list[list[RetrievedDocument]]:
        """
        Run retrieval for each question in parallel. Returns one docs list
        per question, in the same order as *questions*. Questions that are
        chitchat, non-Islamic, or need clarification get an empty docs list
        without calling retrieval at all.
        """
        async def _retrieve_one(q: str, triage_q: TriageResult) -> list[RetrievedDocument]:
            if triage_q.is_non_islamic or not triage_q.has_actionable_request or triage_q.needs_clarification:
                logger.debug("Skipping retrieval for sub-question (non-actionable): %r", q)
                return []
            try:
                retrieval = await self.run(q, triage_q)
                return retrieval.flattened()[: self.top_k]
            except Exception:
                logger.exception("Retrieval failed for sub-question: %r", q)
                return []

        logger.info(f"RetrievalAgent.retrieve_all started (num_questions={len(questions)})")
        start = time.perf_counter()

        results = await asyncio.gather(*[
            _retrieve_one(q, t) for q, t in zip(questions, triage_results)
        ])

        logger.info(
            f"RetrievalAgent.retrieve_all completed (elapsed={time.perf_counter() - start:.2f}s, "
            f"total_docs={sum(len(docs) for docs in results)})"
        )
        return results


async def example():
    from app.agents.triage_agent import TriageAgent, ChitchatType
    from app.providers import ProviderFactory, Provider

    retrieval_agent = RetrievalAgent(top_k=3)
    query = "ما هي اركان الوضوء"

    triage_agent = TriageAgent(llm=ProviderFactory.create(Provider.GEMINI, model="gemini-3.1-flash-lite"))
    # triage = await triage_agent.classify(query)
    questions = [
            "ما حكم الربا؟",
            "هل يجوز أكل لحم الأرنب؟",
            "شكرًا جزيلاً",
            "من فاز بكأس العالم2026 ؟"
        ]
    batch_results = await triage_agent.classify_batch(questions)
    docs_per_question = await retrieval_agent.retrieve_all(questions, batch_results)
    
    total_docs = 0
    for question, docs in zip(questions, docs_per_question):
        print(f"\nQ: {question}  ->  {len(docs)} document(s)")

        if not docs:
            print("  (no retrieval — chitchat / non-actionable / needs clarification)")
            continue

        for doc in docs:
            total_docs += 1
            print(f"  ID: {doc.id}")
            print(f"  Category: {doc.category.value}")
            print(f"  Score: {doc.score:.4f}")
            print(f"  Source: {doc.source_ref}")
            print(f"  Text: {doc.text[:200]}{'…' if len(doc.text) > 200 else ''}")
            print(f"  Metadata keys: {list(doc.metadata.keys())}")
            print("  " + "-" * 38)

    if total_docs == 0:
        print("\nNo results found for any question.")


if __name__ == "__main__":
    asyncio.run(example())