"""
SearchAgent — Tavily-backed general web search fallback for when internal
retrieval (GeneralQuestionRetriever / HadithRetriever / QuranRetriever) comes
back with nothing RelevanceCheckAgent accepts.

Restricted to a fixed whitelist of Islamic knowledge domains (see
islamic_search_domains.DOMAINS) via Tavily's include_domains — no open web
search, no category routing. Same query in, top-k results out, all sourced
from the whitelist. Output is shaped exactly like RetrievedDocument, so it's
a drop-in extra source for the same RelevanceCheckAgent used for internal
retrieval — no separate filtering logic needed downstream.
"""
from __future__ import annotations

import time
from urllib.parse import urlparse

from tavily import AsyncTavilyClient

from app.agents.retrieval_agent import RetrievedDocument
from app.agents.triage_agent import IslamicCategory
from app.agents.helper.islamic_search_domains import DOMAINS

from app.core.config import settings

from app.core import get_logger

logger = get_logger(__name__)


class SearchAgent:

    def __init__(
        self,
        *,
        top_k: int = 5,
        search_depth: str = "advanced",
        allowed_domains: list[str] | None = None,
    ) -> None:
        self.client = AsyncTavilyClient(api_key=settings.TAVILY_API_KEY)
        self.top_k = top_k
        self.search_depth = search_depth
        # Whitelist of Islamic knowledge domains — restricts Tavily results
        # via include_domains, and is enforced again defensively on the
        # returned results in case the API ever returns something outside it.
        self.allowed_domains = list(allowed_domains or DOMAINS)
        self._allowed_domains_set = {d.removeprefix("www.") for d in self.allowed_domains}
        logger.info(
            f"SearchAgent initialized (top_k={top_k}, search_depth={search_depth}, "
            f"allowed_domains={len(self.allowed_domains)})"
        )

    def _is_allowed(self, domain: str) -> bool:
        domain = domain.removeprefix("www.")
        return any(
            domain == allowed or domain.endswith(f".{allowed}")
            for allowed in self._allowed_domains_set
        )

    @staticmethod
    def _to_document(result: dict) -> RetrievedDocument:
        url = result.get("url", "")
        domain = urlparse(url).netloc.removeprefix("www.")
        text = result.get("raw_content") or result.get("content") or ""

        return RetrievedDocument(
            id=url or result.get("title", ""),
            text=text,
            # Web results aren't triaged into general_question/hadith/quran —
            # RetrievedDocument requires a category, so tag every web result
            # as GENERAL_QUESTION as a catch-all bucket. It's metadata that
            # actually distinguishes it (source_type="web", domain).
            category=IslamicCategory.GENERAL_QUESTION,
            score=float(result.get("score", 0.0) or 0.0),
            source_ref=url,
            metadata={
                "title": result.get("title", ""),
                "domain": domain,
                "source_type": "web",
            },
        )

    async def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        if not query or not query.strip():
            return []

        effective_top_k = top_k or self.top_k

        logger.info(
            f"SearchAgent.search started (query_length={len(query)}, "
            f"top_k={effective_top_k}, preview={query[:120]!r})"
        )

        start = time.perf_counter()
        try:
            response = await self.client.search(
                query=query,
                search_depth=self.search_depth,
                max_results=effective_top_k,
                include_raw_content=True,
                include_domains=self.allowed_domains,
            )
        except Exception:
            logger.exception(
                f"Tavily search failed (elapsed={time.perf_counter() - start:.2f}s) "
                f"— returning no web results"
            )
            return []

        results = response.get("results", []) if isinstance(response, dict) else []
        elapsed = time.perf_counter() - start

        docs = [self._to_document(r) for r in results]
        # Drop hits with no usable body text (paywalled/JS-only pages, etc.)
        docs = [d for d in docs if d.text.strip()]

        # Defensive re-filter: Tavily's include_domains should already
        # restrict results, but don't trust that silently — drop anything
        # outside the whitelist rather than let it leak downstream.
        pre_filter_count = len(docs)
        docs = [d for d in docs if self._is_allowed(d.metadata.get("domain", ""))]
        dropped = pre_filter_count - len(docs)
        if dropped:
            logger.warning(
                f"SearchAgent.search dropped {dropped} result(s) outside the "
                f"domain whitelist despite include_domains"
            )

        logger.info(
            f"SearchAgent.search completed (elapsed={elapsed:.2f}s, "
            f"raw_result_count={len(results)}, usable_doc_count={len(docs)})"
        )

        return docs


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

async def example():
    import os

    agent = SearchAgent()
    query = "ما حكم الجمع بين الصلاتين في السفر"

    docs = await agent.search(query)

    if not docs:
        print("No web results found.")
        return

    for doc in docs:
        print(f"URL: {doc.source_ref}")
        print(f"Score: {doc.score}")
        print(f"Domain: {doc.metadata.get('domain')}")
        print(f"Text (short): {doc.text[:200]}...")
        print("-" * 40)


if __name__ == "__main__":
    import asyncio
    asyncio.run(example())