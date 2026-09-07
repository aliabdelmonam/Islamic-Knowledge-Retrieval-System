from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.core import settings, get_logger
from app.services.build_hadith_retrieval import search_hadith

logger = get_logger(__name__)

SEARCH_COL = "clean_hadith"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RetrievedHadith(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    hadith: str = ""
    clean_hadith: str = ""
    score: float = Field(default=0.0, description="Similarity score")
    categories: str = ""
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# RAG wrapper
# ---------------------------------------------------------------------------

class HadithRAG:
    """
    Thin wrapper around the Whoosh-based hadith index (build_hadith_retrieval.search_hadith)
    plus a category filter over the source CSV, mirroring the shape of GeneralQuestionRAG.
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        index_dir: Optional[str] = None,
    ) -> None:
        self.csv_path = csv_path or settings.hadith_csv_path
        self.index_dir = index_dir or settings.hadith_index_dir
        self.search_col = SEARCH_COL

        logger.info(f"Loading hadith dataset from {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        logger.info(
            f"HadithRAG initialized (rows={len(self.df)}, index_dir={self.index_dir})"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedHadith]:
        """
        Full-text search over the hadith index. Synchronous/blocking — callers on an
        asyncio event loop should run this via asyncio.to_thread.
        """
        if not query or not query.strip():
            return []

        raw_results = search_hadith(self.index_dir, query, self.search_col, limit=top_k)
        return [self._to_result(r) for r in raw_results]

    def retrieve_by_category(
        self,
        category: str,
        limit: int = 5,
    ) -> list[RetrievedHadith]:
        """Substring match over the 'categories' column (kept from the original script)."""
        mask = self.df["categories"].fillna("").str.contains(category, case=False, na=False)
        subset = self.df[mask].head(limit)
        return [self._to_result(r) for r in subset.to_dict("records")]

    @staticmethod
    def _to_result(record: dict) -> RetrievedHadith:
        used_keys = {"id", "hadith_id", "hadith", "clean_hadith", "_score", "categories"}
        raw_id = record.get("id") or record.get("hadith_id") or record.get("hadith", "")[:64]
        return RetrievedHadith(
            id=str(raw_id),
            hadith=record.get("hadith", ""),
            clean_hadith=record.get("clean_hadith", ""),
            score=float(record.get("_score", 0.0) or 0.0),
            categories=record.get("categories", ""),
            metadata={k: v for k, v in record.items() if k not in used_keys},
        )


def create_hadith_retriever() -> HadithRAG:
    """Create the retriever and load the hadith dataset once."""
    return HadithRAG()


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

def example():
    retriever = create_hadith_retriever()

    query = "ان الله يحب الجمال"
    print(f"--- Searching for: {query} ---")
    results = retriever.retrieve(query, top_k=3)
    if not results:
        print("No results found.")
    else:
        for i, res in enumerate(results, 1):
            print(f"Result #{i}:")
            print(f"Hadith: {res.hadith}")
            print(f"Clean Hadith (Short): {res.clean_hadith[:200]}...")
            print(f"Score: {res.score:.2f}")
            print("-" * 30)

    print("=" * 60)

    category = "الصدق"
    results = retriever.retrieve_by_category(category)
    if not results:
        print("No results found.")
    else:
        for i, res in enumerate(results, 1):
            print(f"Result #{i}:")
            print(f"Hadith: {res.hadith}")
            print(f"Categories: {res.categories}")
            print("-" * 30)


if __name__ == "__main__":
    example()