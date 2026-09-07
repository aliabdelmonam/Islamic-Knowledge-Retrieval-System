from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.core import settings, get_logger
from app.services import build_index, search

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RetrievedAyah(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    surah_ar: str = ""
    surah_en: str = ""
    text: str = ""
    tafsir: str = ""
    score: float = Field(default=0.0, description="Similarity score")
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# RAG wrapper
# ---------------------------------------------------------------------------

class QuranRAG:
    """
    Thin wrapper around the Whoosh-based Quran index (app.services build_index/search),
    mirroring the shape of GeneralQuestionRAG. Builds the index lazily on first
    construction rather than at import time.
    """

    def __init__(
        self,
        json_path: Optional[str] = None,
        index_dir: Optional[str] = None,
        max_edit_distance: int = 1,
    ) -> None:
        self.json_path = json_path or settings.quran_json_path
        self.index_dir = index_dir or settings.quran_index_dir
        self.max_edit_distance = max_edit_distance

        if not Path(self.index_dir).exists():
            logger.info(f"Quran index not found at {self.index_dir}; building it now")
            build_index(self.json_path, self.index_dir)
        else:
            logger.info(f"Reusing existing Quran index at {self.index_dir}")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedAyah]:
        """
        Full-text/fuzzy search over the Quran index. Synchronous/blocking — callers on
        an asyncio event loop should run this via asyncio.to_thread.
        """
        if not query or not query.strip():
            return []

        raw_results = search(
            self.index_dir,
            query,
            limit=top_k,
            max_edit_distance=self.max_edit_distance,
        )
        return [self._to_result(r) for r in raw_results]

    @staticmethod
    def _to_result(record: dict) -> RetrievedAyah:
        used_keys = {"id", "surah_ar", "surah_en", "text", "tafsir", "_score"}
        return RetrievedAyah(
            id=str(record.get("id", "")),
            surah_ar=record.get("surah_ar", ""),
            surah_en=record.get("surah_en", ""),
            text=record.get("text", ""),
            tafsir=record.get("tafsir") or "",
            score=float(record.get("_score", 0.0) or 0.0),
            metadata={k: v for k, v in record.items() if k not in used_keys},
        )


def create_quran_retriever() -> QuranRAG:
    """Create the retriever, building the index once if it doesn't already exist."""
    return QuranRAG()


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

def example():
    retriever = create_quran_retriever()

    query = "هذا نذير"
    print(f"--- Searching for: {query} ---")
    results = retriever.retrieve(query, top_k=3)

    if not results:
        print("No results found.")
        return

    for i, res in enumerate(results, 1):
        print(f"Result #{i}:")
        print(f"ID: {res.id}")
        print(f"Surah: {res.surah_ar} ({res.surah_en})")
        print(f"Verse: {res.text}")
        print(f"Tafsir (Short): {res.tafsir[:200]}...")
        print(f"Score: {res.score:.2f}")
        print("-" * 30)


if __name__ == "__main__":
    example()