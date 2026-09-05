from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import AsyncQdrantClient, models

from app.core import settings
from app.providers import EmbeddingProviderFactory


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str = ""
    score: float = Field(description="Similarity score")

    fatwa_url: str = ""
    question: str = ""
    answer: str = ""

    metadata: dict = Field(default_factory=dict)


class GeneralQuestionRAG:

    def __init__(
        self,
        collection_name: str,
        client: Optional[AsyncQdrantClient] = None,
    ) -> None:

        self.collection_name = collection_name
        self.embed_model = EmbeddingProviderFactory.create(settings)

        self.client = client or AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> list[RetrievedChunk]:

        if not query or not query.strip():
            return []

        dense_vector =self.embed_model.embed([query])[0]

        result = await self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            using="dense",
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

        return [
            self._to_chunk(point)
            for point in result.points
        ]

    @staticmethod
    def _to_chunk(
        point: models.ScoredPoint,
    ) -> RetrievedChunk:

        payload = point.payload or {}

        return RetrievedChunk(
            id=str(point.id),
            title=payload.get("title", ""),
            score=point.score,
            fatwa_url=payload.get("url", ""),
            question=payload.get("question", ""),
            answer=payload.get("answer", ""),
            metadata={
                key: value
                for key, value in payload.items()
                if key not in {
                    "title",
                    "url",
                    "question",
                    "answer",
                }
            },
        )


def create_retriever() -> GeneralQuestionRAG:
    """
    Create the retriever and load the embedding model once.
    """
    return GeneralQuestionRAG(
        collection_name=settings.collection_name,
    )


async def example():
    retriever = create_retriever()
    query = "ما هي اركان الوضوء"
    results = await retriever.retrieve(query, top_k=3)

    if not results:
        print("No results found.")
        return

    for chunk in results:
        print(f"ID: {chunk.id}")
        print(f"Title: {chunk.title}")
        print(f"Score: {chunk.score}")
        print(f"URL: {chunk.fatwa_url}")
        print(f"Question: {chunk.question}")
        print(f"Answer: {chunk.answer}")
        print(f"Metadata: {chunk.metadata}")
        print("-" * 40)


if __name__ == "__main__":
    import asyncio
    asyncio.run(example())