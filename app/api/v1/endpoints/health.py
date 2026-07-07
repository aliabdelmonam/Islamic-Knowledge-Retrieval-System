"""GET /api/v1/health — pipeline readiness check."""
from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas.response import HealthResponse
from app.core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    state = request.app.state
    components = {
        "embeddings": getattr(state, "embeddings", None) is not None,
        "vectorstore": getattr(state, "vectorstore", None) is not None,
        "bm25_index": getattr(state, "bm25_index", None) is not None,
        "reranker": getattr(state, "reranker", None) is not None,
        "llm": getattr(state, "llm", None) is not None,
        "rag_chain": getattr(state, "rag_chain", None) is not None,
    }
    all_ready = all(components.values())
    return HealthResponse(
        status="ok" if all_ready else "initializing",
        components=components,
        version=settings.api_version,
    )
