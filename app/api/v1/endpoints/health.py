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
        "llm": getattr(state, "llm", None) is not None,
        "triage_agent": getattr(state, "triage_agent", None) is not None,
        "retrieval_agent": getattr(state, "retrieval_agent", None) is not None,
        "answer_agent": getattr(state, "answer_agent", None) is not None,
        "session_store": getattr(state, "session_store", None) is not None,
    }
    all_ready = all(components.values())
    return HealthResponse(
        status="ok" if all_ready else "initializing",
        components=components,
        version=settings.api_version,
    )