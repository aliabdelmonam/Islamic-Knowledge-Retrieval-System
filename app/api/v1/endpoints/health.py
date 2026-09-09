"""GET /api/v1/health — pipeline readiness check."""
from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas.response import HealthResponse
from app.core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    state = request.app.state

    # Required — the /ask and /chat endpoints fail without these.
    required = {
        "response_llm": getattr(state, "response_llm", None) is not None,
        "task_llm": getattr(state, "task_llm", None) is not None,
        "triage_agent": getattr(state, "triage_agent", None) is not None,
        "retrieval_agent": getattr(state, "retrieval_agent", None) is not None,
        "session_store": getattr(state, "session_store", None) is not None,
    }
    # Optional — enables the whitelist web-search fallback in
    # answer_generation.generate_answer, but its absence just means that
    # fallback degrades straight to INSUFFICIENT_EVIDENCE_MESSAGE.
    optional = {
        "search_agent": getattr(state, "search_agent", None) is not None,
    }

    all_ready = all(required.values())
    return HealthResponse(
        status="ok" if all_ready else "initializing",
        components={**required, **optional},
        version=settings.api_version,
    )