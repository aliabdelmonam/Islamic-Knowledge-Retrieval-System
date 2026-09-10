"""
FastAPI application entry point.
Loads the LLM client, triage agent, and retrieval agent (general question,
hadith, and quran retrievers) once during startup lifespan, and stores them
in app.state for zero-cost injection into the /ask and /retrieve endpoints.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging

setup_logging("DEBUG" if settings.debug else "INFO")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the LLM client and both agents once at startup; clean up at shutdown."""
    logger.info("=== Starting Islamic Knowledge RAG API ===")

    from app.providers import Provider, ProviderFactory
    from app.agents.triage_agent import IslamicCategory, TriageAgent
    from app.agents.retrieval_agent import RetrievalAgent
    from app.services.session_store import SessionStore

    # 1. LLM client — shared by  final answer generation
    response_llm = ProviderFactory.create(Provider.GEMINI, model=settings.response_llm)
    app.state.response_llm = response_llm
    logger.info("[1/4] LLM client ready (model=%s).", settings.response_llm)

    # 1.1. LLM client — shared by triage classification and final answer generation
    task_llm = ProviderFactory.create(Provider.GEMINI, model=settings.task_llm)
    app.state.task_llm = task_llm
    logger.info("[1/4] LLM client ready (model=%s).", settings.task_llm)
    

    # 2. Triage agent — routes each question to general_question / hadith / quran
    app.state.triage_agent = TriageAgent(
        llm=app.state.task_llm,
        temperature=settings.triage_temperature,
    )
    logger.info("[2/4] Triage agent ready.")

    # 3. Retrieval agent — constructs all three retrievers:
    #      - GeneralQuestionRetriever (Qdrant, fatwa embeddings)
    #      - HadithRetriever (Whoosh index over the hadith CSV)
    #      - QuranRetriever (Whoosh index over the enriched Quran JSON)
    #    Each one loads/builds its index synchronously in __init__, so this
    #    must happen once here at startup, never per-request.
    app.state.retrieval_agent = RetrievalAgent(top_k=settings.retrieval_top_k)
    logger.info(
        "[3/4] Retrieval agent ready (categories=%s).",
        [c.value for c in app.state.retrieval_agent.retrievers],
    )

    # 4. Session store — backs the multi-turn /chat endpoint with rolling
    #    per-session conversation history.
    app.state.session_store = SessionStore(max_messages=100,ttl_seconds=60 * 60 * 2)
    logger.info("[4/4] Session store ready.")

    logger.info("=== Islamic Knowledge RAG API ready to serve requests ===")
    yield

    # Shutdown
    logger.info("Shutting down Islamic Knowledge RAG API…")
    general_retriever = app.state.retrieval_agent.retrievers.get(IslamicCategory.GENERAL_QUESTION)
    if general_retriever is not None:
        try:
            await general_retriever.retriever.client.close()
        except Exception:
            logger.exception("Error closing Qdrant client")


# ── App factory ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description="Islamic knowledge RAG API — triage + general fatwa / hadith / quran retrieval",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from app.api.v1.router import router
    app.include_router(router)

    return app


app = create_app()