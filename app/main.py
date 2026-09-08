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
    from app.agents.answer_agent import AnswerAgent
    from app.services.session_store import SessionStore

    # 1. LLM client — shared by triage classification, query rewrite,
    #    sufficiency checks, and final answer generation
    llm = ProviderFactory.create(Provider.GEMINI, model=settings.llm_model)
    app.state.llm = llm
    logger.info("[1/5] LLM client ready (model=%s).", settings.llm_model)

    # 2. Triage agent — routes each question to general_question / hadith / quran
    app.state.triage_agent = TriageAgent(
        llm=llm,
        temperature=settings.triage_temperature,
    )
    logger.info("[2/5] Triage agent ready.")

    # 3. Retrieval agent — constructs all three retrievers:
    #      - GeneralQuestionRetriever (Qdrant, fatwa embeddings)
    #      - HadithRetriever (Whoosh index over the hadith CSV)
    #      - QuranRetriever (Whoosh index over the enriched Quran JSON)
    #    Each one loads/builds its index synchronously in __init__, so this
    #    must happen once here at startup, never per-request. AnswerAgent
    #    below reuses these retriever instances directly as its "tools" —
    #    RetrievalAgent.run() itself (parallel, all-at-once) is only used by
    #    the plain /retrieve endpoint.
    app.state.retrieval_agent = RetrievalAgent(top_k=settings.retrieval_top_k)
    logger.info(
        "[3/5] Retrieval agent ready (categories=%s).",
        [c.value for c in app.state.retrieval_agent.retrievers],
    )

    # 4. Answer agent — the agentic layer used by /ask and /chat: upfront
    #    query rewrite, triage, then a bounded tool-calling loop with an LLM
    #    sufficiency check after each call, followed by grounded generation.
    app.state.answer_agent = AnswerAgent(
        llm=llm,
        triage_agent=app.state.triage_agent,
        retrieval_agent=app.state.retrieval_agent,
        max_tool_calls=settings.agent_max_tool_calls,
        top_k=settings.retrieval_top_k,
    )
    logger.info("[4/5] Answer agent ready (max_tool_calls=%d).", settings.agent_max_tool_calls)

    # 5. Session store — backs the multi-turn /chat endpoint with rolling
    #    per-session conversation history.
    app.state.session_store = SessionStore()
    logger.info("[5/5] Session store ready.")

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