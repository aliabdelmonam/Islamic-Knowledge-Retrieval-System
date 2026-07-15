"""
FastAPI application entry point.
Loads all heavy components (embeddings, Qdrant, BM25, reranker, LLM)
during startup lifespan, stores them in app.state for zero-cost injection.
"""
from __future__ import annotations

import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging

setup_logging("DEBUG" if settings.debug else "INFO")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML components once at startup; clean up at shutdown."""
    logger.info("=== Starting Hadith RAG API ===")

    # 1. Embeddings
    from app.services.embeddings import build_embeddings
    app.state.embeddings = build_embeddings(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
        openai_model=settings.openai_embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    logger.info("[1/6] Embeddings ready.")

    # 2. Qdrant vector store
    from app.services.vector_store import get_qdrant_client, get_vectorstore
    client = get_qdrant_client(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        prefer_grpc=settings.qdrant_prefer_grpc,
        timeout=settings.qdrant_timeout,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    app.state.qdrant_client = client
    app.state.vectorstore = get_vectorstore(
        client, app.state.embeddings, settings.collection_name
    )
    logger.info("[2/7] Qdrant vector store ready.")

    # 2b. Category collection (for multi-stage retrieval)
    try:
        cat_info = client.get_collection(settings.category_collection_name)
        app.state.category_collection_ready = (cat_info.points_count or 0) > 0
    except Exception:
        app.state.category_collection_ready = False

    if app.state.category_collection_ready:
        logger.info("[2b/7] Category collection '%s' available.", settings.category_collection_name)
    else:
        logger.warning(
            "[2b/7] Category collection '%s' not found — "
            "run `python scripts/init_category_index.py` to enable multi-stage retrieval.",
            settings.category_collection_name,
        )

    # 3. Load child chunks + parent store from disk
    parent_store_path = settings.models_dir / "parent_store.pkl"
    chunks_path = settings.models_dir / "chunks.pkl"

    if not parent_store_path.exists() or not chunks_path.exists():
        raise RuntimeError(
            f"Parent store or chunks not found. Run `python scripts/init_index.py` first.\n"
            f"Expected: {parent_store_path}, {chunks_path}"
        )

    with open(parent_store_path, "rb") as f:
        app.state.parent_store = pickle.load(f)
    with open(chunks_path, "rb") as f:
        app.state.all_chunks = pickle.load(f)
    logger.info(
        "[3/7] Loaded %d parents, %d child chunks.",
        len(app.state.parent_store),
        len(app.state.all_chunks),
    )

    # 4. BM25 index
    from app.services.bm25_index import load_bm25
    app.state.bm25_index = load_bm25(settings.models_dir / "bm25_index.pkl")
    logger.info("[4/7] BM25 index ready.")

    # 5. Reranker
    from app.services.reranker import get_reranker
    app.state.reranker = get_reranker(
        settings.reranker_model, settings.reranker_max_length
    )
    logger.info("[5/7] Reranker ready.")

    # 6. LLM + RAG chain
    from app.services.llm import build_llm
    from app.services.chain import build_rag_chain

    llm = build_llm(
        provider=settings.llm_provider,
        sbg_model_id=settings.sbg_model_id,
        sbg_base_url=settings.sbg_base_url,
        sbg_api_key=settings.sbg_api_key or "",
        openai_model=settings.openai_model,
        groq_model=settings.groq_model,
        groq_api_key=settings.groq_api_key or "",
        ollama_model=settings.ollama_model,
        hf_model=settings.huggingface_model,
        hf_token=settings.hf_token or "",
        fanar_model=settings.fanar_model,
        fanar_api_key=settings.fanar_api_key or "",
        fanar_base_url=settings.fanar_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    app.state.llm = llm
    app.state.rag_chain = build_rag_chain(llm, settings.system_role)
    logger.info("[6/7] LLM and RAG chain ready.")

    logger.info("=== Hadith RAG API ready to serve requests ===")
    yield

    # Shutdown
    logger.info("Shutting down Hadith RAG API…")
    if hasattr(app.state, "qdrant_client"):
        app.state.qdrant_client.close()


# ── App factory ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description="Production RAG API for Hadith Q&A",
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
