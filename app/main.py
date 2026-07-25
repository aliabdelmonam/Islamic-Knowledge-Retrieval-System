"""
FastAPI application entry point.
Loads all heavy components (embeddings, Qdrant, BM25, LLM)
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

    # 1. Embeddings (LangChain wrapper — used by Qdrant vectorstore)
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
    logger.info("[2/6] Qdrant vector store ready.")

    # 2b. Category collection (for multi-stage retrieval)
    try:
        cat_info = client.get_collection(settings.category_collection_name)
        app.state.category_collection_ready = (cat_info.points_count or 0) > 0
    except Exception:
        app.state.category_collection_ready = False

    if app.state.category_collection_ready:
        logger.info("[2b/6] Category collection '%s' available.", settings.category_collection_name)
    else:
        logger.warning(
            "[2b/6] Category collection '%s' not found — "
            "run `python scripts/init_category_index.py` to enable multi-stage retrieval.",
            settings.category_collection_name,
        )

    # 3. Load child chunks from disk
    chunks_path = settings.models_dir / "chunks.pkl"

    if not chunks_path.exists():
        raise RuntimeError(
            f"Chunks not found. Run `python scripts/init_index.py` first.\n"
            f"Expected: {chunks_path}"
        )

    with open(chunks_path, "rb") as f:
        app.state.all_chunks = pickle.load(f)
    logger.info(
        "[3/6] Loaded %d child chunks.",
        len(app.state.all_chunks),
    )

    # 4. BM25 index
    from app.services.bm25_index import load_bm25
    app.state.bm25_index = load_bm25(settings.models_dir / "bm25_index.pkl")
    logger.info("[4/6] BM25 index ready.")

    # 4b. Hadith BM25 index for candidate lookup
    try:
        from app.services.hadith_search import load_hadith_index
        app.state.hadith_bm25_index, app.state.hadith_records = load_hadith_index(
            settings.models_dir / "hadith_bm25_index.pkl"
        )
        logger.info("[4b/6] Hadith BM25 index ready.")
    except Exception as exc:
        logger.warning("[4b/6] Failed to load Hadith BM25 index: %s", exc)
        app.state.hadith_bm25_index = None
        app.state.hadith_records = None

    # 5. SentenceTransformer for hadith-level embedding similarity
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading SentenceTransformer '%s' on %s for hadith similarity…", settings.embedding_model, device)
    st_model = SentenceTransformer(settings.embedding_model, device=device)
    if device == "cuda":
        st_model.half()
    app.state.embedding_model = st_model
    logger.info("[5/6] SentenceTransformer ready for hadith similarity.")

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
        gemini_model=settings.gemini_model,
        google_api_key=settings.google_api_key or "",
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    app.state.llm = llm
    app.state.rag_chain = build_rag_chain(llm, settings.system_role)
    logger.info("[6/6] LLM and RAG chain ready.")

    # 6b. Agentic RAG graph (optional)
    app.state.agent_graph = None
    if settings.use_agentic_rag:
        from app.services.agentic_rag import build_nodes
        app.state.agent_graph = build_nodes(
            llm=llm,
            vectorstore=app.state.vectorstore,
            bm25_index=app.state.bm25_index,
            all_chunks=app.state.all_chunks,
            embedding_model=st_model,
            qdrant_client=app.state.qdrant_client if app.state.category_collection_ready else None,
            embedding_model_name=settings.embedding_model if app.state.category_collection_ready else "",
            category_collection_name=settings.category_collection_name,
            category_top_k=settings.category_top_k,
            k=settings.retriever_k,
            fetch_k=settings.retriever_fetch_k,
            k_decay=settings.k_decay,
            fetch_k_decay=settings.fetch_k_decay,
            system_role=settings.system_role,
            hadith_bm25_index=app.state.hadith_bm25_index,
            hadith_records=app.state.hadith_records,
            hadith_search_top_k=settings.hadith_search_top_k,
        )
        logger.info("[6b] Agentic RAG graph compiled.")
    else:
        logger.info("[6b] Agentic RAG disabled (USE_AGENTIC_RAG=false).")

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
