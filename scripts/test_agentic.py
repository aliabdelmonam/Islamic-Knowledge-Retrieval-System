"""
Standalone test script for the Agentic RAG pipeline.

Run from the project root:
    python scripts/test_agentic.py

LangSmith tracing is enabled via the settings read from .env:
    LANGCHAIN_TRACING_V2=true          (or LANGSMITH_TRACING=true)
    LANGCHAIN_API_KEY=<key>
    LANGCHAIN_PROJECT=hadith-agentic-rag
"""
from __future__ import annotations

import logging
import os
import pickle
import sys
from pathlib import Path

# ── Make sure project root is on the path ──────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Load all config from .env via pydantic-settings ───────────────────────────
from app.core.config import settings  # reads .env automatically

# ── LangSmith: forward settings → env vars consumed by LangChain/LangGraph ────
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project or "hadith-agentic-rag")
if settings.langsmith_api_key:
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
if settings.langsmith_endpoint:
    os.environ.setdefault("LANGCHAIN_ENDPOINT", settings.langsmith_endpoint)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_agentic")

# ── Test queries ───────────────────────────────────────────────────────────────
TEST_QUERIES = [
    # "ما هي العبادة",
    # "بكم يجوز أن يتصدق الأنسان من ماله قبل موته؟",
    # "ينفع اقعد مع بنت الجيران لوحدنا ؟"
    "اعمل اية لو الدبانة وقعت في كوباية شاي؟"
]


def load_components():
    """Load all required ML components using app.core.config.settings."""
    print(f"fanar api key is : {settings.fanar_api_key}")
    logger.info("Loading embeddings [%s] …", settings.embedding_model)
    from app.services.embeddings import build_embeddings
    embeddings = build_embeddings(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
        openai_model=settings.openai_embedding_model,
        batch_size=settings.embedding_batch_size,
    )

    logger.info("Connecting to Qdrant [%s:%d] …", settings.qdrant_host, settings.qdrant_port)
    from app.services.vector_store import get_qdrant_client, get_vectorstore
    client = get_qdrant_client(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        prefer_grpc=settings.qdrant_prefer_grpc,
        timeout=settings.qdrant_timeout,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    vectorstore = get_vectorstore(client, embeddings, settings.collection_name)

    logger.info("Loading chunks and parent store from [%s] …", settings.models_dir)
    chunks_path = settings.models_dir / "chunks.pkl"
    parent_path = settings.models_dir / "parent_store.pkl"
    with open(chunks_path, "rb") as f:
        all_chunks = pickle.load(f)
    with open(parent_path, "rb") as f:
        parent_store = pickle.load(f)
    logger.info("  %d chunks, %d parents loaded.", len(all_chunks), len(parent_store))

    logger.info("Loading BM25 index …")
    from app.services.bm25_index import load_bm25
    bm25_index = load_bm25(settings.models_dir / "bm25_index.pkl")

    logger.info("Loading Reranker [%s] …", settings.reranker_model)
    from app.services.reranker import get_reranker
    reranker = get_reranker(settings.reranker_model, settings.reranker_max_length)

    logger.info("Building LLM [provider=%s] …", settings.llm_provider)
    from app.services.llm import build_llm
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

    return dict(
        llm=llm,
        vectorstore=vectorstore,
        bm25_index=bm25_index,
        all_chunks=all_chunks,
        parent_store=parent_store,
        reranker=reranker,
        qdrant_client=client,
    )


def main():
    logger.info("=== Agentic RAG — Manual Test Run ===")
    logger.info(
        "Provider: %s | Embedding: %s | Reranker: %s",
        settings.llm_provider,
        settings.embedding_model,
        settings.reranker_model,
    )

    # ls_key = os.getenv("LANGCHAIN_API_KEY")
    # print(f"langchain api key is : {ls_key}")
    # if not ls_key:
    #     logger.warning(
    #         "LANGCHAIN_API_KEY not set — LangSmith tracing will be disabled. "
    #         "Add LANGCHAIN_API_KEY to your .env to enable cloud tracing."
    #     )
    # else:
    #     logger.info("LangSmith tracing enabled → project: %s", os.getenv("LANGCHAIN_PROJECT"))

    components = load_components()

    from app.services.agentic_rag import build_nodes, run_agentic_rag

    logger.info("Compiling Agentic RAG graph …")
    agent_graph = build_nodes(
        llm=components["llm"],
        vectorstore=components["vectorstore"],
        bm25_index=components["bm25_index"],
        all_chunks=components["all_chunks"],
        parent_store=components["parent_store"],
        reranker=components["reranker"],
        qdrant_client=components["qdrant_client"],
        embedding_model_name=settings.embedding_model,
        category_collection_name=settings.category_collection_name,
        category_top_k=settings.category_top_k,
        k=settings.retriever_k,
        fetch_k=settings.retriever_fetch_k,
        system_role=settings.system_role,
    )
    logger.info("Graph compiled successfully.")

    separator = "=" * 70

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{separator}")
        print(f"[Query {i}/{len(TEST_QUERIES)}] {query}")
        print(separator)

        result = run_agentic_rag(query=query, agent_graph=agent_graph)

        print(f"  Loops performed : {result['loop_count']}")
        print(f"  Final query     : {result['final_query']}")
        print(f"  Queries tried   : {result['query_history']}")
        print(f"  Good docs       : {len(result['good_documents'])}")
        print(f"  Last-loop docs  : {len(result['all_documents'])}")
        print()
        print("── Answer ──")
        print(result["answer"])
        print()

        if result["good_documents"]:
            print("── Top good hadiths ──")
            for j, doc in enumerate(result["good_documents"][:3], 1):
                print(f"  [{j}] {doc.hadith[:120]}…")
                print(f"      Rawy: {doc.rawy}  |  Source: {doc.source}  |  Rerank: {doc.rerank_score:.4f}")

    print(f"\n{separator}")
    logger.info("=== Test run complete ===")


if __name__ == "__main__":
    main()
