"""POST /api/v1/ask — full RAG: retrieve + hadith similarity + LLM generation."""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Request

from app.core.config import settings
from app.core.exceptions import LLMError, PipelineNotReadyError, RetrievalError
from app.schemas.request import AskRequest
from app.schemas.response import AskResponse, RetrievedItem
from app.services import chain as chain_svc
from app.services import retriever as retriever_svc

logger = logging.getLogger(__name__)
router = APIRouter()


def _to_retrieved_items(results) -> list[RetrievedItem]:
    return [
        RetrievedItem(hadith=r.hadith, sharh=r.sharh, rawy=r.rawy, source=r.source,
                      hokm=r.hokm, page_id=r.page_id, similarity_score=r.similarity_score)
        for r in results
    ]


@router.post("/ask", response_model=AskResponse)
async def ask_endpoint(body: AskRequest, request: Request) -> AskResponse:
    state = request.app.state
    for attr in ("vectorstore", "bm25_index", "all_chunks", "embedding_model", "rag_chain"):
        if getattr(state, attr, None) is None:
            raise PipelineNotReadyError(attr)

    use_agentic = body.use_agentic if body.use_agentic is not None else settings.use_agentic_rag
    session_id = body.session_id or str(uuid.uuid4())

    if use_agentic:
        if getattr(state, "agent_graph", None) is None:
            from app.services.agentic_rag import build_nodes
            state.agent_graph = build_nodes(
                llm=state.llm, vectorstore=state.vectorstore, bm25_index=state.bm25_index,
                all_chunks=state.all_chunks, embedding_model=state.embedding_model,
                qdrant_client=state.qdrant_client if getattr(state, "category_collection_ready", False) else None,
                category_collection_name=settings.category_collection_name,
                category_top_k=settings.category_top_k, k=body.k, fetch_k=settings.retriever_fetch_k,
                k_decay=settings.k_decay, fetch_k_decay=settings.fetch_k_decay,
                system_role=settings.system_role,
                hadith_bm25_index=getattr(state, "hadith_bm25_index", None),
                hadith_records=getattr(state, "hadith_records", None),
                hadith_search_top_k=settings.hadith_search_top_k,
            )

        from app.services.agentic_rag import run_agentic_rag
        try:
            result = run_agentic_rag(query=body.question, session_id=session_id, agent_graph=state.agent_graph,
                                     k=body.k, fetch_k=settings.retriever_fetch_k)
        except Exception as exc:
            logger.exception("Agentic RAG error: %s", exc)
            raise LLMError(str(exc)) from exc

        docs = result["good_documents"] or result["all_documents"]
        return AskResponse(
            answer=result["answer"], sources=_to_retrieved_items(docs),
            query_rewritten=result["final_query"] if result["final_query"] != result["original_query"] else None,
            agentic=True, loop_count=result["loop_count"], query_history=result["query_history"], session_id=session_id,
        )

    question = body.question
    query_rewritten: str | None = None
    if body.rewrite and settings.groq_api_key:
        from app.services.query_rewriter import rewrite_query
        rewritten = rewrite_query(question, settings.groq_api_key, settings.query_rewrite_model)
        if rewritten != question:
            query_rewritten, question = rewritten, rewritten

    try:
        results = retriever_svc.retrieve(
            query=question, vectorstore=state.vectorstore, bm25_index=state.bm25_index,
            all_chunks=state.all_chunks, embedding_model=state.embedding_model, k=body.k,
        )
    except Exception as exc:
        logger.exception("Retrieval error: %s", exc)
        raise RetrievalError(str(exc)) from exc

    try:
        answer = chain_svc.ask(question=body.question, chain=state.rag_chain, results=results)
    except Exception as exc:
        logger.exception("LLM error: %s", exc)
        raise LLMError(str(exc)) from exc

    return AskResponse(answer=answer, sources=_to_retrieved_items(results), query_rewritten=query_rewritten,
                       agentic=False, session_id=session_id)
