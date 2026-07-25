import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

# Add project root to sys.path so 'app' and 'eval' can be imported directly
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
from tqdm import tqdm

from app.core.config import settings
from app.services.arabic_utils import normalize_arabic
from app.services.bm25_index import load_bm25
from app.services.embeddings import build_embeddings
from app.services.llm import build_llm
from app.services.retriever import retrieve, retrieve_with_category_filter
from app.services.vector_store import get_qdrant_client, get_vectorstore
from eval.config import eval_settings
from eval.evaluator import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_runner")

def load_components():
    logger.info("Loading components for Evaluation...")
    # Embeddings
    embeddings = build_embeddings(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
        openai_model=settings.openai_embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    
    # Qdrant vector store
    client = get_qdrant_client(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        prefer_grpc=settings.qdrant_prefer_grpc,
        timeout=settings.qdrant_timeout,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    vectorstore = get_vectorstore(client, embeddings, settings.collection_name)
    
    cat_collection_ready = False
    try:
        cat_info = client.get_collection(settings.category_collection_name)
        cat_collection_ready = (cat_info.points_count or 0) > 0
    except Exception:
        pass
    
    # Chunks
    chunks_path = settings.models_dir / "chunks.pkl"
    if not chunks_path.exists():
        logger.error("Missing chunks.")
        sys.exit(1)
        
    with open(chunks_path, "rb") as f:
        all_chunks = pickle.load(f)
        
    # BM25
    bm25_index = load_bm25(settings.models_dir / "bm25_index.pkl")
    
    # SentenceTransformer for hadith similarity
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(settings.embedding_model, device=device)
    if device == "cuda":
        embedding_model.half()
    
    # LLM
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
    
    # Agentic RAG
    agent_graph = None
    if settings.use_agentic_rag:
        try:
            from app.services.hadith_search import load_hadith_index
            hadith_bm25_index, hadith_records = load_hadith_index(
                settings.models_dir / "hadith_bm25_index.pkl"
            )
        except Exception:
            hadith_bm25_index = None
            hadith_records = None
            
        from app.services.agentic_rag import build_nodes
        agent_graph = build_nodes(
            llm=llm,
            vectorstore=vectorstore,
            bm25_index=bm25_index,
            all_chunks=all_chunks,
            embedding_model=embedding_model,
            qdrant_client=client if cat_collection_ready else None,
            embedding_model_name=settings.embedding_model if cat_collection_ready else "",
            category_collection_name=settings.category_collection_name,
            category_top_k=settings.category_top_k,
            k=settings.retriever_k,
            fetch_k=settings.retriever_fetch_k,
            k_decay=settings.k_decay,
            fetch_k_decay=settings.fetch_k_decay,
            system_role=settings.system_role,
            hadith_bm25_index=hadith_bm25_index,
            hadith_records=hadith_records,
            hadith_search_top_k=settings.hadith_search_top_k,
        )
    
    return {
        "embeddings": embeddings,
        "qdrant_client": client,
        "vectorstore": vectorstore,
        "cat_collection_ready": cat_collection_ready,
        "all_chunks": all_chunks,
        "bm25_index": bm25_index,
        "embedding_model": embedding_model,
        "llm": llm,
        "agent_graph": agent_graph,
    }

def main():
    logger.info(f"Using dataset: {eval_settings.dataset_csv}")
    if not eval_settings.dataset_csv.exists():
        logger.error("Evaluation dataset not found!")
        sys.exit(1)
        
    df = pd.read_csv(eval_settings.dataset_csv)
    df = df.dropna(subset=[eval_settings.question_column, eval_settings.ground_truth_column])
    if eval_settings.max_samples:
        df = df.sample(n=min(eval_settings.max_samples, len(df)), random_state=eval_settings.random_seed)
    df = df.reset_index(drop=True)
    
    comps = load_components()
    evaluator = Evaluator(
        embeddings_model=comps["embeddings"],
        llm_model=comps["llm"]
    )
    
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row[eval_settings.question_column]
        ground_truth = row[eval_settings.ground_truth_column]
        record_id = row.get(eval_settings.id_column, "UNKNOWN")
        
        # Retrieval
        q_norm = normalize_arabic(question)
        if settings.use_agentic_rag and comps.get("agent_graph"):
            from app.services.agentic_rag import run_agentic_rag
            import uuid
            result = run_agentic_rag(
                query=question,
                session_id=str(uuid.uuid4()),
                agent_graph=comps["agent_graph"],
                k=3,
                fetch_k=8,
            )
            # Agentic generator uses 'good_documents' if not empty, otherwise fallback to 'all_documents'
            docs = result["good_documents"] if result["good_documents"] else result["all_documents"]
        elif comps["cat_collection_ready"]:
            docs = retrieve_with_category_filter(
                query=q_norm,
                vectorstore=comps["vectorstore"],
                bm25_index=comps["bm25_index"],
                all_chunks=comps["all_chunks"],
                embedding_model=comps["embedding_model"],
                qdrant_client=comps["qdrant_client"],
                embedding_model_name=settings.embedding_model,
                category_collection_name=settings.category_collection_name,
                category_top_k=settings.category_top_k,
                k=3,
                fetch_k=8,
            )
        else:
            docs = retrieve(
                query=q_norm,
                vectorstore=comps["vectorstore"],
                bm25_index=comps["bm25_index"],
                all_chunks=comps["all_chunks"],
                embedding_model=comps["embedding_model"],
                k=3,
                fetch_k=8,
            )
            
        all_hadiths = [d.hadith for d in docs]
        
        hit_result = evaluator.evaluate_hit(
            ground_truth=ground_truth,
            retrieved_hadiths=all_hadiths,
            question=question
        )
        
        results.append({
            "record_id": record_id,
            "question": question,
            "ground_truth": ground_truth,
            "hit": hit_result.hit,
            "hit_method": hit_result.method,
            "best_overlap": round(hit_result.best_overlap, 4),
            "n_retrieved": len(docs),
            "matched_hadith_text": hit_result.matched_doc_index
        })
        
    res_df = pd.DataFrame(results)
    hit_rate = res_df["hit"].mean()
    logger.info(f"Total Hits: {res_df['hit'].sum()} / {len(res_df)}")
    logger.info(f"Overall Hit Rate: {hit_rate:.1%}")
    logger.info("\nHit Method Distribution:")
    logger.info(res_df["hit_method"].value_counts(dropna=False).to_string())
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = eval_settings.output_dir / f"eval_results_{timestamp}.csv"
    res_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Results saved to: {out_path}")

if __name__ == "__main__":
    main()
