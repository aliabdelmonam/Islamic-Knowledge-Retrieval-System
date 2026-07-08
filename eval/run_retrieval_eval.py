"""
Retrieval-quality evaluation against the HAQA benchmark.

Runs the full hybrid retriever (dense + BM25 + cross-encoder rerank),
matches retrieved docs against ground-truth Hadith_Matn, and computes:
  • Custom IR metrics  (Recall@k, MRR@k, Hit Rate@k, nDCG@k, Precision@k)
  • RAGAS Non-LLM      (NonLLMContextRecall, NonLLMContextPrecisionWithReference)
  • RAGAS LLM-based    (ContextRecall, ContextPrecision, …)  — optional

Runs on Python 3.14 (project venv with torch/qdrant/etc).
RAGAS is called via subprocess on Python 3.12 (where ragas is installed).

Usage
-----
    python eval/run_retrieval_eval.py          # 20-sample smoke test
    python eval/run_retrieval_eval.py --all    # full 1597-row HAQA
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

# ── project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd
from tqdm import tqdm

from eval.eval_config import EVAL_CONFIG
from eval.ir_metrics import compute_ir_metrics

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retrieval_eval")


# ══════════════════════════════════════════════════════════════════════════════
#  Arabic normalisation & matching (mirrors evaluate_hadith_embeddings.py)
# ══════════════════════════════════════════════════════════════════════════════

ARABIC_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]")
NON_WORD = re.compile(r"[^\w\s\u0600-\u06FF]+", re.UNICODE)


def _normalize_arabic(text: str) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text)
    text = ARABIC_TASHKEEL.sub("", text)
    text = text.replace("\ufeff", "")
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = NON_WORD.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_match(
    ground_truth: str,
    retrieved_text: str,
    strategy: str = "substring_then_fuzzy",
    fuzzy_threshold: float = 0.45,
) -> tuple[bool, str, float]:
    """
    Check if *retrieved_text* matches the *ground_truth* hadith.
    Returns (hit, method, overlap_score).
    """
    gt_norm = _normalize_arabic(ground_truth)
    rt_norm = _normalize_arabic(retrieved_text)

    if len(gt_norm) >= 12 and gt_norm in rt_norm:
        return True, "substring", 1.0

    if strategy == "substring_then_fuzzy":
        ratio = SequenceMatcher(None, gt_norm, rt_norm).ratio()
        if ratio >= fuzzy_threshold:
            return True, "fuzzy", ratio

    return False, "none", 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline bootstrap  (same components as app/main.py lifespan)
# ══════════════════════════════════════════════════════════════════════════════

def _boot_pipeline():
    """Load embeddings, Qdrant, BM25, reranker — returns a dict of components."""
    from app.core.config import settings

    logger.info("Booting pipeline …")

    # 1. Embeddings
    from app.services.embeddings import build_embeddings
    embeddings = build_embeddings(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
        openai_model=settings.openai_embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    logger.info("[1/4] Embeddings ready.")

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
    vectorstore = get_vectorstore(client, embeddings, settings.collection_name)
    logger.info("[2/4] Qdrant vector store ready.")

    # 3. BM25 + chunks + parents
    from app.services.bm25_index import load_bm25
    bm25_index = load_bm25(settings.models_dir / "bm25_index.pkl")
    with open(settings.models_dir / "chunks.pkl", "rb") as f:
        all_chunks = pickle.load(f)
    with open(settings.models_dir / "parent_store.pkl", "rb") as f:
        parent_store = pickle.load(f)
    logger.info("[3/4] BM25 + chunks + parents ready (%d chunks, %d parents).",
                len(all_chunks), len(parent_store))

    # 4. Reranker
    from app.services.reranker import get_reranker
    from app.core.config import settings as _s
    reranker = get_reranker(_s.reranker_model, _s.reranker_max_length)
    logger.info("[4/4] Reranker ready.")

    return {
        "vectorstore": vectorstore,
        "bm25_index": bm25_index,
        "all_chunks": all_chunks,
        "parent_store": parent_store,
        "reranker": reranker,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  RAGAS via subprocess (Python 3.12)
# ══════════════════════════════════════════════════════════════════════════════

RAGAS_HELPER = PROJECT_ROOT / "eval" / "_ragas_helper.py"
PY312 = "py"  # will be called as: py -3.12


def _run_ragas_subprocess(
    questions: list[str],
    retrieved_contexts: list[list[str]],
    reference_contexts: list[list[str]],
    cfg: dict,
) -> dict[str, float]:
    """
    Serialize data to a temp JSON, invoke _ragas_helper.py on Python 3.12,
    read back the scores JSON.
    """
    # Check if any RAGAS metrics are enabled
    non_llm_on = cfg.get("ragas_non_llm", {}).get("enabled", False)
    llm_on = cfg.get("ragas_llm", {}).get("enabled", False)
    if not non_llm_on and not llm_on:
        logger.info("No RAGAS metrics enabled — skipping.")
        return {}

    logger.info("Preparing RAGAS subprocess call (Python 3.12) …")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "ragas_input.json"
        output_path = Path(tmpdir) / "ragas_output.json"

        payload = {
            "questions": questions,
            "retrieved_contexts": retrieved_contexts,
            "reference_contexts": reference_contexts,
            "config": {
                "ragas_llm": cfg.get("ragas_llm", {}),
                "ragas_non_llm": cfg.get("ragas_non_llm", {}),
            },
        }
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        cmd = [PY312, "-3.12", str(RAGAS_HELPER), str(input_path), str(output_path)]
        logger.info("Running: %s", " ".join(cmd))

        # Pass env vars through (for API keys)
        env = os.environ.copy()
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )

        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                logger.info("  [ragas] %s", line)
        if result.stderr:
            for line in result.stderr.strip().split("\n"):
                logger.warning("  [ragas] %s", line)

        if result.returncode != 0:
            logger.error("RAGAS subprocess failed (exit %d)", result.returncode)
            return {"ragas_error": f"exit code {result.returncode}"}

        if not output_path.exists():
            logger.error("RAGAS output file not found: %s", output_path)
            return {"ragas_error": "output file missing"}

        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
#  Main evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="HAQA retrieval evaluation")
    parser.add_argument("--all", action="store_true",
                        help="Run on ALL rows (override max_samples)")
    args = parser.parse_args()

    cfg = EVAL_CONFIG.copy()
    if args.all:
        cfg["max_samples"] = None

    # ── 1. Load HAQA dataset ──────────────────────────────────────────────
    logger.info("Loading HAQA from %s", cfg["haqa_csv"])
    df = pd.read_csv(cfg["haqa_csv"])
    df = df.dropna(subset=[cfg["question_column"], cfg["ground_truth_column"]])
    df = df.reset_index(drop=True)

    if cfg["max_samples"] is not None:
        df = df.sample(n=min(cfg["max_samples"], len(df)),
                       random_state=cfg["random_seed"])
        df = df.reset_index(drop=True)

    logger.info("Evaluating %d questions.", len(df))

    # ── 2. Boot retrieval pipeline ────────────────────────────────────────
    pipeline = _boot_pipeline()

    # ── 3. Retrieve for every question ────────────────────────────────────
    from app.services.retriever import retrieve

    max_k = max(cfg["retriever_k_values"])
    rows = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Retrieving"):
        question = str(row[cfg["question_column"]])
        ground_truth = str(row[cfg["ground_truth_column"]])

        results = retrieve(
            query=question,
            vectorstore=pipeline["vectorstore"],
            bm25_index=pipeline["bm25_index"],
            all_chunks=pipeline["all_chunks"],
            parent_store=pipeline["parent_store"],
            reranker=pipeline["reranker"],
            k=max_k,
            fetch_k=cfg["fetch_k"],
            reranker_batch_size=cfg["reranker_batch_size"],
        )

        # ── Match each retrieved doc against ground truth ─────────────
        relevance = []
        retrieved_texts = []
        best_overlap = 0.0
        best_method = "none"

        for r in results:
            text = r.chunk_text or r.sharh or ""
            retrieved_texts.append(text)
            hit, method, overlap = _is_match(
                ground_truth, text,
                strategy=cfg["match_strategy"],
                fuzzy_threshold=cfg["fuzzy_threshold"],
            )
            relevance.append(1 if hit else 0)
            if overlap > best_overlap:
                best_overlap = overlap
                best_method = method

        # pad if fewer than max_k results
        while len(relevance) < max_k:
            relevance.append(0)
            retrieved_texts.append("")

        rows.append({
            "question_id": row.get("Question_Id", i),
            "question": question,
            "ground_truth": ground_truth,
            "hit_any": int(any(relevance)),
            "best_overlap": round(best_overlap, 4),
            "best_method": best_method,
            "relevance": relevance,
            "retrieved_texts": retrieved_texts,
        })

    # ── 4. Compute custom IR metrics ──────────────────────────────────────
    all_relevance = [r["relevance"] for r in rows]
    ir_scores = compute_ir_metrics(
        all_relevance=all_relevance,
        k_values=cfg["retriever_k_values"],
        metric_names=cfg["ir_metrics"],
    )
    logger.info("Custom IR metrics:\n%s", json.dumps(ir_scores, indent=2))

    # ── 5. RAGAS evaluation (subprocess on Python 3.12) ───────────────────
    questions_list = [r["question"] for r in rows]
    retrieved_list = [r["retrieved_texts"] for r in rows]
    reference_list = [[r["ground_truth"]] for r in rows]

    ragas_scores = _run_ragas_subprocess(
        questions=questions_list,
        retrieved_contexts=retrieved_list,
        reference_contexts=reference_list,
        cfg=cfg,
    )
    if ragas_scores:
        logger.info("RAGAS scores:\n%s", json.dumps(ragas_scores, indent=2))

    # ── 6. Save outputs ──────────────────────────────────────────────────
    run_name = cfg["run_name"] or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg["output_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 6a. per-question CSV
    csv_rows = []
    for r in rows:
        csv_rows.append({
            "question_id": r["question_id"],
            "question": r["question"],
            "ground_truth": r["ground_truth"],
            "hit_any": r["hit_any"],
            "best_overlap": r["best_overlap"],
            "best_method": r["best_method"],
            "retrieved_top1": r["retrieved_texts"][0] if r["retrieved_texts"] else "",
            "relevance": str(r["relevance"]),
            "all_retrieved": " ||| ".join(r["retrieved_texts"]),
        })
    pd.DataFrame(csv_rows).to_csv(out_dir / "per_question.csv", index=False)

    # 6b. combined metrics JSON
    all_metrics = {**ir_scores, **ragas_scores}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    # 6c. frozen config
    serialisable_cfg = {}
    for k, v in cfg.items():
        if isinstance(v, Path):
            serialisable_cfg[k] = str(v)
        elif isinstance(v, dict):
            serialisable_cfg[k] = v
        else:
            serialisable_cfg[k] = v
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(serialisable_cfg, f, indent=2, ensure_ascii=False, default=str)

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("Results saved → %s", out_dir)
    logger.info("─" * 60)
    for k, v in all_metrics.items():
        logger.info("  %-35s  %s", k, v)
    logger.info("═" * 60)


if __name__ == "__main__":
    t0 = time.time()
    main()
    logger.info("Total wall time: %.1f s", time.time() - t0)
