"""
Standalone RAGAS evaluation helper — runs on Python 3.12.

Called as a subprocess by run_retrieval_eval.py, which runs on
Python 3.14 (where torch / qdrant / etc. live).

Usage (called automatically):
    py -3.12  eval/_ragas_helper.py  <input_json>  <output_json>

Input JSON:  { questions, retrieved_contexts, reference_contexts, config }
Output JSON: { metric_name: score, … }
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ragas_helper")


def _build_metrics(cfg: dict):
    """Instantiate RAGAS metric objects based on config."""
    metrics = []

    # Non-LLM
    non_llm = cfg.get("ragas_non_llm", {})
    if non_llm.get("enabled"):
        from ragas.metrics import (
            NonLLMContextRecall,
            NonLLMContextPrecisionWithReference,
        )
        MAP = {
            "NonLLMContextRecall": NonLLMContextRecall,
            "NonLLMContextPrecisionWithReference": NonLLMContextPrecisionWithReference,
        }
        for name in non_llm.get("metrics", []):
            cls = MAP.get(name)
            if cls:
                metrics.append(cls())

    # LLM-based
    llm = cfg.get("ragas_llm", {})
    if llm.get("enabled"):
        from ragas.metrics._context_recall import ContextRecall
        from ragas.metrics._context_precision import ContextPrecision
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._answer_relevance import AnswerRelevancy

        MAP = {
            "ContextRecall": ContextRecall,
            "ContextPrecision": ContextPrecision,
            "Faithfulness": Faithfulness,
            "AnswerRelevancy": AnswerRelevancy,
        }
        for name in llm.get("metrics", []):
            cls = MAP.get(name)
            if cls:
                metrics.append(cls())

    return metrics


def _build_llm(cfg: dict):
    llm_cfg = cfg.get("ragas_llm", {})
    if not llm_cfg.get("enabled"):
        return None
    provider = llm_cfg["llm_provider"]
    model = llm_cfg["llm_model"]
    api_key = os.environ.get(llm_cfg.get("api_key_env", ""), "")

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, api_key=api_key, temperature=0)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=api_key, temperature=0)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]
    retrieved_contexts = data["retrieved_contexts"]
    reference_contexts = data["reference_contexts"]
    cfg = data["config"]

    metrics = _build_metrics(cfg)
    if not metrics:
        logger.warning("No RAGAS metrics enabled.")
        with open(output_path, "w") as f:
            json.dump({}, f)
        return

    llm = _build_llm(cfg)

    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    samples = []
    for q, ret, ref in zip(questions, retrieved_contexts, reference_contexts):
        samples.append(SingleTurnSample(
            user_input=q,
            retrieved_contexts=ret,
            reference_contexts=ref,
        ))

    dataset = EvaluationDataset(samples=samples)
    kwargs = {"dataset": dataset, "metrics": metrics}
    if llm is not None:
        kwargs["llm"] = llm

    logger.info("Running RAGAS evaluate() — %d metrics, %d samples …",
                len(metrics), len(samples))
    result = evaluate(**kwargs)

    scores = {}
    for key, val in result.items():
        if isinstance(val, (int, float)):
            scores[f"ragas_{key}"] = round(float(val), 4)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    logger.info("RAGAS scores saved → %s", output_path)
    for k, v in scores.items():
        logger.info("  %s = %s", k, v)


if __name__ == "__main__":
    main()
