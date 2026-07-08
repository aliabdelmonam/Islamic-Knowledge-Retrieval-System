"""
Evaluation configuration.
Edit this file to control which metrics run, sample sizes, LLM provider, etc.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVAL_CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────────────
    "haqa_csv": PROJECT_ROOT / "data" / "HAQA.csv",
    "question_column": "Question_Text",
    "ground_truth_column": "Hadith_Matn",

    # ── Sampling ──────────────────────────────────────────────────────────────
    "max_samples": 20,          # None = all 1597 rows
    "random_seed": 42,

    # ── Output ────────────────────────────────────────────────────────────────
    "output_dir": PROJECT_ROOT / "eval_runs",
    "run_name": None,           # None → auto-generates timestamp

    # ── Retrieval settings ────────────────────────────────────────────────────
    "retriever_k_values": [1, 3, 5, 10],   # evaluate metrics at each k
    "fetch_k": 25,                          # candidates before reranking
    "reranker_batch_size": 128,

    # ── Matching (for custom IR metrics) ──────────────────────────────────────
    #   "substring_then_fuzzy": exact substring first, then SequenceMatcher
    "match_strategy": "substring_then_fuzzy",
    "fuzzy_threshold": 0.45,

    # ── RAGAS — LLM-based metrics (need an LLM API key) ──────────────────────
    "ragas_llm": {
        "enabled": False,                   # flip to True when you want LLM eval
        "llm_provider": "groq",             # "groq" | "openai"
        "llm_model": "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",      # env-var name that holds the key
        "metrics": [
            "ContextRecall",
            "ContextPrecision",
            # "Faithfulness",               # uncomment to enable
            # "AnswerRelevancy",            # uncomment to enable
        ],
    },

    # ── RAGAS — Non-LLM metrics ──────────────────────────────────────────────
    "ragas_non_llm": {
        "enabled": True,
        "metrics": [
            "NonLLMContextRecall",
            "NonLLMContextPrecisionWithReference",
        ],
    },

    # ── Custom IR metrics (always computed, no LLM needed) ────────────────────
    "ir_metrics": [
        "recall",       # Recall@k
        "mrr",          # Mean Reciprocal Rank@k
        "hit_rate",     # Hit Rate@k  (binary per-query)
        "ndcg",         # nDCG@k
        "precision",    # Precision@k
    ],
}
