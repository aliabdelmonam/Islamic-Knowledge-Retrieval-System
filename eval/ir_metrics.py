"""
Traditional Information-Retrieval metrics.

Every function takes a *relevance_list* — a list of 0/1 ints
whose length equals k, indicating whether each retrieved document
at that position is relevant (1) or not (0).

Example:  relevance_list = [0, 1, 0, 0, 1]  →  5 docs, relevant at pos 2 & 5
"""
from __future__ import annotations

import math
from typing import Sequence


# ── Per-query metrics ──────────────────────────────────────────────────────────

def recall_at_k(relevance: Sequence[int], total_relevant: int = 1) -> float:
    """Recall@k = |relevant ∩ top-k| / |relevant|."""
    if total_relevant == 0:
        return 0.0
    return sum(relevance) / total_relevant


def mrr_at_k(relevance: Sequence[int]) -> float:
    """Mean Reciprocal Rank@k = 1/rank of first relevant doc (0 if none)."""
    for i, r in enumerate(relevance):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def hit_rate_at_k(relevance: Sequence[int]) -> float:
    """Hit Rate@k = 1 if any relevant doc in top-k, else 0."""
    return 1.0 if any(relevance) else 0.0


def ndcg_at_k(relevance: Sequence[int]) -> float:
    """nDCG@k with binary relevance."""
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(relevance))
    # ideal: all relevant docs at the top
    ideal = sorted(relevance, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def precision_at_k(relevance: Sequence[int]) -> float:
    """Precision@k = |relevant ∩ top-k| / k."""
    if not relevance:
        return 0.0
    return sum(relevance) / len(relevance)


# ── Aggregation ────────────────────────────────────────────────────────────────

METRIC_FN = {
    "recall":    recall_at_k,
    "mrr":       mrr_at_k,
    "hit_rate":  hit_rate_at_k,
    "ndcg":      ndcg_at_k,
    "precision": precision_at_k,
}


def compute_ir_metrics(
    all_relevance: list[list[int]],
    k_values: list[int],
    metric_names: list[str],
) -> dict[str, float]:
    """
    Compute aggregate IR metrics across all queries at each k.

    Parameters
    ----------
    all_relevance : list[list[int]]
        Outer list = queries.  Inner list = relevance flags for
        max(k_values) retrieved docs (0/1), in retrieval order.
    k_values : list[int]
        E.g. [1, 3, 5, 10].
    metric_names : list[str]
        Subset of METRIC_FN keys to compute.

    Returns
    -------
    dict  e.g. {"recall@1": 0.35, "recall@5": 0.78, "mrr@5": 0.62, …}
    """
    results: dict[str, float] = {}
    n = len(all_relevance)
    if n == 0:
        return results

    for k in k_values:
        for name in metric_names:
            fn = METRIC_FN[name]
            total = 0.0
            for rel in all_relevance:
                rel_k = rel[:k]
                # recall_at_k expects a total_relevant arg
                if name == "recall":
                    total += fn(rel_k, total_relevant=1)
                else:
                    total += fn(rel_k)
            results[f"{name}@{k}"] = round(total / n, 4)

    return results
