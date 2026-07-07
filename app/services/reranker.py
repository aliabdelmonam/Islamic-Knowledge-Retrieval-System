"""Cross-encoder reranker wrapper (lazy singleton)."""
from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_reranker(model_name: str, max_length: int = 512):
    """
    Lazy-load the CrossEncoder reranker.
    Cached so only one instance is ever created per model_name.
    """
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading reranker: %s on %s", model_name, device)
    model = CrossEncoder(
        model_name,
        max_length=max_length,
        device=device,
    )
    return model


def rerank(
    model,
    query: str,
    passages: list[str],
    batch_size: int = 128,
) -> list[float]:
    """
    Return reranked scores (same order as *passages*).
    """
    if not passages:
        return []
    pairs = [[query, p] for p in passages]
    scores = model.predict(pairs, batch_size=batch_size).tolist()
    return [float(s) for s in scores]
