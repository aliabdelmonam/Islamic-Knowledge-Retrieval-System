"""Build and return embedding models."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_embeddings(
    provider: str,
    model_name: str,
    openai_model: str = "text-embedding-3-small",
    batch_size: int = 256,
):
    """
    Factory for LangChain embedding objects.
    Supports 'huggingface' and 'openai' providers.
    """
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        logger.info("Building OpenAI embeddings: %s", openai_model)
        return OpenAIEmbeddings(model=openai_model)

    if provider == "huggingface":
        import torch
        from langchain_huggingface import HuggingFaceEmbeddings

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Building HuggingFace embeddings: %s on %s", model_name, device)
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"batch_size": batch_size},
        )

    raise ValueError(f"Unknown embeddings provider: {provider!r}")
