"""Hugging Face embedding provider interface and factory."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.core.config import Settings
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Common contract for embedding providers."""

    @abstractmethod
    def create_embeddings(self) -> Any:
        """Return a LangChain-compatible embeddings object."""


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Load cached Hugging Face models, downloading them on first use."""

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings

    def _device(self) -> str:
        if self.settings.embedding_device != "auto":
            return self.settings.embedding_device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def create_embeddings(self) -> Any:
        from langchain_huggingface import HuggingFaceEmbeddings

        device = self._device()
        logger.info("Building Hugging Face embeddings: %s on %s", self.settings.embedding_model, device)
        return HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model,
            cache_folder=str(self.settings.huggingface_cache_dir) if self.settings.huggingface_cache_dir else None,
            model_kwargs={"device": device},
            encode_kwargs={
                "batch_size": self.settings.embedding_batch_size,
                "normalize_embeddings": True,
            },
        )

    def load_sentence_transformer(self) -> "SentenceTransformer":
        """Load from the HF cache or download the configured model when absent."""
        from sentence_transformers import SentenceTransformer

        device = self._device()
        logger.info("Loading Hugging Face model %r on %s (downloads if not cached)", self.settings.embedding_model, device)
        model = SentenceTransformer(
            self.settings.embedding_model,
            device=device,
            cache_folder=str(self.settings.huggingface_cache_dir) if self.settings.huggingface_cache_dir else None,
        )
        if device == "cuda":
            model.half()
        return model


class EmbeddingProviderFactory:
    """Resolve the configured embedding provider to its implementation."""

    @classmethod
    def create(cls, settings: "Settings") -> HuggingFaceEmbeddingProvider:
        if settings.embedding_provider != "huggingface":
            raise ValueError(
                f"Unsupported embedding provider {settings.embedding_provider!r}. "
                "Supported providers: huggingface."
            )
        return HuggingFaceEmbeddingProvider(settings)
