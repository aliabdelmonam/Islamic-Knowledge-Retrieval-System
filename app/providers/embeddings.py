"""Hugging Face embedding provider interface and factory (via LangChain)."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Union

from app.core.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Common contract for embedding providers."""

    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Return a LangChain-compatible embeddings object."""


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Load cached Hugging Face models, downloading them on first use.
    Uses LangChain's HuggingFaceEmbeddings under the hood so this provider
    can be dropped directly into LangChain retrievers/vectorstores as well."""

    def __init__(self, settings: "Settings") -> None:
        print("Initializing HuggingFaceEmbeddingProvider")
        self.settings = settings
        self.model = self._load_model()

    def _device(self) -> str:
        if self.settings.embedding_device != "auto":
            return self.settings.embedding_device

        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        device = self._device()

        logger.info(
            "Loading embedding model %s on %s",
            self.settings.embedding_model,
            device,
        )

        from langchain_huggingface import HuggingFaceEmbeddings

        # model_kwargs/encode_kwargs map onto the same SentenceTransformer(...)
        # constructor args and .encode(...) call args as before.
        model = HuggingFaceEmbeddings(
            model_name="Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
            model_kwargs={"device": device},
            encode_kwargs={
                "batch_size": self.settings.embedding_batch_size,
                "normalize_embeddings": True,
            },
            # cache_folder=(
            #     str(self.settings.huggingface_cache_dir)
            #     if self.settings.huggingface_cache_dir
            #     else None
            # ),
        )
        return model

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]

        # HuggingFaceEmbeddings.embed_documents already returns list[list[float]],
        # normalized per encode_kwargs above — same output shape as
        # `self.model.encode(...).tolist()` did previously.
        return self.model.embed_documents(texts)


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