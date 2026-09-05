"""Hugging Face embedding provider interface and factory."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import  Any,List, Union

from app.core.config import Settings
from sentence_transformers import SentenceTransformer
logger = logging.getLogger(__name__)

class EmbeddingProvider(ABC):
    """Common contract for embedding providers."""

    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Return a LangChain-compatible embeddings object."""


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Load cached Hugging Face models, downloading them on first use."""
    def __init__(self, settings: "Settings") -> None:
        print("Initializing HuggingFaceEmbeddingProvider")
        self.settings = settings
        self.model = self._load_model()

    def _device(self) -> str:
        if self.settings.embedding_device != "auto":
            return self.settings.embedding_device

        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_model(self) ->SentenceTransformer:
        device = self._device()
        
        logger.info(
                "Loading embedding model %s on %s",
                self.settings.embedding_model,
                device,
            )
        
        # model = SentenceTransformer(
        #     self.settings.embedding_model,
        #     device=device,
        #     force_download=True,
        #     cache_folder=(
        #         str(self.settings.huggingface_cache_dir)
        #         if self.settings.huggingface_cache_dir
        #         else None
        #     ),
            
# )
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
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

        embeddings = self.model.encode(
            texts,
            batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.tolist()


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