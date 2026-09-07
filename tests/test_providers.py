"""Offline unit tests for provider construction and model loading."""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.providers.embeddings import EmbeddingProviderFactory, HuggingFaceEmbeddingProvider
from app.providers.llm import CohereProvider, GeminiProvider, GroqProvider, LLMProviderFactory


def settings(**overrides):
    values = {
        "llm_provider": "gemini", "gemini_model": "gemini-test", "google_api_key": "google-key",
        "groq_model": "groq-test", "groq_api_key": "groq-key",
        "cohere_model": "cohere-test", "cohere_api_key": "cohere-key",
        "llm_temperature": 0.2, "llm_max_tokens": 321,
        "embedding_provider": "huggingface", "embedding_model": "hf-test",
        "embedding_batch_size": 64, "embedding_device": "cpu", "huggingface_cache_dir": Path("model-cache"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class LLMProviderTests(unittest.TestCase):
    def _module(self, module_name: str, constructor_name: str):
        module = types.ModuleType(module_name)
        calls = []
        setattr(module, constructor_name, lambda **kwargs: calls.append(kwargs) or kwargs)
        return module, calls

    def test_gemini_forwards_configuration(self):
        module, calls = self._module("langchain_google_genai", "ChatGoogleGenerativeAI")
        with patch.dict(sys.modules, {"langchain_google_genai": module}):
            GeminiProvider(settings()).create_chat_model()
        self.assertEqual(calls[0]["model"], "gemini-test")
        self.assertEqual(calls[0]["max_output_tokens"], 321)

    def test_groq_forwards_configuration(self):
        module, calls = self._module("langchain_groq", "ChatGroq")
        with patch.dict(sys.modules, {"langchain_groq": module}):
            GroqProvider(settings()).create_chat_model()
        self.assertEqual(calls[0]["api_key"], "groq-key")
        self.assertEqual(calls[0]["max_tokens"], 321)

    def test_cohere_forwards_configuration(self):
        module, calls = self._module("langchain_cohere", "ChatCohere")
        with patch.dict(sys.modules, {"langchain_cohere": module}):
            CohereProvider(settings()).create_chat_model()
        self.assertEqual(calls[0]["cohere_api_key"], "cohere-key")
        self.assertEqual(calls[0]["model"], "cohere-test")

    def test_invalid_provider_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported LLM provider"):
            LLMProviderFactory.create(settings(), provider="unknown")

    def test_missing_key_is_rejected_before_import(self):
        with self.assertRaisesRegex(ValueError, "GOOGLE_API_KEY"):
            GeminiProvider(settings(google_api_key=None)).create_chat_model()


class EmbeddingProviderTests(unittest.TestCase):
    def test_embeddings_forward_device_cache_and_batch_size(self):
        module = types.ModuleType("langchain_huggingface")
        calls = []
        module.HuggingFaceEmbeddings = lambda **kwargs: calls.append(kwargs) or kwargs
        with patch.dict(sys.modules, {"langchain_huggingface": module}):
            HuggingFaceEmbeddingProvider(settings()).create_embeddings()
        self.assertEqual(calls[0]["model_kwargs"], {"device": "cpu"})
        self.assertEqual(calls[0]["encode_kwargs"]["batch_size"], 64)
        self.assertEqual(calls[0]["cache_folder"], "model-cache")

    def test_loader_uses_cache_folder_and_cpu(self):
        module = types.ModuleType("sentence_transformers")
        calls = []
        module.SentenceTransformer = lambda *args, **kwargs: calls.append((args, kwargs)) or SimpleNamespace(half=lambda: None)
        with patch.dict(sys.modules, {"sentence_transformers": module}):
            HuggingFaceEmbeddingProvider(settings()).load_sentence_transformer()
        self.assertEqual(calls[0][0], ("hf-test",))
        self.assertEqual(calls[0][1]["device"], "cpu")
        self.assertEqual(calls[0][1]["cache_folder"], "model-cache")

    def test_only_huggingface_embeddings_are_supported(self):
        with self.assertRaisesRegex(ValueError, "Unsupported embedding provider"):
            EmbeddingProviderFactory.create(settings(embedding_provider="other"))
