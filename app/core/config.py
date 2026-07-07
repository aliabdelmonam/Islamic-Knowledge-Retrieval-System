"""
Application configuration via Pydantic BaseSettings.
Values are read from environment variables / .env file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Paths ──────────────────────────────────────────────────────────────────
    project_root: Path = Path(__file__).resolve().parents[2]
    data_csv: Path = Path("data/semantic_clustered_hadiths_per_sharh.csv")
    models_dir: Path = Path("models")  # bm25_index.pkl, parent_store.pkl
    chroma_dir: Path = Path("chroma_db")  # kept for reference / migration
    collection_name: str = "hadith_rag"

    # ── Data ───────────────────────────────────────────────────────────────────
    max_rows: int | None = None          # None = full CSV
    embed_column: str = "sharh"
    metadata_columns: list[str] = Field(
        default=[
            "page_id", "url", "categories", "sharh", "hadith",
            "rawy", "mohadth", "source", "page", "hokm", "takhrij",
        ]
    )

    # ── Chunking ───────────────────────────────────────────────────────────────
    chunk_size: int = 450
    chunk_overlap: int = 40

    # ── Embeddings ─────────────────────────────────────────────────────────────
    embedding_provider: Literal["huggingface", "openai"] = "huggingface"
    embedding_model: str = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 768
    embedding_batch_size: int = 256

    # ── Vector Store (Qdrant) ──────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_prefer_grpc: bool = True
    qdrant_timeout: int = 600
    qdrant_url: str | None = None        # overrides host/port when set
    qdrant_api_key: str | None = None

    # ── Retriever ──────────────────────────────────────────────────────────────
    search_type: Literal["similarity", "mmr", "hybrid"] = "hybrid"
    retriever_k: int = 5
    retriever_fetch_k: int = 25          # for MMR
    lambda_mult: float = 0.5
    hybrid_alpha: float = 0.7           # 0=BM25 only, 1=dense only

    # ── Reranker ───────────────────────────────────────────────────────────────
    reranker_model: str = "Omartificial-Intelligence-Space/ARA-Reranker-V1"
    reranker_max_length: int = 512
    reranker_batch_size: int = 128

    # ── LLM ────────────────────────────────────────────────────────────────────
    llm_provider: Literal[
        "openai", "ollama", "groq", "huggingface", "huggingface_local", "fanar", "sbg"
    ] = "sbg"
    sbg_model_id: str = "openai.gpt-oss-20b-1:0"
    sbg_base_url: str = "http://apiaccess.iti.net.eg/api/v1"
    sbg_api_key: str | None = None

    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3.2"
    groq_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str | None = None
    huggingface_model: str = "silma-ai/SILMA-Kashif-2B-Instruct-v1.0"
    hf_token: str | None = None
    llm_temperature: float = 0.1
    llm_max_tokens: int = 512

    # ── Fanar ──────────────────────────────────────────────────────────────────
    fanar_model: str = "Fanar-C-2-27B"
    fanar_api_key: str | None = None
    fanar_base_url: str = "https://api.fanar.qa/v1"

    # ── Prompt ─────────────────────────────────────────────────────────────────
    prompt_language: str = "ar"
    system_role: str = (
        "انت عالم دين اسلامي تجاوب علي اسئلة من خلال النص المسند اليك و حاول تجنب تاليف كلام ديني "
        "و قم بارفاق الاحاديث الواردة و صحتها و مصدر الاحاديث المتسخدمة"
        "اذا لم تجد جوابا في السياق . ارشده الي استشارة عالم اسلامي افضل للحصول علي اجابة دقيقة"
    )

    # ── Query Rewriting ────────────────────────────────────────────────────────
    query_rewrite_model: str = "llama-3.3-70b-versatile"

    # ── API ────────────────────────────────────────────────────────────────────
    api_title: str = "Hadith RAG API"
    api_version: str = "1.0.0"
    debug: bool = False

    # ── LangSmith ──────────────────────────────────────────────────────────────
    langsmith_tracing: bool = False
    langsmith_endpoint: str | None = None
    langsmith_api_key: str | None = None
    langsmith_project: str | None = None

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "case_sensitive": False,
    }

    @model_validator(mode="after")
    def resolve_paths(self) -> "Settings":
        """Convert relative paths to absolute using project_root."""
        root = self.project_root
        if not self.data_csv.is_absolute():
            self.data_csv = root / self.data_csv
        if not self.models_dir.is_absolute():
            self.models_dir = root / self.models_dir
        if not self.chroma_dir.is_absolute():
            self.chroma_dir = root / self.chroma_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        return self


# Module-level singleton — import `settings` everywhere
settings = Settings()
