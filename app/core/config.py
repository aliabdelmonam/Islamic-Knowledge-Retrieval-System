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
    data_csv: Path = Path("data/Hadith_Filtered_Books.csv")
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


    # ── Embeddings ─────────────────────────────────────────────────────────────
    embedding_provider: Literal["huggingface"] = "huggingface"
    embedding_model: str = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
    embedding_dim: int = 768
    embedding_device: Literal["auto", "cpu", "cuda"] = "cpu"
    embedding_batch_size:int =16
    huggingface_cache_dir: Path | None = "~/.cache/huggingface/hub"

    # ── Vector Store (Qdrant) ──────────────────────────────────────────────────
    # qdrant_prefer_grpc: bool = True
    qdrant_timeout: int = 600
    qdrant_url: str         # overrides host/port when set
    qdrant_api_key: str 
    collection_name: str = "fatwas_arabic_triplet_matryoshka_v2"

    # ── Retriever ──────────────────────────────────────────────────────────────
    search_type: Literal["similarity", "mmr", "hybrid"] = "hybrid"
    retriever_k: int = 5
    retriever_fetch_k: int = 25          # for MMR
    lambda_mult: float = 0.5
    hybrid_alpha: float = 0.7           # 0=BM25 only, 1=dense only

    # ── Category Retriever ─────────────────────────────────────────────────
    category_csv: Path = Path("data/main_categories_description.csv")
    category_collection_name: str = "hadith_categories"
    category_top_k: int = 5              # categories to match per query


    # ── LLM ────────────────────────────────────────────────────────────────────
    llm_provider: Literal["gemini", "groq", "cohere"] = "gemini"

    GROQ_API_KEY: str

    GENERATION_BACKEND: str
    GOOGLE_API_KEY: str
    COHERE_API_KEY: str
    GENERATION_BACKEND:  Literal["gemini", "groq", "cohere"] = "gemini"


    groq_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    google_api_key: str | None = None
    cohere_model: str = "command-a-03-2025"
    cohere_api_key: str | None = None
    triage_temperature: float = 0.0
    llm_max_tokens: int = 3000

    # ── Prompt ─────────────────────────────────────────────────────────────────
    prompt_language: str = "ar"


    # ── Query Rewriting ────────────────────────────────────────────────────────
    query_rewrite_model: str = "llama-3.3-70b-versatile"

    # ── Agentic RAG ───────────────────────────────────────────────────────────
    use_agentic_rag: bool = True        # True = use LangGraph agent by default
    agentic_max_loops: int = 0           # Max retrieve-rewrite cycles
    k_decay: int = 0                     # Decrease k by this per agentic loop
    fetch_k_decay: int = 0               # Decrease fetch_k by this per agentic loop
    hadith_search_top_k: int = 1         # Results per candidate hadith lookup

    # ── Security ──────────────────────────────────────────────────────────────
    enable_prompt_injection_detection: bool = True
    prompt_injection_threshold: int = 1

    # ── API ────────────────────────────────────────────────────────────────────
    api_title: str = "Hadith RAG API"
    api_version: str = "1.0.0"
    debug: bool = False
    TAVILY_API_KEY:str
    # ── Retrieveal ──────────────────────────────────────────────────────────────
    retrieval_top_k: int = 3
    quran_json_path: Path = Path(r"C:\Users\aliab\OneDrive\Desktop\quran\quran_enriched.json")
    quran_index_dir: Path = Path(r"quran_index_final")
    hadith_csv_path: Path = Path(r"C:\Users\aliab\OneDrive\Desktop\hadith\Final_hadith.csv")
    hadith_index_dir: Path = Path(r"hadith_search_index")


    # ── LangSmith ──────────────────────────────────────────────────────────────
    LANGSMITH_TRACING: bool = True
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str | None = None

    # ── History ────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    history_window_size: int = 3

    # Hugging Face
    HF_HUB_DISABLE_SYMLINKS:str
    HF_HUB_DISABLE_SYMLINKS_WARNING:str
    HF_HUB_CACHE:str

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
        if not self.category_csv.is_absolute():
            self.category_csv = root / self.category_csv
        self.models_dir.mkdir(parents=True, exist_ok=True)
        return self


# Module-level singleton — import `settings` everywhere
settings = Settings()


def get_settings() -> Settings:
    return settings


# Backwards-compatible accessor used across the codebase
@property
def _llm_model(self) -> str:  # type: ignore[unused-def]
    """Return the configured model name for the selected LLM provider.

    Some parts of the codebase expect `settings.llm_model`. Expose a
    read-only attribute that maps the selected `llm_provider` to the
    provider-specific model setting.
    """
    provider = (self.llm_provider or "").lower()
    if provider == "groq":
        return getattr(self, "groq_model", "")
    if provider == "cohere":
        return getattr(self, "cohere_model", "")
    # default to gemini-style model name
    return getattr(self, "gemini_model", "")


# Attach property to Settings instance so `settings.llm_model` works
setattr(Settings, "response_llm", _llm_model)

setattr(Settings, "task_llm", _llm_model)

# ── LangSmith/LangChain Environment Variable Forwarding ───────────────────────────
# ── LangSmith/LangChain Environment Variable Forwarding ───────────────────────────
if settings.LANGSMITH_TRACING:
    import os

    if not settings.LANGSMITH_API_KEY or not settings.LANGSMITH_PROJECT:
        raise RuntimeError(
            "LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY and/or "
            "LANGSMITH_PROJECT is missing from your .env file."
        )

    project_name = settings.LANGSMITH_PROJECT.strip('"').strip("'")
    api_key = settings.LANGSMITH_API_KEY.strip('"').strip("'")

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = project_name

    # legacy aliases some langchain versions still read
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project_name