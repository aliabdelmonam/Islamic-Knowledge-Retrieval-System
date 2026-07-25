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

    # ── Category Retriever ─────────────────────────────────────────────────
    category_csv: Path = Path("data/main_categories_description.csv")
    category_collection_name: str = "hadith_categories"
    category_top_k: int = 5              # categories to match per query


    # ── LLM ────────────────────────────────────────────────────────────────────
    llm_provider: Literal[
        "openai", "ollama", "groq", "huggingface", "huggingface_local", "fanar", "sbg", "gemini"
    ] = "sbg"
    sbg_model_id: str = "qwen.qwen3-vl-235b-a22b"
    sbg_base_url: str = "http://apiaccess.iti.net.eg/api/v1"
    sbg_api_key: str | None = None

    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3.2"
    groq_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str | None = None
    huggingface_model: str = "silma-ai/SILMA-Kashif-2B-Instruct-v1.0"
    hf_token: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    google_api_key: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2000

    # ── Fanar ──────────────────────────────────────────────────────────────────
    fanar_model: str = "Fanar"
    fanar_api_key: str | None = None
    fanar_base_url: str = "https://api.fanar.qa/v1"

    # ── Prompt ─────────────────────────────────────────────────────────────────
    prompt_language: str = "ar"
    system_role: str = (
    "أنت عالم متخصص في العلوم الإسلامية. أجب عن أسئلة المستخدم اعتمادًا حصريًا على النصوص والسياق المقدم لك، "
    "ولا تضف معلومات أو أحكامًا شرعية من عندك إذا لم تكن مدعومة بالسياق. "
    "عند الاستشهاد بحديث نبوي، اذكر نص الحديث، ودرجة صحته، ومصدره، واسم الكتاب ورقم الحديث إن كان متوفرًا. "
    "إذا تعددت الأدلة، فرتبها بوضوح مع بيان وجه الاستدلال. "
    "إذا لم يكن في السياق ما يكفي للإجابة، فاذكر ذلك صراحة، ولا تخمّن أو تؤلف إجابة، "
    "وانصح المستخدم بالرجوع إلى عالم أو جهة إفتاء موثوقة للحصول على فتوى أو إجابة دقيقة."
)

    # ── Query Rewriting ────────────────────────────────────────────────────────
    query_rewrite_model: str = "llama-3.3-70b-versatile"

    # ── Agentic RAG ───────────────────────────────────────────────────────────
    use_agentic_rag: bool = True        # True = use LangGraph agent by default
    agentic_max_loops: int = 1           # Max retrieve-rewrite cycles
    k_decay: int = 0                     # Decrease k by this per agentic loop
    fetch_k_decay: int = 0               # Decrease fetch_k by this per agentic loop
    hadith_search_top_k: int = 1         # Results per candidate hadith lookup

    # ── API ────────────────────────────────────────────────────────────────────
    api_title: str = "Hadith RAG API"
    api_version: str = "1.0.0"
    debug: bool = False

    # ── LangSmith ──────────────────────────────────────────────────────────────
    langsmith_tracing: bool = True
    langsmith_endpoint: str | None = None
    langsmith_api_key: str | None = None
    langsmith_project: str | None = None

    # ── History ────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    history_window_size: int = 4

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

# ── LangSmith/LangChain Environment Variable Forwarding ───────────────────────────
if settings.langsmith_tracing:
    import os
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if settings.langsmith_project:
        # Strip any quotes that might be present in the .env file
        project_name = settings.langsmith_project.strip('"').strip("'")
        os.environ["LANGCHAIN_PROJECT"] = project_name
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint

