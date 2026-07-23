from __future__ import annotations

from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EvalSettings(BaseSettings):
    # ── Paths & Data ──────────────────────────────────────────────────────────
    project_root: Path = Path(__file__).resolve().parents[1]
    dataset_csv: Path = Path("data/Bukhari_Eval.csv")
    output_dir: Path = Path("eval_runs")
    
    # ── Data Customization ────────────────────────────────────────────────────
    question_column: str = "question_ar"
    ground_truth_column: str = "hadith_ar"
    id_column: str = "Record_Id"
    
    # ── Execution ─────────────────────────────────────────────────────────────
    max_samples: int | None = 3          # None = full CSV
    random_seed: int = 42

    # ── Metrics Thresholds ────────────────────────────────────────────────────
    min_substring_len: int = 12
    min_token_overlap: float = 0.45
    embedding_threshold: float = 0.7
    
    # ── LLM Fallback ──────────────────────────────────────────────────────────
    llm_prompt_template: str = (
        "You are an expert Islamic evaluator. You will be provided with a list of retrieved hadiths "
        "and a ground truth hadith. Compare the ground truth to each of the retrieved hadiths. "
        "Determine if ANY of the retrieved hadiths carry the exact same meaning or it can answer the question, even if not similiar to ground truth"
        "even if the exact wording differs slightly.\n\n"
        "Ground Truth Hadith: {ground_truth}\n\n"
        "Retrieved Hadiths:\n{retrieved_hadiths}\n\n"
        "If you find a substantially matching hadith, reply EXACTLY with 'TRUE: <INDEX>' where <INDEX> is the index number (e.g., TRUE: 0).\n"
        "If NO hadith matches, reply exactly with 'FALSE'."
    )

    model_config = SettingsConfigDict(
        env_prefix="EVAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def resolve_paths(self) -> "EvalSettings":
        """Convert relative paths to absolute using project_root."""
        root = self.project_root
        if not self.dataset_csv.is_absolute():
            self.dataset_csv = root / self.dataset_csv
        if not self.output_dir.is_absolute():
            self.output_dir = root / self.output_dir
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self


# Module-level singleton
eval_settings = EvalSettings()
