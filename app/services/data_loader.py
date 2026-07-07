"""Load hadith CSV data into LangChain Documents."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_hadith_documents(
    csv_path: Path,
    embed_column: str,
    metadata_columns: list[str],
    max_rows: int | None = None,
) -> list[Document]:
    """
    Load the hadith CSV and return one Document per unique sharh value.
    Each document's metadata contains list-valued fields (one entry per hadith row).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if max_rows:
        df = df.head(max_rows)
    df = df[df[embed_column].notna()]

    docs: list[Document] = []
    for sharh_text, group in df.groupby(embed_column, sort=False):
        metadata = {
            col: group[col].fillna("").tolist()
            for col in metadata_columns
        }
        docs.append(
            Document(
                page_content=str(sharh_text).strip(),
                metadata=metadata,
            )
        )

    logger.info("Loaded %d documents from %s", len(docs), csv_path.name)
    return docs
