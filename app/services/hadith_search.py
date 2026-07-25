"""
Hadith-level search service using BM25.
Used to identify and fetch authentic hadiths and their metadata from the clean hadith dataset.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from rank_bm25 import BM25Okapi

from app.services.arabic_utils import normalize_arabic

logger = logging.getLogger(__name__)


@dataclass
class HadithRecord:
    page_id: str
    hadith: str
    hadith_clean: str
    rawy: str
    source: str
    hokm: str
    sharh: str
    url: str = ""
    takhrij: str = ""
    categories: str = ""
    mohadth: str = ""
    page: str = ""


def build_hadith_bm25(
    csv_path: Path,
    max_rows: int | None = None,
) -> tuple[BM25Okapi, list[HadithRecord]]:
    """
    Load hadiths from CSV, extract relevant columns, normalize the text,
    and build a BM25 index over `hadith_clean`.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Hadith CSV file not found at {csv_path}")

    logger.info("Reading hadith CSV for BM25: %s", csv_path)
    df = pd.read_csv(csv_path)
    if max_rows is not None:
        df = df.head(max_rows)

    # Filter out empty or null hadiths/hadith_clean
    df = df[df["hadith_clean"].notna() & df["hadith"].notna()]

    records: list[HadithRecord] = []
    corpus: list[list[str]] = []

    for _, row in df.iterrows():
        # Standardize strings, handle potential lists format in some fields
        def _clean_field(v):
            if pd.isna(v):
                return ""
            v_str = str(v).strip()
            # If it's stored as a string representation of a list: ['val'], strip symbols
            if v_str.startswith("[") and v_str.endswith("]"):
                try:
                    import ast
                    parsed = ast.literal_eval(v_str)
                    if isinstance(parsed, list):
                        return ", ".join(str(item) for item in parsed)
                except Exception:
                    pass
            return v_str

        rec = HadithRecord(
            page_id=_clean_field(row.get("page_id", "")),
            hadith=_clean_field(row.get("hadith", "")),
            hadith_clean=_clean_field(row.get("hadith_clean", "")),
            rawy=_clean_field(row.get("rawy", "")),
            source=_clean_field(row.get("source", "")),
            hokm=_clean_field(row.get("hokm", "")),
            sharh=_clean_field(row.get("sharh", "")),
            url=_clean_field(row.get("url", "")),
            takhrij=_clean_field(row.get("takhrij", "")),
            categories=_clean_field(row.get("categories", "")),
            mohadth=_clean_field(row.get("mohadth", "")),
            page=_clean_field(row.get("page", "")),
        )
        records.append(rec)

        # Tokenize normalized version for BM25
        normalized = normalize_arabic(rec.hadith_clean)
        corpus.append(normalized.split())

    logger.info("Building BM25 index over %d hadiths...", len(records))
    index = BM25Okapi(corpus)
    return index, records


def save_hadith_index(index: BM25Okapi, records: list[HadithRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"index": index, "records": records}, f)
    logger.info("Saved hadith BM25 index (%d records) to %s", len(records), path)


def load_hadith_index(path: Path) -> tuple[BM25Okapi, list[HadithRecord]]:
    if not path.exists():
        raise FileNotFoundError(f"Hadith BM25 index not found at {path}. Please run initialization script first.")
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info("Loaded hadith BM25 index (%d records)", len(data["records"]))
    return data["index"], data["records"]


def search_hadith(
    index: BM25Okapi,
    records: list[HadithRecord],
    query: str,
    k: int = 5,
) -> list[tuple[HadithRecord, float]]:
    """Return top-k (HadithRecord, score) pairs (score > 0 only)."""
    tokens = normalize_arabic(query).split()
    if not tokens:
        return []
    scores = index.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [(records[i], float(scores[i])) for i in ranked[:k] if scores[i] > 0]


def lookup_candidate_hadiths(
    index: BM25Okapi,
    records: list[HadithRecord],
    candidate_queries: list[str],
    k_per_query: int = 3,
) -> list[HadithRecord]:
    """
    Search list of candidate hadith queries, deduplicate results, and return matching HadithRecords.
    """
    matched_records: list[HadithRecord] = []
    seen_texts: set[str] = set()

    for q in candidate_queries:
        if not q.strip():
            continue
        results = search_hadith(index, records, q, k=k_per_query)
        for rec, score in results:
            # We can deduplicate based on clean text to avoid duplicate entries
            normalized = normalize_arabic(rec.hadith)
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                matched_records.append(rec)

    logger.info(
        "Candidate lookup: searched %d queries → retrieved %d unique records",
        len(candidate_queries), len(matched_records)
    )
    return matched_records
