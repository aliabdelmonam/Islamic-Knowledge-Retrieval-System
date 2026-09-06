from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

from whoosh import index
from whoosh.analysis import RegexTokenizer
from whoosh.fields import ID, Schema, STORED, TEXT
from whoosh.query import FuzzyTerm, Or

# ---------------------------------------------------------------------------
# Arabic normalization
# ---------------------------------------------------------------------------

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u0670]")
_TATWEEL = "\u0640"


def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = re.sub(r"[إأآاٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    text = re.sub(r"\s+", " ", text).strip()
    return text


_ARABIC_TOKENIZER = RegexTokenizer(r"[^\s]+")

SCHEMA = Schema(
    id=ID(stored=True, unique=True),
    text_norm=TEXT(analyzer=_ARABIC_TOKENIZER, stored=False),
    record=STORED,
)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_index(json_path: str, index_dir: str) -> str:
    """Reads a JSON array of ayah records and builds a Whoosh index in
    index_dir. Returns index_dir so callers can chain:
        idx = build_index(json, "quran_index")
    """
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    ix = index.create_in(index_dir, SCHEMA)
    writer = ix.writer(limitmb=256)

    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)

    with writer:  # commits on success, cancels on exception
        for rec in records:
            writer.add_document(
                id=str(rec["id"]),
                text_norm=normalize_arabic(rec["text_clean"]),
                record=rec,
            )

    return index_dir


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(
    index_dir: str,
    query_str: str,
    limit: int = 10,
    max_edit_distance: int = 1,
) -> list[dict]:
    """Fuzzy-searches text_clean and returns full original records (each with
    an added _score key), best match first."""
    ix = index.open_dir(index_dir)
    terms = normalize_arabic(query_str).split()
    if not terms:
        return []

    # Whoosh only supports maxdist of 0-2
    maxdist = max(0, min(int(max_edit_distance), 2))

    q = Or([
        FuzzyTerm("text_norm", t, maxdist=maxdist, prefixlength=1)
        for t in terms
    ])

    with ix.searcher() as searcher:
        hits = searcher.search(q, limit=limit)
        return [{**hit["record"], "_score": hit.score} for hit in hits]


def get_by_id(index_dir: str, ayah_id: str) -> Optional[dict]:
    """Exact lookup, no fuzzing."""
    ix = index.open_dir(index_dir)
    with ix.searcher() as searcher:
        hit = searcher.document(id=ayah_id)
        return hit["record"] if hit else None


__all__ = ["normalize_arabic", "build_index", "search", "get_by_id"]


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

def _example():
    try:  # avoid UnicodeEncodeError on Windows consoles/redirects
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    json_path = Path(r"C:\Users\aliab\OneDrive\Desktop\quran\quran_enriched.json")
    
    index_directory = build_index(str(json_path), "quran_index_final")  # now returns the dir

    print("-- exact-ish query --")
    for r in search(index_directory, "بسم الله الرحمن الرحيم"):
        print(r["id"], r["_score"], r["text_clean"])

    print("-- exact-ish query --")
    for r in search(index_directory, "هذا نذير"):
           print(r["id"], r["_score"], r["text_clean"])
   

    print("-- get_by_id --")
    print(get_by_id(index_directory, "2:255")["surah_ar"])


if __name__ == "__main__":
    _example()