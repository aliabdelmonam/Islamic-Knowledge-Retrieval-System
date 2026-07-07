"""Arabic text normalization utilities."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import pandas as pd

ARABIC_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]")
NON_WORD = re.compile(r"[^\w\s\u0600-\u06FF]+", re.UNICODE)


def normalize_arabic(text: str) -> str:
    """Remove diacritics, normalize letters, clean non-word chars."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text)
    text = ARABIC_TASHKEEL.sub("", text)
    text = text.replace("\ufeff", "")
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = NON_WORD.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_set(text: str) -> set[str]:
    return {t for t in normalize_arabic(text).split() if len(t) > 1}


def token_overlap_ratio(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta)


def sequence_matcher(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_arabic(a), normalize_arabic(b)).ratio()
