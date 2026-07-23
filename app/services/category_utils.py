"""
Category parsing and normalization utilities.

The `categories` column in the hadith dataset stores values like:
    "أذان - رفع الصوت بالأذان ، أطعمة - أكل اللحم"

Each entry is a "high-level - sub-category" pair separated by Arabic commas (،).
This module extracts the high-level categories for use in filtering.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def parse_high_level_categories(categories_value) -> list[str]:
    """
    Parse a raw categories string/list and return deduplicated high-level
    category names.

    Parameters
    ----------
    categories_value : str | list[str] | None
        Either a single string with entries separated by ``،`` (Arabic comma),
        or a list of such strings (as stored in grouped metadata).

    Returns
    -------
    list[str]
        Unique high-level category names, preserving first-seen order.

    Examples
    --------
    >>> parse_high_level_categories("أذان - رفع الصوت ، أطعمة - أكل اللحم")
    ['أذان', 'أطعمة']
    >>> parse_high_level_categories(None)
    []
    """
    if categories_value is None:
        return []

    # Normalize input to a list of strings
    if isinstance(categories_value, (list, tuple)):
        raw_strings = [str(v) for v in categories_value if v and str(v).strip()]
    else:
        raw_strings = [str(categories_value)]

    seen: set[str] = set()
    result: list[str] = []

    for raw in raw_strings:
        # Split on Arabic comma with optional surrounding whitespace
        for part in re.split(r"\s+،\s+", raw):
            part = part.strip()
            if not part:
                continue

            # Extract the main category (before first - or –)
            high_level = re.split(r"\s*[-–]\s*", part, maxsplit=1)[0].strip()

            if high_level and high_level not in seen:
                seen.add(high_level)
                result.append(high_level)

    return result