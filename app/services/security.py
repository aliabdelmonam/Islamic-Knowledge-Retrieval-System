# -*- coding: utf-8 -*-
"""
Heuristic prompt-injection detector (defense-in-depth layer, NOT a sole defense).
See notes at bottom of file for what this cannot catch and what to pair it with.
"""
import re
import unicodedata
import base64

# ---------- normalization ----------

ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\uFEFF\u2060]")
DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")

# common lookalike characters used to dodge literal string matches
HOMOGLYPH_MAP = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "А": "a", "Е": "e", "О": "o", "Р": "p", "С": "c", "Ѕ": "s", "Х": "x",
    "і": "i", "І": "i", "ј": "j", "ԁ": "d", "ѕ": "s", "ⅼ": "l",
})

LEET_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

_SPACED_RUN_RE = re.compile(
    r"\b(?:[A-Za-z\u0600-\u06FF][\s\.\-_\*]){3,}[A-Za-z\u0600-\u06FF]\b"
)

def _collapse_spaced_runs(text: str) -> str:
    """Collapses obfuscation like 'i.g.n.o.r.e' or 'ت ج ا ه ل' into one token."""
    def _join(m):
        return re.sub(r"[\s\.\-_\*]", "", m.group(0))
    return _SPACED_RUN_RE.sub(_join, text)

def normalize_text(text: str, apply_leet: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = text.translate(HOMOGLYPH_MAP)
    text = text.lower()
    text = DIACRITICS_RE.sub("", text)
    text = text.replace("ـ", "")
    text = re.sub(r"[أإآ]", "ا", text)
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = _collapse_spaced_runs(text)
    if apply_leet:
        text = text.translate(LEET_MAP)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _extract_base64_candidates(text: str):
    """Finds base64-ish tokens and tries to decode them for recursive scanning."""
    candidates = []
    for tok in re.findall(r"[A-Za-z0-9+/]{16,}={0,2}", text):
        try:
            decoded = base64.b64decode(tok, validate=True).decode("utf-8", errors="strict")
            if decoded.isprintable():
                candidates.append(decoded)
        except Exception:
            continue
    return candidates

# ---------- pattern groups ----------
# Each entry: (regex, weight). Weight lets you tune severity instead of a flat +1.

IGNORE_PATTERNS = [
    r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above|earlier|your|ur)?\s*(rules?|instructions?|guidelines?|prompts?)",
    r"disregard\s+(all\s+)?(previous|prior|above|your)?\s*(rules?|instructions?|guidelines?)",
    r"forget\s+(all\s+)?(your|the|everything)\s*(you\s+were\s+told|instructions?|rules?)?",
    r"discard\s+(everything|all)\s+(you\s+were\s+told|your\s+instructions?)",
    r"drop\s+(your\s+)?(guidelines?|rules?|restrictions?|instructions?)",
    r"stop\s+(following|obeying)\s+(your\s+)?(rules?|instructions?)",
    r"(new|updated)\s+instructions?\s*(override|replace|supersede)",
    r"تجاهل\s+(كل\s+)?(التعليمات|الارشادات|الاوامر|القواعد)",
    r"انس[يوا]?\s+(كل\s+)?(التعليمات|الاوامر|القواعد)",
    r"الغ[يوا]?\s+(كل\s+)?(التعليمات|القواعد)",
    r"متتجاهلش|تجاهلي|تجاهله",  # dialectal variants
]

PROMPT_EXTRACTION_PATTERNS = [
    r"system\s*prompt", r"hidden\s*prompt", r"system\s*instructions?",
    r"hidden\s*instructions?", r"system\s*message", r"developer\s*message",
    r"reveal\s+(your\s+)?(system\s+)?prompt", r"show\s+(me\s+)?(your\s+)?(system\s+)?prompt",
    r"(print|repeat|output)\s+(everything|the\s+text)\b.{0,40}?\b(written\s+)?(above|before)",
    r"what\s+(were|are)\s+(the\s+)?(exact\s+)?instructions?\s+(you\s+)?(were\s+)?given",
    r"what\s+instructions?\s+(did\s+you|were\s+you)\s+(receive|given|told)",
    r"repeat\s+everything\s+above",
    r"البرومبت", r"التعليمات\s+الداخليه", r"تعليمات\s+النظام",
    r"اعرض\s+التعليمات", r"اظهر\s+التعليمات", r"ايه\s+هي\s+التعليمات",
    r"قوليلي\s+(كل\s+حاجه\s+)?(مكتوبه|مكتوب)\s+فوق",
]

ROLE_OVERRIDE_PATTERNS = [
    r"act\s+as", r"pretend\s+(to\s+be|you\s+are)", r"roleplay",
    r"from\s+now\s+on\s+you\s+are", r"you\s+are\s+now",
    r"you\s+will\s+now\s+(behave|act|respond)\s+as",
    r"imagine\s+you\s+are\s+(an?\s+)?(ai\s+)?(without|with\s+no)",
    r"let'?s\s+play\s+a\s+game\s+where\s+you\s+have\s+no",
    r"behave\s+as\s+an?\s+assistant\s+with\s+no",
    r"تصرف\s+كانك", r"مثل\s+انك", r"اعتبر\s+نفسك", r"انت\s+الان",
    r"خليك\s+(دلوقتي\s+)?شخصيه\s+(تانيه|مختلفه)",
]

JAILBREAK_PATTERNS = [
    r"jailbreak", r"\bdan\b", r"do\s+anything\s+now", r"unrestricted",
    r"bypass\s+(your\s+)?(rules?|filters?|safety)", r"override\s+(your\s+)?(rules?|safety)",
    r"without\s+any\s+(rules?|restrictions?|limits?)", r"no\s+restrictions?\s+(from\s+now|at\s+all)",
    r"اكسر\s+القيود", r"تجاوز\s+القيود", r"بدون\s+قيود", r"تخطي\s+الحمايه",
]

# Note: removed bare "reasoning"/"cot" as standalone triggers — too many false
# positives on legitimate hadith/isnad questions. Only fire when paired with
# clear reveal/meta intent.
META_EXTRACTION_PATTERNS = [
    r"(show|reveal|print)\s+(me\s+)?(your\s+)?(internal\s+)?(reasoning|chain\s+of\s+thought|cot)\b",
    r"internal\s+reasoning", r"confidential\s+prompt", r"hidden\s+instructions?",
    r"طريقه\s+تفكيرك", r"خطوات\s+التفكير\s+الداخلي", r"التعليمات\s+السريه",
]

# Content that looks like an instruction smuggled inside "retrieved document"
# text — this is the RAG-specific attack surface (indirect injection).
DOC_INJECTION_PATTERNS = [
    r"\[?\s*system\s*(note|message)?\s*[:\]]",
    r"\[?\s*assistant\s*(note|instructions?)?\s*[:\]]",
    r"the\s+assistant\s+must\s+now",
    r"ملاحظه\s+للنظام", r"يجب\s+على\s+المساعد\s+الان",
]

GROUPS = [
    ("ignore_instructions", IGNORE_PATTERNS, 2),
    ("prompt_extraction", PROMPT_EXTRACTION_PATTERNS, 2),
    ("role_override", ROLE_OVERRIDE_PATTERNS, 1),
    ("jailbreak", JAILBREAK_PATTERNS, 2),
    ("meta_extraction", META_EXTRACTION_PATTERNS, 1),
    ("doc_injection", DOC_INJECTION_PATTERNS, 2),
]

COMPILED_GROUPS = [
    (name, re.compile("|".join(f"(?:{p})" for p in patterns), re.I), weight)
    for name, patterns, weight in GROUPS
]


def _scan(text: str):
    """Scans a single normalized string, returns (score, matched_groups)."""
    score = 0
    matched = []
    for name, pattern, weight in COMPILED_GROUPS:
        if pattern.search(text):
            score += weight
            matched.append(name)
    return score, matched


def prompt_injection_score(text: str, max_depth: int = 2):
    """
    Returns (score, matched_group_names).
    Checks the normal-normalized text, a leet-normalized variant, and
    recursively decodes/scans any base64 payloads found inside the text.
    """
    total_score = 0
    all_matched = set()

    variants = [normalize_text(text, apply_leet=False), normalize_text(text, apply_leet=True)]
    for v in variants:
        s, m = _scan(v)
        total_score += s
        all_matched.update(m)

    if max_depth > 0:
        for decoded in _extract_base64_candidates(text):
            s, m = prompt_injection_score(decoded, max_depth=max_depth - 1)
            if m:
                total_score += s
                all_matched.update(m)
                all_matched.add("base64_encoded_payload")

    return total_score, sorted(all_matched)
