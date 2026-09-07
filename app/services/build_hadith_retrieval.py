"""
Whoosh full-text search over a Hadith CSV.
Indexes the "clean" text column; stores ALL columns so search returns the full row.

Usage:
    python hadith_search.py build --csv hadiths.csv --clean-col clean_text --index-dir hadith_index
    python hadith_search.py search --index-dir hadith_index --query "الصلاة" --limit 10
"""

import argparse
import csv
import os
import re
import sys

from whoosh import index
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.analysis import RegexTokenizer, Filter
from whoosh.qparser import QueryParser, OrGroup
from whoosh.query import FuzzyTerm
from nltk.stem.isri import ISRIStemmer

_isri = ISRIStemmer()

# Arabic-aware tokenizer: word chars incl. Arabic range + tatweel
ARABIC_WORD_RE = re.compile(r"[\w\u0600-\u06FF]+")
arabic_analyzer = RegexTokenizer(expression=ARABIC_WORD_RE)


class ISRIStemFilter(Filter):
    """Reduces each Arabic token to its ISRI root, so different word forms
    (verb conjugations, plurals, tanween suffixes, etc.) match each other."""

    def __call__(self, tokens):
        for t in tokens:
            t.text = _isri.stem(t.text)
            yield t


# Used for the fuzzy/partial-match field: tokenize -> stem
stem_analyzer = arabic_analyzer | ISRIStemFilter()


def build_schema(fieldnames, clean_col):
    """
    Dynamically build a schema:
      clean_col        -> TEXT, exact-token index (stored, shown in results)
      clean_col+'_stem'-> TEXT, root-stemmed index (searched against, not stored
                           separately since it's derived from clean_col)
      every other column -> STORED (returned in results, not searchable)
      doc_id -> ID, unique
    """
    stem_field = f"{clean_col}_stem"
    schema_fields = {
        "doc_id": ID(stored=True, unique=True),
        clean_col: TEXT(analyzer=arabic_analyzer, stored=True),
        stem_field: TEXT(analyzer=stem_analyzer, stored=False),
    }
    for col in fieldnames:
        if col != clean_col:
            schema_fields[col] = STORED
    return Schema(**schema_fields)


def build_index(csv_path, clean_col, index_dir):
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if clean_col not in fieldnames:
            sys.exit(f"Column '{clean_col}' not found. Available columns: {fieldnames}")

        os.makedirs(index_dir, exist_ok=True)
        schema = build_schema(fieldnames, clean_col)
        ix = index.create_in(index_dir, schema)
        writer = ix.writer(limitmb=512, procs=1)

        stem_field = f"{clean_col}_stem"
        count = 0
        for i, row in enumerate(reader):
            doc = {"doc_id": str(i)}
            for col in fieldnames:
                doc[col] = row.get(col) or ""
            # feed the same clean text into the stem field; Whoosh runs it
            # through stem_analyzer at index time
            doc[stem_field] = row.get(clean_col) or ""
            writer.add_document(**doc)
            count += 1

        writer.commit()
        print(f"Indexed {count} rows into '{index_dir}' (searchable column: '{clean_col}')")


def search_hadith(index_dir, query_str, clean_col, limit=10, fuzzy_fallback=True):
    """
    Searches the stemmed field so word-form differences (tanween, verb
    conjugation, plurals...) don't break matching, and missing/reordered
    words still return the best-overlap candidate (OrGroup, BM25-ranked).

    fuzzy_fallback: only kicks in if the plain stem search returns nothing.
    Fuzzy matching on stemmed roots is unreliable as a *primary* strategy —
    roots are short (2-4 chars), so edit-distance-1 fuzzy matches almost any
    random word and buries the real result under noise. It's also the slow
    part (full term-dict scan per term). So: stem match first, fuzzy only
    as a last resort, and only against the raw (unstemmed) field so it's
    matching real words, not near-random short roots.
    """
    stem_field = f"{clean_col}_stem"
    ix = index.open_dir(index_dir)
    with ix.searcher() as searcher:
        stemmed_terms = [_isri.stem(t) for t in ARABIC_WORD_RE.findall(query_str)]
        stemmed_query_str = " ".join(stemmed_terms) if stemmed_terms else query_str

        parser = QueryParser(stem_field, schema=ix.schema, group=OrGroup.factory(0.9))
        q = parser.parse(stemmed_query_str)
        results = searcher.search(q, limit=limit)

        if len(results) == 0 and fuzzy_fallback:
            # last resort: fuzzy on the ORIGINAL (unstemmed) words, with a
            # prefix lock so it doesn't wander into unrelated short words
            fuzzy_parser = QueryParser(
                clean_col,
                schema=ix.schema,
                group=OrGroup.factory(0.9),
                termclass=lambda fieldname, text, boost=1.0: FuzzyTerm(
                    fieldname, text, boost, maxdist=1, prefixlength=2
                ),
            )
            q = fuzzy_parser.parse(query_str)
            results = searcher.search(q, limit=limit)

        rows = []
        for hit in results:
            row = dict(hit)  # full stored row, all original CSV columns
            row["_score"] = hit.score
            rows.append(row)
        return rows


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--csv", required=True)
    p_build.add_argument("--clean-col", required=True)
    p_build.add_argument("--index-dir", default="hadith_index")

    p_search = sub.add_parser("search")
    p_search.add_argument("--index-dir", default="hadith_index")
    p_search.add_argument("--clean-col", required=True)
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()

    if args.cmd == "build":
        build_index(args.csv, args.clean_col, args.index_dir)
    elif args.cmd == "search":
        results = search_hadith(args.index_dir, args.query, args.clean_col, args.limit)
        for r in results:
            print("-" * 60)
            for k, v in r.items():
                print(f"{k}: {v}")


if __name__ == "__main__":
    main()