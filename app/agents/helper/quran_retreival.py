import sys
from pathlib import Path

from app.services import build_index, search  # file 1 must live at app/services.py

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

INDEX_DIR = "quran_index_final"
QURAN_JSON = r"C:\Users\aliab\OneDrive\Desktop\quran\quran_enriched.json"

# 1. Build the index once if it doesn't exist yet, then reuse it.
if not Path(INDEX_DIR).exists():
    print("Index not found — building (first run only)...")
    build_index(QURAN_JSON, INDEX_DIR)
index_directory = INDEX_DIR

# 2. Perform a search for 'هذا نذير'
query = "هذا نذير"
print(f"--- Searching for: {query} ---")

results = search(index_directory, query, limit=3, max_edit_distance=1)

# 3. Display the results nicely
if not results:
    print("No results found.")
else:
    for i, res in enumerate(results, 1):
        tafsir = res.get("tafsir") or ""
        print(f"Result #{i}:")
        print(f"ID: {res['id']}")
        print(f"Surah: {res['surah_ar']} ({res['surah_en']})")
        print(f"Verse: {res['text']}")
        print(f"Tafsir (Short): {tafsir[:200]}...")
        print(f"Score: {res.get('_score', 0):.2f}")
        print("-" * 30)