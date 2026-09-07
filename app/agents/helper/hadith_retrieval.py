import sys
from pathlib import Path
import pandas as pd

from app.services.build_hadith_retrieval import search_hadith, ISRIStemFilter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

INDEX_DIR = "hadith_search_index"
hadith_csv_path = r"C:\Users\aliab\OneDrive\Desktop\hadith\Final_hadith.csv"

index_directory = INDEX_DIR
SEARCH_COL = "clean_hadith"

df = pd.read_csv(hadith_csv_path)  

def hadith_by_category(category_str: str, limit: int = 5):
   
    mask = df['categories'].fillna('').str.contains(category_str, case=False, na=False)
    subset = df[mask].head(limit)
    # Return a list of records (dicts) so callers can iterate uniformly
    return subset.to_dict("records")

def hadith_by_text(query:str, limit:int=10):
    results = search_hadith(INDEX_DIR, query, SEARCH_COL, limit=limit)
    return results

def example():
    
    query = "ان الله يحب الجمال"
    print(f"--- Searching for: {query} ---")
    results = hadith_by_text(query, limit=3)
    if not results:
        print("No results found.")
    else:
        for i, res in enumerate(results, 1):
            print(f"Result #{i}:")
            print(f"Hadith: {res['hadith']}")
            print(f"Clean Hadith (Short): {res['clean_hadith'][:200]}...")
            print(f"Score: {res.get('_score', 0):.2f}")
            print("-" * 30)
    print(100*"=")
    print(100*"=")
    print(100*"=")
    cate="الصدق"
    results = hadith_by_category(cate)
    if not results:
        print("No results found.")
    else:
        for i, res in enumerate(results, 1):
            print(f"Result #{i}:")
            print(f"Hadith: {res['hadith']}")
            print(f"Clean Hadith (Short): {res['clean_hadith'][:200]}...")
            print(f"Score: {res.get('_score', 0):.2f}")
            print("-" * 30)

if __name__ == "__main__":
    example()