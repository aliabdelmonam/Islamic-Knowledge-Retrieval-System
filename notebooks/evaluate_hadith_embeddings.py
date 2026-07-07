import argparse
import pickle
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
from pathlib import Path

ARABIC_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]")
NON_WORD = re.compile(r"[^\w\s\u0600-\u06FF]+", re.UNICODE)


def normalize_arabic(text: str) -> str:
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


def sequence_matcher(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_arabic(a), normalize_arabic(b)).ratio()


def load_emb_lookup(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path, "rb") as f:
        emb_lookup = pickle.load(f)
    hadiths = list(emb_lookup.keys())
    vecs = np.vstack([np.asarray(v) for v in emb_lookup.values()])
    return hadiths, vecs


def embed_queries(model_name: str, queries: list[str], batch_size: int = 32) -> np.ndarray:
    model = SentenceTransformer(model_name)
    return model.encode(queries, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True)


def topk_similar(query_vec: np.ndarray, emb_matrix: np.ndarray, topk: int = 5):
    # cosine similarity
    qn = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    mats = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-12)
    sims = mats.dot(qn)
    idx = np.argsort(-sims)[:topk]
    return idx, sims[idx]


def evaluate(emb_pkl: Path, haqa_csv: Path, hf_model: str, topk: int, out_csv: Path):
    hadith_texts, emb_matrix = load_emb_lookup(emb_pkl)
    df = pd.read_csv(haqa_csv)
    df = df.dropna(subset=["Question_Text", "Hadith_Matn"]).reset_index(drop=True)

    queries = df["Question_Text"].astype(str).tolist()
    q_embs = embed_queries(hf_model, queries)

    results = []
    for i, row in df.iterrows():
        qvec = q_embs[i]
        idxs, sims = topk_similar(qvec, emb_matrix, topk=topk)
        retrieved = [hadith_texts[j] for j in idxs]

        # check hadith match (substring or sequence matcher fallback)
        truth = str(row["Hadith_Matn"]).strip()
        truth_norm = normalize_arabic(truth)
        hit = False
        method = "none"
        best_overlap = 0.0
        best_match = None
        for r in retrieved:
            if len(truth_norm) >= 12 and truth_norm in normalize_arabic(r):
                hit = True
                method = "substring"
                best_overlap = 1.0
                best_match = r
                break
            overlap = sequence_matcher(truth, r)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = r
        if not hit and best_overlap >= 0.45:
            hit = True
            method = "token_overlap"

        results.append({
            "Question_Id": row.get("Question_Id"),
            "Question_Text": row.get("Question_Text"),
            "Hadith_Matn": truth,
            "Hit": hit,
            "Method": method,
            "Best_Overlap": best_overlap,
            "Best_Match": best_match,
            "TopK_Retrieved": " ||| ".join(retrieved),
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)
    hit_rate = out_df["Hit"].mean()
    print(f"Evaluated {len(out_df)} examples — Hit rate@{topk}: {hit_rate:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--emb-pkl", type=str, required=True, help="Path to emb_lookup.pkl")
    p.add_argument("--haqa-csv", type=str, default="data/HAQA.csv")
    p.add_argument("--hf-model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--out", type=str, default="eval_runs/hadith_emb_eval.csv")
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    evaluate(Path(args.emb_pkl), Path(args.haqa_csv), args.hf_model, args.topk, Path(args.out))
