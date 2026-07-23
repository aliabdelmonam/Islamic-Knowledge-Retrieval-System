from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.services.arabic_utils import normalize_arabic, sequence_matcher
from eval.config import eval_settings

logger = logging.getLogger(__name__)


@dataclass
class EvalHitResult:
    hit: bool
    method: str | None
    best_overlap: float
    matched_doc_index: str | None


def compute_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


class Evaluator:
    def __init__(self, embeddings_model, llm_model):
        self.embeddings_model = embeddings_model
        self.llm_model = llm_model
        self.min_substring_len = eval_settings.min_substring_len
        self.min_token_overlap = eval_settings.min_token_overlap
        self.embedding_threshold = eval_settings.embedding_threshold

    def evaluate_hit(self, ground_truth: str, retrieved_hadiths: List[str], question: str) -> EvalHitResult:
        """
        Evaluate if any of the retrieved_hadiths match the ground_truth.
        Uses explicit fallback strategy:
        """
        target = normalize_arabic(ground_truth)
        if len(target) < 3:
            return EvalHitResult(False, "none", 0.0, None)

        # 1. Substring Match
        for i, hadith in enumerate(retrieved_hadiths):
            blob_norm = normalize_arabic(hadith)
            if len(target) >= self.min_substring_len and target in blob_norm:
                return EvalHitResult(True, "substring", 1.0, f"idx_{i}: {hadith}")

        # 2. Sequence Matcher
        best_seq_overlap = 0.0
        best_seq_idx = -1
        best_seq_hadith = ""

        for i, hadith in enumerate(retrieved_hadiths):
            blob_norm = normalize_arabic(hadith)
            overlap = sequence_matcher(target, blob_norm)
            if overlap > best_seq_overlap:
                best_seq_overlap = overlap
                best_seq_hadith = hadith
                best_seq_idx = i

        if best_seq_overlap >= self.min_token_overlap:
            return EvalHitResult(True, "token_overlap", best_seq_overlap, f"idx_{best_seq_idx}: {best_seq_hadith}")

        # 3. Embedding Similarity
        best_emb_sim = 0.0
        best_emb_idx = -1
        try:
            target_emb = self.embeddings_model.embed_query(target)
            docs_emb = self.embeddings_model.embed_documents([normalize_arabic(h) for h in retrieved_hadiths])
            
            for i, emb in enumerate(docs_emb):
                sim = compute_cosine_similarity(target_emb, emb)
                if sim > best_emb_sim:
                    best_emb_sim = sim
                    best_emb_idx = i
            
            if best_emb_sim >= self.embedding_threshold:
                return EvalHitResult(True, "embedding", best_emb_sim, f"idx_{best_emb_idx}: {retrieved_hadiths[best_emb_idx]}")
        except Exception as e:
            logger.warning(f"Embedding similarity failed: {e}")

        # 4. LLM Fallback
        # Submit all retrieved hadiths in a single request to the LLM to avoid overwhelming the API
        try:
            formatted_hadiths = "\n".join([f"[{i}] {h}" for i, h in enumerate(retrieved_hadiths)])
            prompt = eval_settings.llm_prompt_template.format(
                retrieved_hadiths=formatted_hadiths,
                ground_truth=ground_truth,
                question=question
            )
            messages = [
                SystemMessage(content="You are a strict evaluation system."),
                HumanMessage(content=prompt)
            ]
            response = self.llm_model.invoke(messages)
            content = response.content.strip().upper()
            
            if content.startswith("TRUE"):
                try:
                    idx_str = content.split(":")[1].strip()
                    hit_idx = int(idx_str)
                    hit_hadith = retrieved_hadiths[hit_idx]
                except (IndexError, ValueError):
                    hit_idx = 0
                    hit_hadith = retrieved_hadiths[0]

                return EvalHitResult(
                    hit=True,
                    method="llm_fallback",
                    best_overlap=max(best_seq_overlap, best_emb_sim),
                    matched_doc_index=f"idx_{hit_idx}: {hit_hadith}"
                )
        except Exception as e:
            logger.warning(f"LLM fallback failed: {e}")

        # None of the fallback stages succeeded
        overall_best_overlap = max(best_seq_overlap, best_emb_sim)
        formatted_hadiths = "\n\n\n".join([f"[{i}] {h}" for i, h in enumerate(retrieved_hadiths)])
        return EvalHitResult(False, "none", overall_best_overlap, formatted_hadiths)
