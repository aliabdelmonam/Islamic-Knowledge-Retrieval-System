"""Response schemas for the Hadith RAG API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class RetrievedItem(BaseModel):
    hadith: str
    sharh: str
    rawy: str
    source: str
    hokm: str
    page_id: str = ""
    rerank_score: Optional[float] = None


class AskResponse(BaseModel):
    answer: str
    sources: list[RetrievedItem]
    query_rewritten: Optional[str] = None


class RetrieveResponse(BaseModel):
    results: list[RetrievedItem]
    query_used: str


class HealthResponse(BaseModel):
    status: str
    components: dict[str, bool]
    version: str
