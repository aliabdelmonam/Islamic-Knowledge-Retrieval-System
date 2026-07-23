"""Request schemas for the Hadith RAG API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question in Arabic")
    k: int = Field(5, ge=1, le=20, description="Number of results to retrieve")
    rewrite: bool = Field(True, description="Whether to rewrite the query from colloquial to MSA")
    use_agentic: bool | None = Field(None, description="Override agentic RAG mode (None = use server default)")
    filters: dict | None = Field(None, description="Optional metadata filters")


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query in Arabic")
    k: int = Field(5, ge=1, le=20, description="Number of results")
    filters: dict | None = Field(None, description="Optional metadata filters")
