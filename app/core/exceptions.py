"""
Custom exception types for the Hadith RAG pipeline.
"""
from __future__ import annotations

from fastapi import HTTPException, status


class PipelineNotReadyError(HTTPException):
    """Raised when a required pipeline component hasn't been loaded yet."""
    def __init__(self, component: str):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Pipeline component not ready: {component}. "
                   "The service may still be initializing.",
        )


class RetrievalError(HTTPException):
    """Raised when retrieval fails."""
    def __init__(self, detail: str = "Retrieval failed"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        )


class LLMError(HTTPException):
    """Raised when the LLM call fails."""
    def __init__(self, detail: str = "LLM generation failed"):
        super().__init__(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail,
        )


class DataNotFoundError(HTTPException):
    """Raised when required data files are missing."""
    def __init__(self, path: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Required data file not found: {path}",
        )
