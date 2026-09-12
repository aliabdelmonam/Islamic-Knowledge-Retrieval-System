"""API v1 router — aggregates all endpoint modules."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints import ask, health, retrieve, chat

router = APIRouter(prefix="/api/v1")
router.include_router(health.router, tags=["health"])
router.include_router(ask.router, tags=["Chat"])
router.include_router(chat.router, tags=["Chat"])
router.include_router(retrieve.router, tags=["rag"])
