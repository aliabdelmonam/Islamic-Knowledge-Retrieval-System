"""In-memory conversation history store, keyed by session_id.

Process-local and single-instance — fine for one API worker. If you scale to
multiple workers/instances or need history to survive a restart, swap the
dict-backed internals here for Redis (or any KV store) without changing the
public interface used by the chat endpoint.
"""
from __future__ import annotations

import asyncio
import time
import uuid

from app.providers import Message


class SessionStore:

    def __init__(self, max_messages: int = 20, ttl_seconds: int = 60 * 60 * 2):
        self._sessions: dict[str, list[Message]] = {}
        self._last_seen: dict[str, float] = {}
        self._max_messages = max_messages
        self._ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()

    async def get_history(self, session_id: str) -> list[Message]:
        async with self._lock:
            self._evict_expired()
            return list(self._sessions.get(session_id, []))

    async def append_turn(
        self,
        session_id: str,
        user_message: Message,
        assistant_message: Message,
    ) -> None:
        async with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.append(user_message)
            history.append(assistant_message)
            if len(history) > self._max_messages:
                del history[: len(history) - self._max_messages]
            self._last_seen[session_id] = time.monotonic()

    def new_session_id(self) -> str:
        return str(uuid.uuid4())

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [
            sid for sid, last in self._last_seen.items()
            if now - last > self._ttl_seconds
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._last_seen.pop(sid, None)