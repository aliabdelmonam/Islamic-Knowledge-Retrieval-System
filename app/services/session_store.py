"""Conversation history store, keyed by session_id, backed by LangChain's
BaseChatMessageHistory (HumanMessage/AIMessage) per session.

Process-local and single-instance — fine for one API worker. To scale out or
persist across restarts, swap `_new_history()` to build a Redis-backed
BaseChatMessageHistory (e.g. langchain-community's RedisChatMessageHistory)
instead of InMemoryChatMessageHistory. Nothing else in this class needs to
change, since eviction/get_last_k/append_turn only talk to the
BaseChatMessageHistory interface, not to InMemoryChatMessageHistory directly.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Optional

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from app.providers import Message


def _to_lc_message(msg: Message) -> BaseMessage:
    """Only user/assistant turns belong in stored history — system prompts
    are injected fresh per-request in answer_generation, not persisted."""
    if msg.role == "user":
        return HumanMessage(content=msg.content)
    if msg.role == "assistant":
        return AIMessage(content=msg.content)
    raise ValueError(f"Unsupported role for history storage: {msg.role!r}")


def _from_lc_message(msg: BaseMessage) -> Message:
    if isinstance(msg, HumanMessage):
        role = "user"
    elif isinstance(msg, AIMessage):
        role = "assistant"
    else:
        role = "system"
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    return Message(role=role, content=content)


class SessionStore:

    def __init__(self, max_messages: int = 20, ttl_seconds: int = 60 * 60 * 2):
        self._histories: dict[str, InMemoryChatMessageHistory] = {}
        self._last_seen: dict[str, float] = {}
        self._max_messages = max_messages  # counts individual messages, not turns
        self._ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()

    def _new_history(self) -> InMemoryChatMessageHistory:
        return InMemoryChatMessageHistory()

    async def get_history(self, session_id: str) -> list[Message]:
        """Full stored history for a session (subject to max_messages cap)."""
        return await self.get_last_k(session_id, k=None)

    async def get_last_k(self, session_id: str, k: Optional[int] = None) -> list[Message]:
        """Return the last k *turns* (k user+assistant pairs = 2k messages).
        k=None returns everything currently stored for the session."""
        async with self._lock:
            self._evict_expired()
            history = self._histories.get(session_id)
            if history is None:
                return []

            messages = history.messages
            if k is not None:
                messages = messages[-(k * 2):]

            self._last_seen[session_id] = time.monotonic()
            return [_from_lc_message(m) for m in messages]

    async def append_turn(
        self,
        session_id: str,
        user_message: Message,
        assistant_message: Message,
    ) -> None:
        async with self._lock:
            history = self._histories.setdefault(session_id, self._new_history())
            history.add_message(_to_lc_message(user_message))
            history.add_message(_to_lc_message(assistant_message))

            if len(history.messages) > self._max_messages:
                trimmed = history.messages[-self._max_messages:]
                history.clear()
                for m in trimmed:
                    history.add_message(m)

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
            self._histories.pop(sid, None)
            self._last_seen.pop(sid, None)