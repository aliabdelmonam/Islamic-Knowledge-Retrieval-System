"""
LLM provider factory + custom ChatSBG implementation.
Supports: sbg, openai, groq, ollama, huggingface, huggingface_local, fanar
"""
from __future__ import annotations

import logging
from typing import Any, Iterator, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)


# ── ChatSBG ───────────────────────────────────────────────────────────────────

class ChatSBG(BaseChatModel):
    """Minimal LangChain wrapper around the SBG (ITI) REST API."""

    model_id: str = "openai.gpt-oss-20b-1:0"
    base_url: str = "http://apiaccess.iti.net.eg/api/v1"
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 512

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:  # type: ignore[override]
        return "sbg"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        import requests

        # Separate system message from the rest
        system_prompt: str = ""
        chat_messages: list[dict] = []
        for m in messages:
            if m.type == "system":
                system_prompt = str(m.content)
            else:
                role = "user" if m.type == "human" else m.type
                chat_messages.append({"role": role, "content": str(m.content)})

        payload: dict = {
            "model_id": self.model_id,
            "messages": chat_messages,
            # "max_tokens": self.max_tokens,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt
        logger.info("SBG  payload is %s", payload)
        resp = requests.post(
            f"{self.base_url}/student/chat",
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        # Flat response: { "output_text": "...", ... }
        logger.info("SBG  output is %s", data)
        text: str = data.get("output_text") or ""
        if not text:
            logger.warning("SBG returned empty output_text. Full response: %s", data)

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text))]
        )



# ── Factory ────────────────────────────────────────────────────────────────────

def build_llm(
    provider: str,
    *,
    sbg_model_id: str = "openai.gpt-oss-20b-1:0",
    sbg_base_url: str = "http://apiaccess.iti.net.eg/api/v1",
    sbg_api_key: str = "",
    openai_model: str = "gpt-4o-mini",
    groq_model: str = "llama-3.3-70b-versatile",
    groq_api_key: str = "",
    ollama_model: str = "llama3.2",
    hf_model: str = "silma-ai/SILMA-Kashif-2B-Instruct-v1.0",
    hf_token: str = "",
    fanar_model: str = "Fanar-C-2-27B",
    fanar_api_key: str = "",
    fanar_base_url: str = "https://api.fanar.qa/v1",
    temperature: float = 0.1,
    max_tokens: int = 512,
):
    """Return a configured LangChain BaseChatModel for the given provider."""
    logger.info("Building LLM: provider=%s", provider)

    if provider == "sbg":
        return ChatSBG(
            model_id=sbg_model_id,
            base_url=sbg_base_url,
            api_key=sbg_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=openai_model, temperature=temperature, max_tokens=max_tokens)

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=groq_model,
            api_key=groq_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=ollama_model, temperature=temperature)

    if provider in ("huggingface", "huggingface_local"):
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        endpoint = HuggingFaceEndpoint(
            repo_id=hf_model,
            huggingfacehub_api_token=hf_token,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
        return ChatHuggingFace(llm=endpoint)

    if provider == "fanar":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=fanar_model,
            api_key=fanar_api_key,
            base_url=fanar_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unknown LLM provider: {provider!r}")
