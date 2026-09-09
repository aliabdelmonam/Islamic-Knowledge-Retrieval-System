# ---------------------------------------------------------------------------
# Cohere (via LangChain)
# ---------------------------------------------------------------------------
from .llm_interface import GenerationClient, GenerationResponse, Message, ProviderError, Provider
from typing import Any, Optional
from app.core import settings
from pydantic import BaseModel

def _extract_text(content: Any) -> str:
    """AIMessage.content can be a plain string or a list of content
    blocks (e.g. [{"type": "text", "text": "..."}]) depending on the
    provider/model. Normalize to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)

class CohereClient(GenerationClient):
    """Wraps Cohere's chat API via LangChain's ChatCohere (so calls show up in LangSmith)."""

    def __init__(self, model: str = "command-r-plus", **kwargs):
        api_key = settings.COHERE_API_KEY
        if not api_key:
            raise ValueError("COHERE_API_KEY not set and no api_key provided")
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.api_key = api_key  # base class doesn't store this itself

        from langchain_cohere import ChatCohere  # lazy import
        self._ChatCohere = ChatCohere

    @property
    def provider_name(self) -> str:
        return Provider.COHERE.value

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        output_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> GenerationResponse:
        try:
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

            lc_messages: list[Any] = []
            for m in messages:
                if m.role == "system":
                    lc_messages.append(SystemMessage(content=m.content))
                elif m.role == "user":
                    lc_messages.append(HumanMessage(content=m.content))
                else:
                    lc_messages.append(AIMessage(content=m.content))

            llm = self._ChatCohere(
                model=self.model,
                cohere_api_key=self.api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            raw_ai_message = None
            if output_schema is not None:
                structured_llm = llm.with_structured_output(output_schema, include_raw=True)
                result = await structured_llm.ainvoke(lc_messages)
                raw_ai_message = result["raw"]
                parsed = result["parsed"]
                text = parsed.model_dump_json() if isinstance(parsed, BaseModel) else str(parsed)
            else:
                raw_ai_message = await llm.ainvoke(lc_messages)
                text = _extract_text(raw_ai_message.content)

            usage_meta = getattr(raw_ai_message, "usage_metadata", None) or {}
            usage = {
                "input_tokens": usage_meta.get("input_tokens"),
                "output_tokens": usage_meta.get("output_tokens"),
            } if usage_meta else {}

            response_metadata = getattr(raw_ai_message, "response_metadata", None) or {}
            finish_reason = response_metadata.get("finish_reason")

            return GenerationResponse(
                text=text,
                provider=self.provider_name,
                model=self.model,
                raw=raw_ai_message,
                usage=usage,
                finish_reason=finish_reason,
            )
        except Exception as e:
            raise ProviderError(self.provider_name, e) from e