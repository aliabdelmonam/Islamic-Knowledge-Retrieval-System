# ---------------------------------------------------------------------------
# Google Gemini (via LangChain)
# ---------------------------------------------------------------------------
from .llm_interface import GenerationClient, GenerationResponse, Message, ProviderError, Provider
from typing import Any
from app.core import settings
from pydantic import BaseModel


class GeminiClient(GenerationClient):
    """Wraps Google Gemini via LangChain's ChatGoogleGenerativeAI (so calls show up in LangSmith)."""

    def __init__(self, model: str = "gemini-2.5-flash", **kwargs):
        api_key = settings.GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set and no api_key provided")
        super().__init__(model=model, api_key=api_key, **kwargs)

        from langchain_google_genai import ChatGoogleGenerativeAI  # lazy import

        self._ChatGoogleGenerativeAI = ChatGoogleGenerativeAI
        self._base_kwargs = kwargs  # extra ctor-time kwargs (unrelated to per-call generation_config)

    @property
    def provider_name(self) -> str:
        return Provider.GEMINI.value

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

            system_msgs = [m.content for m in messages if m.role == "system"]
            turn_msgs = [m for m in messages if m.role != "system"]

            lc_messages: list[Any] = []
            if system_msgs:
                lc_messages.append(SystemMessage(content="\n".join(system_msgs)))

            for m in turn_msgs:
                if m.role == "user":
                    lc_messages.append(HumanMessage(content=m.content))
                else:
                    lc_messages.append(AIMessage(content=m.content))

            llm = self._ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs,
            )

            raw_ai_message = None
            text: str

            if output_schema is not None:
                # include_raw=True gives us back both the parsed object and the
                # underlying AIMessage (for usage/finish_reason), same as before.
                structured_llm = llm.with_structured_output(output_schema, include_raw=True)
                result = await structured_llm.ainvoke(lc_messages)
                raw_ai_message = result["raw"]
                parsed = result["parsed"]
                text = parsed.model_dump_json() if isinstance(parsed, BaseModel) else str(parsed)
            else:
                raw_ai_message = await llm.ainvoke(lc_messages)
                text = raw_ai_message.content

            usage_meta = getattr(raw_ai_message, "usage_metadata", None) or {}
            usage = {
                "input_tokens": usage_meta.get("input_tokens"),
                "output_tokens": usage_meta.get("output_tokens"),
            } if usage_meta else {}

            finish_reason = None
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