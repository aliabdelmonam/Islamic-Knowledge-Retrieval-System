from __future__ import annotations

import time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.providers import GenerationClient, Message, ProviderError
from app.agents.helper.triage_agent_system_prompt import TRIAGE_SYSTEM_PROMPT

from app.core import get_logger

logger = get_logger(__name__)
# ---------------------------------------------------------------------------
# Chitchat categories (kept from the customer-service version; canned
# responses are in Arabic since the pipeline serves Arabic questions)
# ---------------------------------------------------------------------------

class ChitchatType(str, Enum):
    GREETING = "greeting"
    FAREWELL = "farewell"
    THANKS = "thanks"
    SMALL_TALK = "small_talk"
    NONE = "none"  # no chitchat framing present


_CANNED_RESPONSES: dict[ChitchatType, str] = {
    ChitchatType.GREETING: "وعليكم السلام ورحمة الله، أهلاً بك! تفضل بسؤالك الشرعي.",
    ChitchatType.FAREWELL: "في أمان الله، لا تتردد في العودة بأي سؤال آخر.",
    ChitchatType.THANKS: "العفو، جزاك الله خيرًا. هل هناك ما يمكنني مساعدتك به أيضًا؟",
    ChitchatType.SMALL_TALK: "يسعدني الحديث، لكن تفضل إن كان لديك سؤال شرعي يمكنني مساعدتك فيه.",
}
_DEFAULT_CANNED_RESPONSE = "كيف يمكنني مساعدتك اليوم؟"


# ---------------------------------------------------------------------------
# Islamic knowledge domain categories
# ---------------------------------------------------------------------------
# Multi-label: a single question can legitimately touch more than one
# category at once (e.g. a fatwa question that cites a hadith, or a question
# asking for a Quranic ruling that also needs general fiqh context).

class IslamicCategory(str, Enum):
    GENERAL_QUESTION = "general_question"  # general fiqh/fatwa, not tied to a specific hadith or ayah
    HADITH = "hadith"                      # question is about, or requires, hadith retrieval
    QURAN = "quran"                        # question is about, or requires, Quranic verses/tafsir


CATEGORY_DESCRIPTIONS: dict[IslamicCategory, str] = {
    IslamicCategory.GENERAL_QUESTION: (
        "أسئلة الفتوى العامة أو الفقهية التي لا ترتبط بشكل مباشر بحديث أو آية بعينها."
    ),
    IslamicCategory.HADITH: (
        "أسئلة تتعلق بحديث نبوي محدد، أو تتطلب البحث في كتب الحديث للإجابة عليها "
        "(مثل: صحة حديث، شرح حديث، تخريج حديث)."
    ),
    IslamicCategory.QURAN: (
        "أسئلة تتعلق بآية أو سورة قرآنية محددة، أو تتطلب تفسيرًا أو بحثًا في القرآن للإجابة عليها."
    ),
}


# ---------------------------------------------------------------------------
# Combined structured output
# ---------------------------------------------------------------------------

class TriageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    has_actionable_request: bool = Field(
        description="True if the message contains a real Islamic-knowledge question, "
                     "even alongside a greeting/thanks/farewell.",
    )
    chitchat_type: ChitchatType = Field(
        description="The greeting/farewell/thanks/small_talk framing present, if any. "
                     "'none' if has_actionable_request is true with no chitchat framing.",
    )

    categories: list[IslamicCategory] = Field(
        default_factory=list,
        description="All categories that apply to the question. Not mutually exclusive — "
                     "a question can be tagged with more than one category "
                     "(e.g. ['hadith', 'general_question']).",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1, max_length=500)
    needs_clarification: bool = Field(
        description="True when there's an actionable request but it's too ambiguous "
                     "to reliably route to a category.",
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> "TriageResult":
        if not self.has_actionable_request:
            if self.categories:
                logger.warning(
                    f"TriageResult inconsistency: categories={[c.value for c in self.categories]} "
                    f"present but has_actionable_request=False"
                )
                raise ValueError("categories must be empty when there is no actionable request")
        elif not self.needs_clarification:
            if not self.categories:
                logger.warning(
                    "TriageResult inconsistency: actionable, non-clarification result "
                    "has no categories"
                )
                raise ValueError(
                    "an actionable, non-clarification result requires at least one category"
                )
        if len(self.categories) != len(set(self.categories)):
            logger.warning(
                f"TriageResult inconsistency: duplicate categories={[c.value for c in self.categories]}"
            )
            raise ValueError("categories must not contain duplicates")
        return self

    def canned_response(self, *, active_ticket: bool = False, pending_question: Optional[str] = None) -> str:
        """Only meaningful when has_actionable_request is False."""
        base = _CANNED_RESPONSES.get(self.chitchat_type, _DEFAULT_CANNED_RESPONSE)
        logger.debug(
            f"Building canned response (chitchat_type={self.chitchat_type.value}, "
            f"active_ticket={active_ticket}, has_pending_question={bool(pending_question)})"
        )
        if active_ticket and pending_question:
            return f"{base} {pending_question}"
        return base


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TriageAgent:
    def __init__(
        self,
        llm: GenerationClient,
        *,
        temperature: float = 0.0,
        max_tokens: int = 250,
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(
            f"TriageAgent initialized (llm={type(llm).__name__}, "
            f"temperature={temperature}, max_tokens={max_tokens})"
        )

    @staticmethod
    def _messages(user_message: str, conversation_history: Optional[list[Message]]) -> list[Message]:
        messages = [Message(role="system", content=TRIAGE_SYSTEM_PROMPT)]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append(Message(role="user", content=user_message))
        logger.debug(
            f"Built triage message list (history_messages="
            f"{len(conversation_history) if conversation_history else 0}, "
            f"total_messages={len(messages)})"
        )
        return messages

    async def classify(
        self,
        user_message: str,
        conversation_history: Optional[list[Message]] = None,
    ) -> TriageResult:
        logger.info(
            f"Triage classify started (message_length={len(user_message) if user_message else 0}, "
            f"history_messages={len(conversation_history) if conversation_history else 0}, "
            f"preview={(user_message or '')[:120]!r})"
        )

        if not user_message or not user_message.strip():
            logger.warning(
                "Empty or whitespace-only message received; skipping LLM call "
                "and returning non-actionable result"
            )
            return TriageResult(
                has_actionable_request=False,
                chitchat_type=ChitchatType.NONE,
                categories=[],
                confidence=0.0,
                reasoning="Empty message.",
                needs_clarification=False,
            )

        logger.debug("Calling LLM for triage classification...")
        start = time.perf_counter()
        try:
            response = await self.llm.generate(
                messages=self._messages(user_message.strip(), conversation_history),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                output_schema=TriageResult,
            )
        except ProviderError:
            logger.error(
                f"Provider error during triage LLM call "
                f"(elapsed={time.perf_counter() - start:.2f}s)"
            )
            raise
        except Exception as exc:
            logger.exception(
                f"Unexpected error during triage LLM call "
                f"(elapsed={time.perf_counter() - start:.2f}s): {exc!r}"
            )
            raise RuntimeError("Triage generation failed") from exc

        elapsed = time.perf_counter() - start
        logger.debug(
            f"LLM responded (elapsed={elapsed:.2f}s, response_length={len(response.text)})"
        )
        logger.debug(f"Raw LLM triage response: {response.text!r}")

        try:
            result = TriageResult.model_validate_json(response.text)
        except ValidationError:
            logger.error(
                f"LLM response failed TriageResult schema validation "
                f"(elapsed={elapsed:.2f}s). Raw response: {response.text!r}"
            )
            raise

        logger.info(
            f"Triage completed (elapsed={elapsed:.2f}s) -> "
            f"actionable={result.has_actionable_request}, "
            f"chitchat_type={result.chitchat_type.value}, "
            f"categories={[c.value for c in result.categories]}, "
            f"confidence={result.confidence:.2f}, "
            f"needs_clarification={result.needs_clarification}"
        )
        logger.debug(f"Triage reasoning: {result.reasoning}")

        return result


__all__ = [
    "CATEGORY_DESCRIPTIONS",
    "ChitchatType",
    "IslamicCategory",
    "TriageResult",
    "TriageAgent",
]


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

async def _example():
    from app.providers import ProviderFactory, Provider

    llm = ProviderFactory.create(Provider.GEMINI, model="gemini-3.1-flash-lite")
    agent = TriageAgent(llm=llm)

    examples = [
        "السلام عليكم",
        "ما حكم الجمع بين الصلاتين في السفر؟",
        "ما صحة حديث: إنما الأعمال بالنيات؟",
        "ما تفسير آية الكرسي؟",
        "هل حديث كذا يفسر معنى آية كذا؟",  # multi-label: hadith + quran
        "شكرًا جزيلاً",
    ]

    for msg in examples:
        result = await agent.classify(msg)
        if not result.has_actionable_request:
            print(f"{msg!r} -> chitchat ({result.chitchat_type.value}): {result.canned_response()!r}")
        else:
            print(f"{msg!r} -> categories={[c.value for c in result.categories]} "
                  f"confidence={result.confidence} needs_clarification={result.needs_clarification}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_example())