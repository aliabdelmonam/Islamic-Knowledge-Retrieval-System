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

# Shown when a question is understandable and actionable, but entirely
# outside the Islamic-knowledge domain this system serves (e.g. general
# trivia, coding help, weather, sports) — distinct from chitchat, which is
# conversational framing (greetings/thanks) rather than an off-topic request.
_NON_ISLAMIC_RESPONSE = (
    "أعتذر، أنا مساعد متخصص في الإجابة عن الأسئلة الشرعية والإسلامية فقط "
    "(مثل الفقه، الحديث، القرآن والتفسير). لا يمكنني مساعدتك في هذا السؤال، "
    "لكن يسعدني الإجابة عن أي سؤال شرعي لديك."
)


# ---------------------------------------------------------------------------
# Islamic knowledge domain categories
# ---------------------------------------------------------------------------

class IslamicCategory(str, Enum):
    GENERAL_QUESTION = "general_question"
    HADITH = "hadith"
    QURAN = "quran"


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
# Combined structured output — single-question result
# ---------------------------------------------------------------------------

class TriageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    has_actionable_request: bool = Field(
        description="True if the message contains a real Islamic-knowledge question, "
                     "even alongside a greeting/thanks/farewell. Must be False when "
                     "is_non_islamic is True.",
    )
    is_non_islamic: bool = Field(
        default=False,
        description="True if the message is a clear, understandable request/question "
                     "that has nothing to do with Islamic knowledge (e.g. general trivia, "
                     "coding help, weather, sports, other religions' unrelated topics). "
                     "This is distinct from chitchat_type — chitchat is conversational "
                     "framing (greeting/thanks/farewell), not an off-topic request.",
    )
    chitchat_type: ChitchatType = Field(
        description="The greeting/farewell/thanks/small_talk framing present, if any. "
                     "'none' if has_actionable_request is true with no chitchat framing, "
                     "or if is_non_islamic is true.",
    )
    categories: list[IslamicCategory] = Field(
        default_factory=list,
        description="All categories that apply to the question. Not mutually exclusive — "
                     "a question can be tagged with more than one category "
                     "(e.g. ['hadith', 'general_question']). Must be empty when "
                     "is_non_islamic is True.",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1, max_length=500)
    needs_clarification: bool = Field(
        description="True when there's an actionable Islamic-knowledge request but it's "
                     "too ambiguous to reliably route to a category. Must be False when "
                     "is_non_islamic is True.",
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> "TriageResult":
        if self.is_non_islamic:
            if self.has_actionable_request:
                logger.warning(
                    "TriageResult inconsistency: is_non_islamic=True but "
                    "has_actionable_request=True"
                )
                raise ValueError("has_actionable_request must be False when is_non_islamic is True")
            if self.categories:
                logger.warning(
                    f"TriageResult inconsistency: categories={[c.value for c in self.categories]} "
                    f"present but is_non_islamic=True"
                )
                raise ValueError("categories must be empty when is_non_islamic is True")
            if self.needs_clarification:
                logger.warning(
                    "TriageResult inconsistency: is_non_islamic=True but needs_clarification=True"
                )
                raise ValueError("needs_clarification must be False when is_non_islamic is True")
        elif not self.has_actionable_request:
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
        if self.is_non_islamic:
            logger.debug("Building non-Islamic redirect response")
            return _NON_ISLAMIC_RESPONSE
        base = _CANNED_RESPONSES.get(self.chitchat_type, _DEFAULT_CANNED_RESPONSE)
        logger.debug(
            f"Building canned response (chitchat_type={self.chitchat_type.value}, "
            f"active_ticket={active_ticket}, has_pending_question={bool(pending_question)})"
        )
        if active_ticket and pending_question:
            return f"{base} {pending_question}"
        return base


# ---------------------------------------------------------------------------
# Batch structured output — one call, many questions
# ---------------------------------------------------------------------------

class TriageBatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        description="The exact question text this classification result applies to, "
                     "copied verbatim from the numbered input list.",
    )
    has_actionable_request: bool
    is_non_islamic: bool = False
    chitchat_type: ChitchatType
    categories: list[IslamicCategory] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1, max_length=500)
    needs_clarification: bool

    @model_validator(mode="after")
    def validate_consistency(self) -> "TriageBatchItem":
        if self.is_non_islamic:
            if self.has_actionable_request:
                raise ValueError("has_actionable_request must be False when is_non_islamic is True")
            if self.categories:
                raise ValueError("categories must be empty when is_non_islamic is True")
            if self.needs_clarification:
                raise ValueError("needs_clarification must be False when is_non_islamic is True")
        elif not self.has_actionable_request:
            if self.categories:
                raise ValueError("categories must be empty when there is no actionable request")
        elif not self.needs_clarification:
            if not self.categories:
                raise ValueError(
                    "an actionable, non-clarification result requires at least one category"
                )
        if len(self.categories) != len(set(self.categories)):
            raise ValueError("categories must not contain duplicates")
        return self

    def to_triage_result(self) -> TriageResult:
        """Drop the `question` field to reuse everything built on TriageResult
        (canned_response, downstream type checks) unchanged."""
        return TriageResult(
            has_actionable_request=self.has_actionable_request,
            is_non_islamic=self.is_non_islamic,
            chitchat_type=self.chitchat_type,
            categories=self.categories,
            confidence=self.confidence,
            reasoning=self.reasoning,
            needs_clarification=self.needs_clarification,
        )


class BatchTriageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[TriageBatchItem] = Field(min_length=1)


_BATCH_TRIAGE_INSTRUCTIONS = """\
سيتم تزويدك بعدة أسئلة مرقمة. صنّف كل سؤال على حدة بشكل مستقل تمامًا عن
الأسئلة الأخرى، وأعد نتيجة تصنيف واحدة لكل سؤال، بنفس ترتيب الأسئلة أدناه.
انسخ نص كل سؤال حرفيًا في حقل "question" الخاص به حتى يمكن مطابقة النتائج
بالأسئلة الأصلية. لا تدمج أو تخلط بين الأسئلة عند التصنيف.

تذكّر: إذا كان أحد الأسئلة لا علاقة له إطلاقًا بالمجال الإسلامي (مثل أسئلة
عامة، برمجة، طقس، رياضة)، ضع is_non_islamic=true لذلك السؤال تحديدًا، بغض
النظر عن تصنيف باقي الأسئلة في نفس الدفعة.

الأسئلة:
{numbered_questions}
"""


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
                is_non_islamic=False,
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
            logger.debug("Raw batch response: %r", response)
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
            f"is_non_islamic={result.is_non_islamic}, "
            f"chitchat_type={result.chitchat_type.value}, "
            f"categories={[c.value for c in result.categories]}, "
            f"confidence={result.confidence:.2f}, "
            f"needs_clarification={result.needs_clarification}"
        )
        logger.debug(f"Triage reasoning: {result.reasoning}")

        return result

    async def classify_batch(
        self,
        questions: list[str],
        conversation_history: Optional[list[Message]] = None,
    ) -> list[TriageResult]:
        """
        Classify multiple independent questions in a single LLM call.
        Returns one TriageResult per input question, in the same order as
        *questions*. Falls back to sequential single-question `classify()`
        calls if the batch call fails validation or the model doesn't
        return a result for every question.
        """
        questions = [q.strip() for q in questions if q and q.strip()]

        if not questions:
            logger.warning("classify_batch called with no non-empty questions.")
            return []

        if len(questions) == 1:
            return [await self.classify(questions[0], conversation_history)]

        logger.info(f"Triage classify_batch started (num_questions={len(questions)})")

        numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
        batch_prompt = _BATCH_TRIAGE_INSTRUCTIONS.format(numbered_questions=numbered)

        start = time.perf_counter()
        try:
            response = await self.llm.generate(
                messages=self._messages(batch_prompt, conversation_history),
                temperature=self.temperature,
                max_tokens=self.max_tokens * len(questions),
                output_schema=BatchTriageResult,
            )
            # print("*"*50)
            # print(f"Batch response:\n{response}")
            batch_result = BatchTriageResult.model_validate_json(response.text)
            # print("*"*50)
            # print(f"Batch result:\n{batch_result}")
            if len(batch_result.results) != len(questions):
                raise ValueError(
                    f"Expected {len(questions)} results, got {len(batch_result.results)}"
                )

            # by_question = {item.question.strip(): item for item in batch_result.results}
            ordered: list[TriageResult] = []
            # if len(batch_result.results) != len(questions):
            #     raise ValueError(
            #         f"Expected {len(questions)} results, got {len(batch_result.results)}"
            #     )
            ordered = [item.to_triage_result() for item in batch_result.results]

            elapsed = time.perf_counter() - start
            logger.info(
                f"Triage classify_batch completed (elapsed={elapsed:.2f}s, "
                f"num_questions={len(questions)})"
            )
            return ordered

        except Exception as exc:
            logger.warning(
                f"Batch triage failed ({exc!r}), falling back to sequential "
                f"single-question classification for {len(questions)} questions."
            )
            return [
                await self.classify(q, conversation_history)
                for q in questions
            ]


__all__ = [
    "CATEGORY_DESCRIPTIONS",
    "ChitchatType",
    "IslamicCategory",
    "TriageResult",
    "TriageBatchItem",
    "BatchTriageResult",
    "TriageAgent",
]


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

async def _example():
    from app.providers import ProviderFactory, Provider

    llm = ProviderFactory.create(Provider.GEMINI, model="gemini-3.1-flash-lite")
    agent = TriageAgent(llm=llm)

    # examples = [
        # "من فاز بكأس العالم2026 ؟",       # non-Islamic
    # ]

    # for msg in examples:
        # result = await agent.classify(msg)
        # if result.is_non_islamic:
            # print(f"{msg!r} -> non-Islamic: {result.canned_response()!r}")
        # elif not result.has_actionable_request:
            # print(f"{msg!r} -> chitchat ({result.chitchat_type.value}): {result.canned_response()!r}")
        # else:
            # print(f"{msg!r} -> categories={[c.value for c in result.categories]} "
                #   f"confidence={result.confidence} needs_clarification={result.needs_clarification}")
    # print("="*100)
    batch_results = await agent.classify_batch([
        "ما حكم الربا؟",
        "هل يجوز أكل لحم الأرنب؟",
        "شكرًا جزيلاً",
        "من فاز بكأس العالم2026 ؟"
    ])
    for q, r in zip(
        ["ما حكم الربا؟", "هل يجوز أكل لحم الأرنب؟", "شكرًا جزيلاً", "من فاز بكأس العالم2026 ؟"],
        batch_results,
    ):
        if not r.has_actionable_request:
            print(f"{q!r} -> chitchat ({r.chitchat_type.value})")
        else:
            print(f"{q!r} -> categories={[c.value for c in r.categories]}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(_example())