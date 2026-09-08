"""
LLM query rewriting: converts Egyptian/colloquial Arabic to MSA.
Falls back to original query if rewriting fails.
"""
from __future__ import annotations

from typing import Any
import logging
import re
from app.providers import GenerationClient,Message,GenerationResponse

logger = logging.getLogger(__name__)

# _REWRITE_PROMPT = """\
# You are an expert Arabic linguist specializing in Islamic texts.
# Your task is to rewrite the given Arabic question into formal Modern Standard Arabic (MSA),
# preserving the original meaning accurately. Remove any colloquial expressions,
# slang, or dialect-specific phrases while maintaining the semantic content.

# Question: {question}

# Provide ONLY the rewritten question in MSA, nothing else.
# """

_REWRITE_PROMPT="""\
أنت أداة إعادة صياغة أسئلة ضمن نظام استرجاع معرفة إسلامي.
مهمتك الوحيدة هي تحويل آخر رسالة من المستخدم إلى سؤال مستقل الفهم (self-contained) بالاعتماد
على سياق المحادثة السابق، دون الإجابة عن السؤال نفسه.

القواعد:
- حلّ الإحالات الضمنية (هو، هي، ذلك، هذا الحكم، السؤال السابق...) بالرجوع إلى المحادثة السابقة.
- لا تُضِف معلومات أو افتراضات غير موجودة في المحادثة.
- إذا كانت رسالة المستخدم مستقلة الفهم أصلاً ولا تحتاج سياقًا، أعدها كما هي دون تغيير.
- أخرج السؤال المعاد صياغته فقط، ضمن الحقل المطلوب، دون أي شرح أو مقدمات.
"""
 

def rewrite_query(query: str, llm: GenerationClient, temperature: float = 0.2, **kwargs: Any) -> str:
    """
    Rewrite *query* from colloquial Arabic to MSA via Groq.
    Returns the original query on any error.
    """
    try:

        messages = [
            Message(role="user", content=_REWRITE_PROMPT.format(question=query))
        ]
        response = llm.generate(
            messages=messages,
            temperature=temperature,
            output_schema=None,
            **kwargs
        )

        rewritten = response.text.strip().strip('"').strip("'")

        if rewritten:
            logger.info("Query rewritten: %r → %r", query, rewritten)
            return rewritten
        else:
            logger.warning("Query rewriting returned empty, using original query.")
            return query

    except Exception as exc:
        logger.warning("Query rewriting failed (%s), using original query.", exc)
        return query

