"""
Fixed, non-LLM-generated fallback strings for the answer agent.

Kept as plain constants (not model output) so the refusal itself can never
hallucinate. INSUFFICIENT_EVIDENCE_MESSAGE is an interim placeholder for the
case where no retrieved evidence clears the sufficiency check — swap this out
once online/web search is wired in as an additional fallback tool.
"""

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "لا أملك حاليًا معلومات كافية وموثوقة من المصادر المتاحة للإجابة على هذا السؤال بدقة. "
    "يُرجى مراجعة مصادر موثوقة أخرى (مثل مواقع الإفتاء الرسمية أو كتب أهل العلم المعتمدة) "
    "للتأكد من الإجابة الصحيحة. سنعمل على تحسين قدرتنا على البحث عبر مصادر إضافية قريبًا."
)

CLARIFICATION_NEEDED_MESSAGE = (
    "سؤالك يحتاج إلى توضيح أكثر حتى أستطيع تحديد المصدر المناسب للإجابة — "
    "هل يمكنك تفصيل سؤالك أكثر؟"
)

__all__ = ["INSUFFICIENT_EVIDENCE_MESSAGE", "CLARIFICATION_NEEDED_MESSAGE"]