"""
Agentic RAG pipeline using LangGraph (v2).

Flow:
    retrieve → grade_documents ──┬──→ generate  (if enough good docs OR max loops)
                                 └──→ rewrite_query → retrieve → ...

Key features
------------
- **Hadith-level similarity**: retrieval uses embedding similarity on hadiths,
  not sharh-level cross-encoder reranking.
- **retrieval_history**: all hadiths retrieved across every loop are visible to the LLM.
- **query_history**: rewritten queries are tracked so the LLM avoids repeating them.
- **good_documents**: hadiths graded as relevant are accumulated across loops.
- **Hadith-only grading**: the grader sees only hadith text (no sharh).
  Sharh is included only in the final generation step.
- **Single LLM grading call**: the grade is stored in state and the edge reads it.
- **Decay**: fetch_k and k can decrease per iteration to narrow the search.
"""
from __future__ import annotations

import logging
from typing import List, Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from app.services.chain import ask, build_rag_chain, format_docs
from app.services.retriever import RetrievedResult, retrieve, retrieve_with_category_filter

logger = logging.getLogger(__name__)

try:
    from app.core.config import settings as _settings
    MAX_LOOPS = _settings.agentic_max_loops
except Exception:
    MAX_LOOPS = 2  # fallback

# ── Agent State ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """State passed between nodes in the LangGraph graph."""
    query: str                                  # Current (possibly rewritten) query
    original_query: str                         # Never mutated — original user question
    documents: List[RetrievedResult]            # Latest retrieved results (current loop)
    good_documents: List[RetrievedResult]       # Accumulated relevant hadiths across loops
    retrieval_history: List[str]                # Hadith texts from every loop (for LLM context)
    query_history: List[str]                    # All queries tried so far
    loop_count: int                             # Number of retrieval loops so far
    grade_decision: str                         # "relevant" | "not_relevant" — set by grade node
    answer: str                                 # Final generated answer
    current_k: int                              # Current top-k (may decay per loop)
    current_fetch_k: int                        # Current fetch_k (may decay per loop)


# ── Grading prompt (hadith-only, no sharh) ─────────────────────────────────────

_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "أنت مقيّم دقيق لمدى صلة الأحاديث النبوية بأسئلة المستخدم.\n\n"
            "ستُعرض عليك مجموعة من الأحاديث (النص فقط بدون الشرح).\n"
            "مهمتك: لكل حديث، قرّر هل هو ذو صلة بسؤال المستخدم أم لا.\n\n"
            "أجب بالصيغة التالية فقط — سطر لكل حديث:\n"
            "1: نعم\n"
            "2: لا\n"
            "...\n\n"
            "لا تضف أي شرح أو تعليق إضافي.",
        ),
        (
            "human",
            "السؤال: {question}\n\n"
            "الاستعلامات المجربة سابقاً: {query_history}\n\n"
            "الأحاديث:\n{hadiths}",
        ),
    ]
)

_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
أنت خبير في العلوم الإسلامية وبناء استعلامات البحث.

هدفك ليس إعادة صياغة السؤال فقط، بل إنشاء استعلام جديد يزيد احتمال العثور
على الأحاديث المناسبة.

اتبع الخطوات التالية داخلياً:

1. استخرج المفهوم الإسلامي الأساسي في السؤال.
2. ارجع خطوة للخلف (Step Back) إلى المفهوم أو الباب الإسلامي الأشمل.
   أمثلة:
   - بر الوالدين ← الأخلاق ← حقوق الوالدين
   - الغضب ← الأخلاق ← كظم الغيظ
   - الرزق ← التوكل، القناعة، البركة، الصدقة
   - الصلاة في السفر ← أحكام السفر ← الرخص
3. أضف المفاهيم الإسلامية المرتبطة التي قد ترد في الأحاديث.
4. استخدم مصطلحات شرعية وألفاظاً حديثية ومرادفات معروفة.
5. إذا كان السؤال يعتمد على قصة أو حكم أو فضيلة، فابحث أيضاً بالمفهوم العام
   وليس بالألفاظ الحرفية.
6. لا تغيّر نية المستخدم أو موضوع السؤال.
7. لا تكرر الاستعلامات السابقة إذا كانت متشابهة.

يمكنك الاستفادة من:
- أسماء الأبواب الفقهية.
- أبواب العقيدة.
- أبواب الآداب والأخلاق.
- أسماء العبادات.
- ألفاظ الأحاديث المشهورة.
- المصطلحات الشرعية المرتبطة.

الاستعلامات التي جُرّبت سابقاً:
{tried_queries}

الأحاديث التي سبق استرجاعها:
{retrieval_history}

أخرج استعلاماً واحداً فقط يصلح للبحث، بدون أي شرح.
""",
        ),
        (
            "human",
            """
السؤال الأصلي:
{original_question}

الاستعلام الحالي:
{current_query}
""",
        ),
    ]
)

_HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
أنت خبير في العلوم الإسلامية وحفظ الأحاديث النبوية.
مهمتك هي اقتراح نصوص أو أجزاء من أحاديث نبوية (candidate_hadiths) تعتقد أنها تجيب عن سؤال المستخدم وتطابق سياقه.

أخرج الناتج بصيغة JSON فقط كالتالي، كقائمة من النصوص:
[
  "نص أو جزء من الحديث المقترح الأول",
  "نص أو جزء من الحديث المقترح الثاني"
]

قواعد هامة:
- لا تكتب أي نص أو تعليق خارج صيغة JSON.
- اقترح الأحاديث التي تشعر أنها مناسبة، لا يوجد حد معين للعدد، يمكنك اقتراح حديث واحد أو أكثر.
- إذا لم تكن هناك أحاديث معينة تقترحها، اجعل القائمة فارغة [].
- تأكد من صحة تنسيق JSON (يجب أن يكون مصفوفة نصوص).
""",
        ),
        (
            "human",
            """السؤال: {question}""",
        ),
    ]
)


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _format_hadiths_only(results: list[RetrievedResult]) -> str:
    """Format hadiths for grading — hadith text only, no sharh."""
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r.hadith}")
    return "\n".join(parts)


def _parse_grade_response(response: str, num_docs: int) -> list[bool]:
    """
    Parse the LLM grade response like '1: نعم\\n2: لا\\n...'
    Returns a list of booleans (True = relevant).
    Falls back to all-relevant if parsing fails.
    """
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    results: list[bool] = []
    for line in lines:
        results.append("نعم" in line)

    # If parsing issue, pad with False
    while len(results) < num_docs:
        results.append(False)
    return results[:num_docs]


def _parse_hyde_response(response: str) -> list[str]:
    """
    Parse JSON response from the HyDE LLM.
    Returns list of candidate hadiths.
    """
    import json
    import re
    cleaned = response.strip()

    # Strip markdown code block markers if present
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)

    try:
        candidates = json.loads(cleaned)
        if not isinstance(candidates, list):
            candidates = []
        return [str(c).strip() for c in candidates if str(c).strip()]
    except Exception as e:
        logger.warning("Failed to parse JSON HyDE response: %s.", e)
        return []


# ── Node & graph builder ───────────────────────────────────────────────────────

def build_nodes(
    llm,
    vectorstore,
    bm25_index,
    all_chunks,
    parent_store,
    embedding_model,
    qdrant_client=None,
    embedding_model_name: str = "",
    category_collection_name: str = "hadith_categories",
    category_top_k: int = 5,
    k: int = 5,
    fetch_k: int = 25,
    k_decay: int = 0,
    fetch_k_decay: int = 0,
    system_role: str = "",
    hadith_bm25_index=None,
    hadith_records=None,
    hadith_search_top_k: int = 3,
):
    """
    Build and return the compiled LangGraph agent.

    Parameters
    ----------
    llm              : LangChain BaseChatModel for grading, rewriting, and generation.
    vectorstore      : Qdrant vector store.
    bm25_index       : BM25Okapi index.
    all_chunks       : List of all child Document chunks.
    parent_store     : List of parent Document objects.
    embedding_model  : SentenceTransformer for hadith-level similarity.
    qdrant_client    : QdrantClient for category retrieval.
    embedding_model_name : Model name for encoding category queries.
    category_collection_name : Qdrant collection for categories.
    category_top_k   : Number of categories to match per query.
    k                : Number of final results to retrieve per loop.
    fetch_k          : Broad fetch count for dense + BM25 search.
    k_decay          : Reduce k by this amount each iteration (min 1).
    fetch_k_decay    : Reduce fetch_k by this amount each iteration (min k).
    system_role      : System prompt for the final RAG generation chain.
    hadith_bm25_index : Pre-built BM25 index on clean hadiths.
    hadith_records   : List of HadithRecord.
    hadith_search_top_k : Number of hadiths to fetch per candidate.
    """

    from app.core.config import settings
    from app.services.llm import build_llm

    def get_node_llm(p: str, m: str):
        return build_llm(
            provider=p,
            sbg_model_id=m if p == "sbg" else settings.sbg_model_id,
            sbg_base_url=settings.sbg_base_url,
            sbg_api_key=settings.sbg_api_key or "",
            openai_model=m if p == "openai" else settings.openai_model,
            groq_model=m if p == "groq" else settings.groq_model,
            groq_api_key=settings.groq_api_key or "",
            ollama_model=m if p == "ollama" else settings.ollama_model,
            hf_model=m if p in ("huggingface", "huggingface_local") else settings.huggingface_model,
            hf_token=settings.hf_token or "",
            fanar_model=m if p == "fanar" else settings.fanar_model,
            fanar_api_key=settings.fanar_api_key or "",
            fanar_base_url=settings.fanar_base_url,
            gemini_model=m if p == "gemini" else settings.gemini_model,
            google_api_key=settings.google_api_key or "",
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    hyde_llm = get_node_llm("sbg", "qwen.qwen3-vl-235b-a22b").with_fallbacks([get_node_llm("gemini", "gemini-3.5-flash-lite")])
    grade_llm = get_node_llm("sbg", "openai.gpt-oss-120b-1:0").with_fallbacks([get_node_llm("gemini", "gemini-3.5-flash-lite")])
    rewrite_llm = get_node_llm("sbg", "qwen.qwen3-vl-235b-a22b").with_fallbacks([get_node_llm("gemini", "gemini-3.5-flash-lite")])
    generate_llm = get_node_llm("sbg", "qwen.qwen3-vl-235b-a22b").with_fallbacks([get_node_llm("gemini", "gemini-3.5-flash-lite")])

    rag_chain = build_rag_chain(generate_llm, system_role)
    grade_chain = _GRADE_PROMPT | grade_llm | StrOutputParser()
    rewrite_chain = _REWRITE_PROMPT | rewrite_llm | StrOutputParser()
    hyde_chain = _HYDE_PROMPT | hyde_llm | StrOutputParser()


    # ── Node: retrieve ──────────────────────────────────────────────────────

    def node_retrieve(state: AgentState) -> dict:
        query = state["query"]
        loop = state["loop_count"]
        cur_k = state["current_k"]
        cur_fetch_k = state["current_fetch_k"]
        logger.info(
            "[Agent] RETRIEVE — query=%r  loop=%d  k=%d  fetch_k=%d",
            query[:80], loop, cur_k, cur_fetch_k,
        )

        import concurrent.futures

        def _do_hybrid_search() -> list[RetrievedResult]:
            # Use category-filtered retrieval if category index is available
            if qdrant_client and embedding_model_name:
                return retrieve_with_category_filter(
                    query=query,
                    vectorstore=vectorstore,
                    bm25_index=bm25_index,
                    all_chunks=all_chunks,
                    parent_store=parent_store,
                    embedding_model=embedding_model,
                    qdrant_client=qdrant_client,
                    embedding_model_name=embedding_model_name,
                    category_collection_name=category_collection_name,
                    category_top_k=category_top_k,
                    k=cur_k,
                    fetch_k=cur_fetch_k,
                )
            else:
                return retrieve(
                    query=query,
                    vectorstore=vectorstore,
                    bm25_index=bm25_index,
                    all_chunks=all_chunks,
                    parent_store=parent_store,
                    embedding_model=embedding_model,
                    k=cur_k,
                    fetch_k=cur_fetch_k,
                )

        def _do_hyde_search() -> list[RetrievedResult]:
            if hadith_bm25_index is None or hadith_records is None:
                return []
            try:
                # LLM call for candidate hadiths
                response: str = hyde_chain.invoke({"question": state["query"]})
                logger.info("[Agent] HyDE raw response: %r", response[:200])
                candidates = _parse_hyde_response(response)
                logger.info("[Agent] Parsed HyDE — %d candidate hadiths", len(candidates))

                if not candidates:
                    return []

                from app.services.hadith_search import lookup_candidate_hadiths
                matched = lookup_candidate_hadiths(
                    index=hadith_bm25_index,
                    records=hadith_records,
                    candidate_queries=candidates,
                    k_per_query=hadith_search_top_k,
                )
                
                tool_found = []
                for rec in matched:
                    tool_found.append(
                        RetrievedResult(
                            hadith=rec.hadith,
                            sharh=rec.sharh,
                            rawy=rec.rawy,
                            source=rec.source,
                            hokm=rec.hokm,
                            page_id=rec.page_id,
                            chunk_text="",
                            similarity_score=1.0,
                        )
                    )
                logger.info("[Agent] Hadith HyDE lookup fetched %d hadiths", len(tool_found))
                return tool_found
            except Exception as e:
                logger.exception("Error in HyDE search thread: %s", e)
                return []

        # Run A1 (hybrid) and A2 (hyde) in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_hybrid = executor.submit(_do_hybrid_search)
            future_hyde = executor.submit(_do_hyde_search)
            docs = future_hybrid.result()
            hyde_docs = future_hyde.result()

        # Merge tool-found hadiths if present
        if hyde_docs:
            logger.info("[Agent] Merging %d HyDE-found hadiths into retrieved list", len(hyde_docs))
            seen = {d.hadith for d in docs}
            added = 0
            for td in hyde_docs:
                if td.hadith not in seen:
                    docs.append(td)
                    seen.add(td.hadith)
                    added += 1
            logger.info("[Agent] Added %d unique HyDE-found hadiths to retrieved list", added)

        # Append hadith texts to retrieval history
        new_history = list(state["retrieval_history"])
        for d in docs:
            new_history.append(d.hadith)

        # Add current query to query history (if not already there)
        q_history = list(state["query_history"])
        if query not in q_history:
            q_history.append(query)

        return {
            "documents": docs,
            "retrieval_history": new_history,
            "query_history": q_history,
        }

    # ── Node: grade_documents ───────────────────────────────────────────────

    def node_grade_documents(state: AgentState) -> dict:
        docs = state["documents"]
        if not docs:
            logger.info("[Agent] GRADE — no documents retrieved")
            return {"grade_decision": "not_relevant"}

        hadiths_text = _format_hadiths_only(docs)
        query_history_str = " | ".join(state["query_history"]) or "—"

        response: str = grade_chain.invoke({
            "question": state["original_query"],
            "query_history": query_history_str,
            "hadiths": hadiths_text,
        })
        logger.info("[Agent] GRADE raw response: %r", response[:200])

        grades = _parse_grade_response(response, len(docs))

        # Accumulate good docs
        new_good = list(state["good_documents"])
        seen_hadiths = {d.hadith for d in new_good}  # deduplicate
        newly_added = 0
        for doc, is_relevant in zip(docs, grades):
            if is_relevant and doc.hadith not in seen_hadiths:
                new_good.append(doc)
                seen_hadiths.add(doc.hadith)
                newly_added += 1

        good_count = len(new_good)
        cur_k = state["current_k"]
        logger.info(
            "[Agent] GRADE — %d/%d relevant this round, %d added, %d total good docs",
            sum(grades), len(docs), newly_added, good_count,
        )

        # Decide: enough good docs → relevant; else → not_relevant
        decision = "relevant" if good_count >= cur_k else "not_relevant"
        logger.info("[Agent] GRADE decision=%s (need %d, have %d)", decision, cur_k, good_count)

        return {
            "good_documents": new_good,
            "grade_decision": decision,
        }

    # ── Routing edge (reads state only — NO LLM call) ──────────────────────

    def edge_grade_decision(state: AgentState) -> Literal["generate", "rewrite_query"]:
        """Route based on grade_decision already stored in state."""
        if state["loop_count"] >= MAX_LOOPS:
            logger.info("[Agent] Max loops (%d) reached — forcing generation.", MAX_LOOPS)
            return "generate"

        if state["grade_decision"] == "relevant":
            return "generate"

        return "rewrite_query"

    # ── Node: rewrite_query ─────────────────────────────────────────────────

    def node_rewrite_query(state: AgentState) -> dict:
        tried = "\n".join(f"- {q}" for q in state["query_history"]) or "—"
        history_sample = state["retrieval_history"][-10:]  # last 10 hadiths
        history_str = "\n".join(f"- {h[:100]}" for h in history_sample) or "—"

        response: str = rewrite_chain.invoke({
            "original_question": state["original_query"],
            "current_query": state["query"],
            "tried_queries": tried,
            "retrieval_history": history_str,
        })
        new_query = response.strip().strip('"').strip("'")
        logger.info("[Agent] REWRITE — %r → %r", state["query"][:60], new_query[:80])

        # Apply decay for next iteration
        next_k = max(1, state["current_k"] - k_decay)
        next_fetch_k = max(next_k, state["current_fetch_k"] - fetch_k_decay)

        if k_decay or fetch_k_decay:
            logger.info(
                "[Agent] DECAY — k: %d→%d, fetch_k: %d→%d",
                state["current_k"], next_k,
                state["current_fetch_k"], next_fetch_k,
            )

        return {
            "query": new_query,
            "loop_count": state["loop_count"] + 1,
            "current_k": next_k,
            "current_fetch_k": next_fetch_k,
        }

    # ── Node: generate (uses good_documents with full sharh) ────────────────

    def node_generate(state: AgentState) -> dict:
        good_docs = state["good_documents"]
        # Restrict to refusing to answer if no good hadiths were accumulated
        if not good_docs:
            logger.info("[Agent] GENERATE — no good hadiths accumulated; refusing to answer")
            return {"answer": "لم أجد أحاديث تتوافق مع سؤالك."}

        logger.info(
            "[Agent] GENERATE — using %d good docs",
            len(good_docs),
        )

        answer = ask(
            question=state["original_query"],
            chain=rag_chain,
            results=good_docs,
        )
        return {"answer": answer}

    # ── Build graph ─────────────────────────────────────────────────────────

    graph = StateGraph(AgentState)

    graph.add_node("retrieve", node_retrieve)
    graph.add_node("grade_documents", node_grade_documents)
    graph.add_node("rewrite_query", node_rewrite_query)
    graph.add_node("generate", node_generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        edge_grade_decision,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


# ── Public helper ──────────────────────────────────────────────────────────────

def run_agentic_rag(
    query: str,
    agent_graph,
    k: int = 5,
    fetch_k: int = 25,
) -> dict:
    """
    Invoke the compiled LangGraph agent and return a rich result dict.

    Returns
    -------
    dict with keys:
         original_query  – the user's original question
         final_query     – the query used in the last retrieval round
         answer          – the generated answer string
         good_documents  – list of RetrievedResult graded as relevant
         all_documents   – list of RetrievedResult from the last loop
         loop_count      – how many retrieval loops were performed
         query_history   – all queries tried
    """
    initial_state: AgentState = {
        "query": query,
        "original_query": query,
        "documents": [],
        "good_documents": [],
        "retrieval_history": [],
        "query_history": [],
        "loop_count": 0,
        "grade_decision": "",
        "answer": "",
        "current_k": k,
        "current_fetch_k": fetch_k,
    }

    final_state: AgentState = agent_graph.invoke(initial_state)

    return {
        "original_query": final_state["original_query"],
        "final_query": final_state["query"],
        "answer": final_state["answer"],
        "good_documents": final_state["good_documents"],
        "all_documents": final_state["documents"],
        "loop_count": final_state["loop_count"],
        "query_history": final_state["query_history"],
    }
