# 🕋 RAG-Hadith: Advanced Islamic Knowledge Retrieval System

RAG-Hadith is a production-grade, modular Retrieval-Augmented Generation (RAG) system specialized in Islamic Q&A across hadith, Quran, and general fiqh sources. It runs on a **LangChain-based multi-provider LLM layer** (Gemini, Cohere, Groq) with full **LangSmith tracing**, a **multi-turn conversational pipeline** with persistent session history, and a **multi-question pipeline** that can decompose, classify, retrieve, and answer several unrelated questions asked in a single message.

The system retrieves authenticated hadiths, Quranic verses with tafsir, and general fatwas; resolves colloquial Arabic queries into Modern Standard Arabic; classifies and routes questions by domain; and falls back to a whitelist-restricted web search when internal sources are insufficient — all while remaining strictly grounded in retrieved sources to avoid hallucinated rulings.

---

## 🚀 Key Features

### 1. Multi-Provider LLM Layer (LangChain-based)
* **Unified `GenerationClient` interface** over three providers — Google Gemini, Cohere, and Groq — each wrapped with its dedicated LangChain integration (`ChatGoogleGenerativeAI`, `ChatCohere`, `ChatGroq`) rather than raw provider SDKs.
* **Structured output** via `with_structured_output(schema, include_raw=True)` for every provider, used throughout the pipeline (triage, query rewriting, batch classification) instead of manual JSON parsing.
* **`ProviderFactory`** resolves a configured provider string (`"gemini"`, `"cohere"`, `"groq"`) to the matching LangChain-backed client at runtime.
* **LangSmith tracing** end-to-end: every `/chat` request is wrapped in a single `@traceable` root span, so query rewriting, triage, retrieval, and generation all nest under one request trace instead of appearing as disconnected runs. Enable via the standard `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT` environment variables.

### 2. Multi-Turn Conversation with Session History
* **`SessionStore`**: an async, LangChain-native (`InMemoryChatMessageHistory`-backed) conversation store keyed by `session_id`, with TTL-based eviction and a configurable message retention cap.
* **`get_last_k(session_id, k)`**: retrieves the last *k* conversational turns (user + assistant pairs) for use in prompting — decoupled from how much history is retained in storage vs. how much is actually fed to any given LLM call.
* Different pipeline stages request different history depths: query rewriting typically uses a short window (reference resolution only), while final generation uses a longer window (conversational continuity).
* The frontend persists `session_id` in `sessionStorage` across the browser session, round-tripping it with every `/chat` request so multi-turn context survives page reloads (but not tab closure).

### 3. Query Rewriting & Decomposition
* **Colloquial-to-MSA rewriting**: a single LLM call converts Egyptian/colloquial Arabic input into formal Modern Standard Arabic before it reaches retrieval or generation.
* **Multi-question decomposition**: the same call detects whether a user message actually contains multiple unrelated questions (e.g. two distinct fiqh topics bundled together) and splits them into standalone, self-contained MSA questions — biased conservatively toward *not* splitting unless topics are clearly distinct.
* **History-aware disambiguation**: resolves pronouns and implicit references (e.g. "the second hadith," "that ruling") using recent conversation history, without answering the question or injecting unrequested information.
* Falls back to the original, unmodified query on any error, ensuring rewriting/decomposition failures never block the pipeline.

### 4. Batched Triage & Non-Islamic Detection
* **Structured triage classification** determines, per question: whether it's an actionable Islamic-knowledge request, its domain categories (`general_question`, `hadith`, `quran` — multi-label), chitchat framing (greeting/thanks/farewell/small talk), and whether it needs clarification.
* **Non-Islamic detection**: a dedicated `is_non_islamic` flag identifies clear, understandable requests that are simply outside the system's domain (general trivia, coding help, sports, weather, etc.) — distinct from conversational chitchat — and routes them to an explicit redirect response rather than being misclassified or forced through retrieval.
* **`classify_batch`**: classifies multiple decomposed questions in a **single LLM call** for efficiency, with automatic fallback to sequential per-question classification if batch structured-output parsing or schema validation fails — reliability is prioritized over the latency savings of batching.

### 5. Parallel Multi-Source Retrieval
* **Category-routed retrievers**: `GeneralQuestionRetriever` (fatwa corpus), `QuranRetriever` (ayat + tafsir, Whoosh-backed lexical search), and `HadithRetriever` (hadith corpus, Whoosh-backed), each async and dispatched concurrently per question based on its triage categories.
* **`retrieve_all`**: runs retrieval for every decomposed question **in parallel** via `asyncio.gather`, skipping retrieval entirely for chitchat/non-Islamic/needs-clarification questions, with per-question error isolation so one question's retrieval failure doesn't cancel the others.
* Blocking lexical search calls (Whoosh) are offloaded via `asyncio.to_thread` so they don't stall sibling async retrieval tasks.

### 6. Grounded Answer Generation with Web Fallback
* **Two-stage generation** (single-question path): generate from internally retrieved documents first; if the LLM signals the sources are insufficient (a literal `"None"` sentinel per the system prompt), fall back to a whitelist-restricted web search (`SearchAgent`) and retry generation against those results. Returns a canned insufficient-evidence message only if both stages fail.
* **Pooled multi-question generation**: for decomposed multi-question messages, all questions and their independently retrieved documents are combined into a single labeled context and answered in **one LLM call**, producing a single coherent response — trading the per-question web-fallback precision of the single-question path for simplicity and lower latency.
* **Strict grounding + completeness instructions**: the generation system prompt enforces that answers rely only on supplied sources (no external/general knowledge, no fabrication), while also requiring synthesis across multiple relevant sources, inclusion of Quran/hadith/ijma evidence where present, and answers that default to thorough detail unless the user explicitly asks for brevity.

### 7. Defense-In-Depth Security
* **Obfuscation Parsing**: Normalizes zero-width spaces, diacritics (Tashkeel), baseline expansions, and collapses obfuscated character runs (e.g. `ت ج ا ه ل`).
* **Homoglyph Mapping**: Resolves lookalike Unicode characters (e.g., Cyrillic characters mimicking Latin letters) used to bypass prompt injection attempts.
* **Base64 Decoding**: Scans for Base64 payloads and decodes them for nested inspection.

---

## 📂 Project Structure

```directory
RAG-Hadith/
├── app/                              # FastAPI Application Core
│   ├── api/                          # API Routers & Endpoints
│   │   └── v1/
│   │       ├── endpoints/            # /ask, /chat, /retrieve, /health routers
│   │       └── router.py
│   ├── agents/                       # Core pipeline agents
│   │   ├── triage_agent.py           # Triage classification (single + batched), non-Islamic detection
│   │   ├── retrieval_agent.py        # Multi-source retrieval (single + parallel multi-question)
│   │   ├── search_agent.py           # Whitelist-restricted web search fallback
│   │   └── helper/
│   │       ├── answer_generation.py          # Two-stage generation + pooled multi-question generation
│   │       ├── retrieval_agent_system_prompt.py
│   │       ├── triage_agent_system_prompt.py
│   │       ├── general_knowledge_retrieval.py
│   │       ├── hadith_retrieval.py
│   │       └── quran_retreival.py
│   ├── core/                         # Settings, logging configuration, custom exceptions
│   │   ├── config.py                 # Pydantic BaseSettings config schema
│   │   └── logging.py
│   ├── providers/                    # LangChain-based LLM provider clients + factory
│   │   ├── llm_interface.py          # GenerationClient ABC, Message, GenerationResponse, Provider enum
│   │   ├── llm_factory.py            # ProviderFactory — resolves provider name -> client
│   │   ├── google_provider.py        # ChatGoogleGenerativeAI wrapper
│   │   ├── cohere_provider.py        # ChatCohere wrapper
│   │   ├── groq_provider.py          # ChatGroq wrapper
│   │   └── embeddings.py             # HuggingFaceEmbeddings-based embedding provider
│   ├── schemas/                      # Request and response models (Pydantic)
│   │   ├── request.py                # AskRequest, ChatRequest
│   │   └── response.py               # AskResponse, ChatResponse
│   └── services/
│       ├── session_store.py          # Multi-turn conversation history store (session_id-keyed)
│       └── query_rewriter.py         # MSA rewriting + multi-question decomposition
├── eval/                              # Evaluation & Benchmark Pipeline
│   ├── config.py                     # Evaluation settings
│   ├── evaluator.py                  # String overlap, embedding similarity, and LLM matching
│   └── run_eval.py                   # Evaluation runner over benchmark datasets
├── frontend/                          # Static Client SPA Dashboard
│   ├── index.html                    # Main Q&A dashboard
│   ├── app.js                        # Chat orchestration, session_id persistence
│   ├── settings.html                 # RAG parameter configuration
│   ├── duas.html                     # Supplications list
│   ├── ahadith.html                  # Hadith browsing interface
│   ├── mawaqit.html                  # Prayer timings dashboard
│   ├── favorites.html                # Saved resources page
│   └── style.css                     # Styling
├── scripts/                           # Maintenance & Indexing Scripts
│   ├── init_index.py                 # Embeds dataset chunks and indexes to Qdrant
│   ├── init_bm25.py                  # Generates lexical BM25 index on disk
│   ├── init_category_index.py        # Builds high-level category vector index
│   └── init_hadith_index.py          # Builds direct Hadith-level BM25 index
├── notebooks/                         # Experimental Jupyter Notebooks
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # API Container Configuration
└── docker-compose.yml                 # Container configuration for API, Qdrant, and Redis
```

---

## 🛠️ Configuration & Setup

### 1. Prerequisites
* Python 3.10 or 3.11
* Docker & Docker Compose (optional, for containerized database services)

### 2. Environment Setup
Clone the repository and copy the environment template:
```bash
cp .env.example .env
```
Fill out the keys in `.env`:
* **LLM Provider Settings**: Set `LLM_PROVIDER` to `gemini`, `groq`, or `cohere`, then choose the matching `*_MODEL` variable.
* **API Keys**: Enter the matching key: `GOOGLE_API_KEY`, `GROQ_API_KEY`, or `COHERE_API_KEY`.
* **Qdrant Settings**: Define your vector store connection details. If running Qdrant locally, the defaults (`localhost:6333`) are ready to go.
* **LangSmith (optional, recommended)**: Set `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT` to enable end-to-end request tracing across rewrite, triage, retrieval, and generation.
* **Session Store**: `SessionStore` is in-memory and process-local by default (`max_messages`, `ttl_seconds` configurable at construction). For multi-worker or multi-instance deployments, swap its internal `InMemoryChatMessageHistory` for a Redis-backed `BaseChatMessageHistory` implementation — the public interface (`get_last_k`, `append_turn`, `new_session_id`) does not need to change.

### 3. Initialize the Indexes
Before running the server, you must index your dataset. Run the scripts in the following order:

```bash
# 1. Chunk and upload the primary commentary dataset to Qdrant
python scripts/init_index.py

# 2. Build the primary commentary lexical (BM25) index
python scripts/init_bm25.py

# 3. Index high-level categories descriptions to Qdrant
python scripts/init_category_index.py

# 4. Build the direct Hadith-level BM25 search index
python scripts/init_hadith_index.py
```
*Note: Ensure the datasets (e.g. `Hadith_Filtered_Books.csv`, `main_categories_description.csv`) are present in the `data/` directory.*

### 4. Running the Server

#### Option A: Running Locally with Python
Activate your virtual environment, install dependencies, and run via `uvicorn`:
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

#### Option B: Running with Docker Compose
To build and spin up the FastAPI service, Qdrant vector database, and Redis cache in containers:
```bash
docker-compose up --build
```

---

## 🖥️ Frontend Dashboard
The user-facing side of the project is a sleek, static SPA located in the `/frontend` directory.
To run it, simply open `frontend/index.html` in your browser (or use a simple server like Live Server).

**Features**:
* **QA Dashboard**: Ask questions (including multiple questions in one message), toggle retrieve-only mode, and see answers alongside detailed source metadata.
* **Persistent multi-turn sessions**: `session_id` is stored in `sessionStorage` and round-tripped automatically with each `/chat` request, so conversation history survives page reloads within the same browser tab.
* **Settings Screen**: Modify LLM temperature, system prompts, retrieval sizes (`top_k`), and switch backend LLM providers in real-time.
* **Prayer Times, Favorites, and Duas**: Additional utility tabs for daily Islamic rituals.

---

## 🧪 Evaluation Pipeline
The codebase includes a specialized benchmarking pipeline under the `eval/` folder to run retrieval and generation accuracy tests.

To run the evaluation benchmarks:
```bash
python eval/run_eval.py
```

**Evaluation Strategies**:
1. **Substring match**: Simple normalized character containment.
2. **Sequence match**: Sequence-based token-level overlap.
3. **Cosine Embedding Similarity**: Query/Response semantic alignment.
4. **LLM Fallback**: If standard algorithms fail, an evaluator LLM checks semantic equivalency of the answer against the ground truth.

---

## 📡 API Endpoints

### `POST /api/v1/chat`
Multi-turn conversational endpoint. Supports single or multiple questions in one message, session-based history, and full LangSmith tracing under a single `chat_request` trace.

**Request Payload (`ChatRequest`)**:
```json
{
  "message": "ما حكم الربا؟ وهل يجوز أكل لحم الأرنب؟",
  "session_id": "optional-session-id",
  "top_k": 5
}
```

**Response Payload (`ChatResponse`)**:
```json
{
  "answer": "...(single coherent answer covering all sub-questions)...",
  "sources": [
    {
      "hadith": "إنما الأعمال بالنيات...",
      "rawy": "عمر بن الخطاب",
      "source": "صحيح البخاري",
      "hokm": "صحيح"
    }
  ],
  "categories": ["general_question"],
  "chitchat_type": "none",
  "needs_clarification": false,
  "session_id": "returned-or-newly-minted-session-id"
}
```

**Pipeline for `/chat`**:
1. **Rewrite + decompose** (`query_rewriter.rewrite_query`) — colloquial → MSA, splits multi-question messages, resolves references via short-window history.
2. **Batched triage** (`TriageAgent.classify_batch`) — one LLM call classifying all decomposed questions; detects non-Islamic and chitchat questions separately from actionable ones.
3. **Parallel retrieval** (`RetrievalAgent.retrieve_all`) — concurrent per-question retrieval across general/hadith/Quran sources, skipped for non-actionable questions.
4. **Pooled generation** (`generate_multi_answer` / `generate_answer` for the single-question case) — one coherent, strictly source-grounded answer; single-question messages retain the full two-stage web-search fallback.
5. **History persisted** — the original user message and final stitched answer are appended to the session as one turn.

### `POST /api/v1/ask`
Stateless single-turn endpoint (no session history). Retains the original two-stage generation with web-search fallback.

**Request Payload (`AskRequest`)**:
```json
{
  "question": "ما حكم العمل بالنيات؟",
  "top_k": 5
}
```

**Response Payload (`AskResponse`)**:
```json
{
  "answer": "إنما الأعمال بالنيات وإنما لكل امرئ ما نوى...",
  "sources": [
    {
      "hadith": "إنما الأعمال بالنيات...",
      "sharh": "هذا الحديث أصل عظيم من أصول الإسلام...",
      "rawy": "عمر بن الخطاب",
      "source": "صحيح البخاري",
      "hokm": "صحيح"
    }
  ],
  "categories": ["hadith"],
  "chitchat_type": "none",
  "needs_clarification": false,
  "session_id": "optional-session-id"
}
```

### `POST /api/v1/retrieve`
Retrieves matching documents (general/hadith/Quran) without generating an LLM response.

### `GET /api/v1/health`
Checks backend and service database readiness.

---

## 🔭 Known Limitations / Roadmap
* **Multi-question generation lacks per-question web fallback**: `generate_multi_answer` pools all sub-questions into one call, so if the model judges the overall response insufficient, individually well-supported sub-answers can be lost alongside genuinely unsupported ones. The single-question path does not have this limitation.
* **`ChatResponse` schema is single-question-shaped**: `categories`, `chitchat_type`, and `needs_clarification` are currently aggregated (union/OR) across decomposed sub-questions rather than reported per-question. A future revision may introduce a `list[SubAnswer]` structure for finer-grained multi-question responses.
* **`SessionStore` is in-memory and single-process** by default; swap to a Redis-backed `BaseChatMessageHistory` for multi-worker deployments.