# Backends, credentials & costs

How each Nanny feature is powered, **what credential it needs**, **where it
runs**, and **roughly what it costs**. Two independent axes matter:

1. **Which Google surface** a feature talks to — the **Gemini Developer API**
   (an API key, from [AI Studio](https://aistudio.google.com/apikey)) vs.
   **Vertex AI** (a GCP project + service-account/ADC auth).
2. **Where the process runs** — your machine (local), **Cloud Run** (the
   dashboard/API), **Vertex AI Agent Runtime** (the deployed graph), or
   **GitHub Pages** (the static frontend).

> ⚠️ **Cost figures are ballpark estimates (early 2026) for relative
> comparison, not billing.** Model/RAG/infra prices change often — verify
> against the official pricing pages linked at the bottom before you rely on a
> number. "≈" means order-of-magnitude.

---

## 1. Feature → needs → runs where → cost

| Feature | Credential it needs | Developer API key? | Vertex? | Where it runs | Cost model |
|---|---|---|---|---|---|
| **Offline heuristics** (quick-tap, log summary, deterministic fallbacks) | none | — | — | anywhere | **$0** |
| **LLM agents** (Classifier / Responder / Insights / Sitter) | `GEMINI_API_KEY` **or** `GOOGLE_GENAI_USE_VERTEXAI=true` + project | ✅ (one path) | ✅ (other path) | wherever the graph runs | per-token; **free tier** on Dev API |
| **Google Search grounding** (`google_search` tool) | rides the model backend — no extra key | ✅ or | ✅ | with the InsightsAgent | per grounded request |
| **Reference corpus — File Search** (`NANNY_CORPUS_BACKEND=file_search`/auto) | `GEMINI_API_KEY` | ✅ **required** | ❌ not on Vertex | server-side at Google | storage + embeddings (near-free small) |
| **Reference corpus — local BM25** (`…=local`, default fallback) | none | — | — | the server's own disk | **$0** (compute only) |
| **Reference corpus — old Vertex RAG** (removed) | GCP project + embeddings | ❌ | ✅ | Vertex | embeddings + RAG (**was the blocked/costly path**) |
| **Speech-to-text — browser** (default mic) | none | — | — | the browser | **$0** |
| **Speech-to-text — Cloud STT** (`NANNY_STT_ENABLED=true`) | GCP creds | ❌ | ✅ (GCP) | Cloud Run | per audio-minute |
| **Dashboard / API** | `GOOGLE_CLOUD_PROJECT` (+ token/CORS) | — | — | **Cloud Run** | requests + instance time |
| **Deployed graph** (`NANNY_AGENT_ENGINE_RESOURCE_NAME`) | GCP project + ADC | ❌ | ✅ | **Agent Runtime** | managed runtime + model calls |
| **Frontend** (`docs/`) | none | — | — | **GitHub Pages** | **$0** |

Nanny picks the model path automatically (`nanny/llm.py`): a Gemini key →
Developer API; `GOOGLE_GENAI_USE_VERTEXAI=true` + a project → Vertex; neither →
offline heuristics. The corpus path is `nanny/corpus.py`: File Search when a key
is present and reachable, else local BM25 (`NANNY_CORPUS_BACKEND` pins it).

---

## 2. Developer API key vs. Vertex — what each unlocks

| Capability | Gemini Developer API (`GEMINI_API_KEY`) | Vertex AI (`GOOGLE_GENAI_USE_VERTEXAI=true`) |
|---|---|---|
| Auth | one API key from AI Studio | GCP project + ADC / service account |
| Run the LLM agents | ✅ | ✅ |
| Google Search grounding | ✅ | ✅ |
| **Gemini File Search RAG** | ✅ **only here** | ❌ (Developer-API feature) |
| Vertex AI RAG Engine | ❌ | ✅ (not used by Nanny anymore) |
| Agent Runtime deploy | ❌ | ✅ **only here** |
| Cloud Speech-to-Text | ❌ | ✅ (GCP) |
| Free tier for the model | ✅ (rate-limited) | ❌ (billed from request 1) |
| Enterprise controls (VPC-SC, data residency, IAM) | limited | ✅ |

**Key takeaway:** File Search and Agent Runtime are on *opposite* surfaces. You
can run the model on Vertex **and** still use File Search — Nanny builds the File
Search client with `vertexai=False` + the key, so a Vertex-model deployment just
needs a `GEMINI_API_KEY` added for the corpus.

---

## 3. RAG options compared (incl. Python libraries)

What actually retrieves the parent's reference passages. Nanny ships the first
two; the rest are alternatives you could swap into `nanny/corpus.py`.

| Option | Type | Key needed | Runs where | Retrieval quality | Deps / weight | Per-use cost |
|---|---|---|---|---|---|---|
| **Gemini File Search** *(shipped)* | managed, semantic + cited | `GEMINI_API_KEY` | Google (server-side) | ★★★★ semantic, NotebookLM-style | none (SDK already present) | index embeddings once (**≈ free for small corpora**) + a small model call per query |
| **Local BM25 — `rank-bm25`** *(shipped fallback)* | lexical, in-process | none | your server/instance | ★★☆ keyword; good on topical Qs | tiny, pure-Python | **$0** |
| **Local embeddings — `sentence-transformers`** | semantic, in-process | none | your server/instance | ★★★★ semantic | heavy (`torch`, ~90 MB model download) | **$0** API; needs CPU/RAM |
| **`chromadb`** (bundled MiniLM) | local vector DB | none | your server/instance | ★★★★ semantic | heavy dep tree; downloads a model | **$0** API; local compute |
| **`FAISS` + your embeddings** | local vector index | none (or a key if embeddings are hosted) | your server/instance | ★★★★ (as good as the embeddings) | C++/Python lib | **$0** if embeddings are local |
| **Vertex AI RAG Engine** *(removed)* | managed, semantic | GCP project | Vertex | ★★★★ | Vertex SDK | embeddings + RAG + region; **the path that was blocked/costly** |
| **Hosted vector DB** (Pinecone, Weaviate Cloud…) | managed vector DB | provider key | that provider | ★★★★ | provider SDK | **$0 starter tier → ~$50+/mo** |

Rules of thumb:
- **Cheapest that "just works" anywhere:** local BM25 (shipped fallback). $0, no
  key, no downloads — but lexical, and its index lives on the host's disk (so on
  Cloud Run it's per-instance/ephemeral).
- **Best hosted experience, still cheap:** File Search. Managed + semantic +
  persistent server-side; embedding a whole parenting book is a **one-time
  cents-scale** charge, queries are a small model call.
- **Best fully-local semantic:** `sentence-transformers`/`chromadb` — $0 API but
  a fat dependency + model download and real CPU/RAM at query time.

---

## 4. Where things run — infra cost

| Surface | What runs there | Free tier | Rough cost |
|---|---|---|---|
| **Local** (`_LocalRunnerBackend`) | everything in-process | — | **$0** (your machine) |
| **GitHub Pages** | `docs/` static frontend | yes | **$0** |
| **Cloud Run** (dashboard/API) | FastAPI app; in-process graph unless Agent Runtime is set | 2M req/mo + generous CPU/mem monthly | with `--min-instances=1` (always warm): **≈ $5–25/mo** for a small instance + per-request compute |
| **Vertex AI Agent Runtime** | the deployed ADK graph | — | managed runtime (vCPU-hr + GiB-hr) **+** model calls — **higher than Cloud Run**, tens of $/mo floor |
| **Cloud SQL** (optional durable log) | Postgres for `nanny/store.py` | — | smallest instance **≈ $10–30/mo** |

Nanny's default deploy is **Cloud Run dashboard running the graph in-process**
(no Agent Runtime) — the cheapest cloud path. Agent Runtime is the
enterprise/managed option.

---

## 5. Per-use cost of the Google calls (ballpark)

| Call | Rough price | A single Nanny turn |
|---|---|---|
| **Gemini Flash** model tokens | ≈ $0.1–0.3 / 1M input, ≈ $0.4–2.5 / 1M output (Dev API **free tier** for low volume) | an insights answer ≈ 1–3k tokens → **fractions of a cent** |
| **Google Search grounding** | ≈ **$35 / 1,000** grounded prompts after a small free daily allowance | only when the InsightsAgent actually grounds a turn |
| **File Search — index** (embeddings at upload) | ≈ $0.15 / 1M tokens, **one-time per document** | a ~250-page book ≈ 150k tokens → **≈ $0.02 once** |
| **File Search — query** | query embeddings ≈ free; retrieval is one small `generate_content` | **fraction of a cent / query** |
| **Cloud Speech-to-Text** | ≈ $0.016–0.024 / minute (free 60 min/mo) | a spoken log is seconds → **~$0** (and the browser mic is free) |

**Worked example — a small hosted deployment** (Dev API key + File Search +
Cloud Run, ~a few hundred insights/day):
- Model + grounding: **cents/day** on paid tier, likely **$0** within the free
  tier at low volume.
- File Search: **cents one-time** to index the guide + parent uploads; queries
  negligible.
- Cloud Run always-warm instance: **≈ $5–25/mo** — usually the biggest line item.
- **Total: roughly the Cloud Run instance cost**, i.e. low tens of dollars a
  month, with the AI calls near-noise at this scale.

**Zero-cost path:** run locally (or Cloud Run scaled-to-zero) with
`NANNY_CORPUS_BACKEND=local`, browser mic, and either the Dev API free tier or
the fully-offline heuristics. **$0** beyond your own machine.

---

## 6. Recommended configurations

| Goal | Model | Corpus | Deploy | Notes |
|---|---|---|---|---|
| **Free local demo** | Dev API free tier *or* offline | `local` BM25 | local / Pages | no billing at all |
| **Cheap hosted (recommended)** | `GEMINI_API_KEY` | File Search (`auto`) | Cloud Run dashboard | set the key so uploads are managed + persistent; redeploy + re-seed after corpus changes |
| **Enterprise** | Vertex (`GOOGLE_GENAI_USE_VERTEXAI`) | File Search (add a `GEMINI_API_KEY`) *or* local | Agent Runtime + Cloud Run | IAM/VPC-SC controls; highest cost |

---

## Official pricing (verify these — numbers above are estimates)

- Gemini Developer API — <https://ai.google.dev/pricing>
- Vertex AI (models, grounding, RAG Engine) — <https://cloud.google.com/vertex-ai/generative-ai/pricing>
- Cloud Run — <https://cloud.google.com/run/pricing>
- Vertex AI Agent Engine — <https://cloud.google.com/vertex-ai/generative-ai/pricing>
- Cloud Speech-to-Text — <https://cloud.google.com/speech-to-text/pricing>
- Cloud SQL — <https://cloud.google.com/sql/pricing>
