"""The InsightsAgent: an evidence-grounded research concierge.

A fourth real ``google.adk.agents.LlmAgent`` (alongside the Classifier and
Responder in ``nanny/agents.py``) that consumes a summary of the baby's
activity log and answers the parent's question — or, when there's no question,
proactively surfaces the most useful observation — grounded in reputable
guidance rather than opinion.

The agent always reasons over a deterministic summary of the client's own log
plus the baby's profile (age/measurements); on top of that it draws on opt-in
retrieval tools (the same opt-in philosophy as ``NANNY_API_TOKEN``):

1. ADK's built-in ``google_search`` grounding tool — general web search,
   piggybacking on whatever Gemini/Vertex access the agent already has, so no
   separate API key or search-engine setup to configure or misconfigure.
   Trade-off: it's model-side grounding, so results can't be restricted to a
   fixed list of domains the way a scoped Custom Search Engine could be.
   Toggleable per parent from the Corpus tab.
2. The parent's reference documents — the shared UNICEF guide plus their own
   uploaded files, retrieved from the reference corpus (``nanny/corpus.py``):
   Google's managed Gemini File Search when a key allows it, else a local BM25
   index — each toggleable in the Corpus tab.

With none configured (local dev, tests, this sandbox), the agent still answers
from the log summary + baby profile; with no API key at all it falls back
to a deterministic summary via ``_summarize_insights`` — exactly how the
Classifier/Responder degrade offline. All optional integrations import their
dependencies lazily inside helpers so this module imports cleanly with no
credentials and no extra packages installed.
"""

from __future__ import annotations

import logging
import os

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.retrieval.base_retrieval_tool import BaseRetrievalTool
from google.genai import types

from . import corpus
from .llm import _model_available, _summarize_insights
from .security import screen_text

logger = logging.getLogger("nanny.research")

_MODEL_NAME = os.environ.get("NANNY_GEMINI_MODEL", "gemini-flash-latest")


def _text_response(text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)])
    )


# ADK's built-in google_search is attached with bypass_multi_tools_limit=True
# (see _optional_research_tools), which wraps it into a sub-agent named this —
# that wrapping is what lets it coexist with other tools (e.g. the RAG
# retrieval tool) and gives it a normal name/declaration we can filter out
# below like any other tool.
_GOOGLE_SEARCH_TOOL_NAME = "google_search_agent"


def _filter_disabled_tools_callback(callback_context, llm_request):
    """Removes the Google Search tool from the model's tool list entirely for
    a turn where the parent has switched it off in the Corpus tab.

    Hard enforcement, per the Corpus-tab design: a tool absent from
    ``llm_request`` cannot be called by the model no matter what the prompt
    says, unlike a prompt instruction telling it not to use one.
    """
    enabled_sources = callback_context.state.get("enabled_sources") or {}
    if enabled_sources.get("google_search", True):
        return None
    name = _GOOGLE_SEARCH_TOOL_NAME
    llm_request.tools_dict.pop(name, None)
    for tool in llm_request.config.tools or []:
        if tool.function_declarations:
            tool.function_declarations = [
                d for d in tool.function_declarations if d.name != name
            ]
    return None


def _insights_security_callback(callback_context, llm_request):
    """Screens the parent's question for prompt injection / secrets before the
    model call — the same guard the ClassifierAgent applies to chat input.

    Proactive turns carry no user-supplied question, so there's nothing to
    screen and this is a no-op for them.
    """
    state = callback_context.state
    question = state.get("question") or ""
    if not question.strip():
        return None
    reason = screen_text(question)
    if reason is None:
        return None
    logger.warning("InsightsAgent: blocked question: %s", reason)
    state["security_blocked"] = True
    state["error"] = reason
    state["last_status"] = "error"
    state["used_llm_response"] = False
    return _text_response(f"Sorry, I can't help with that: {reason}")


def _insights_summary_response(state) -> LlmResponse:
    """Produces the deterministic, log-grounded summary as a synthetic
    LlmResponse. Shared by the offline-fallback (no key) and model-error (live
    call failed) callbacks so both degrade to the exact same summary."""
    context = state.get("insights_context") or {}
    question = state.get("question") or ""
    state["used_llm_response"] = False
    return _text_response(_summarize_insights(context, question))


def _insights_offline_fallback_callback(callback_context, llm_request):
    """Returns a deterministic, log-grounded summary instead of calling Gemini
    when no model backend (AI-Studio key or Vertex) is configured."""
    if _model_available():
        return None
    return _insights_summary_response(callback_context.state)


def _insights_model_error_callback(*, callback_context, llm_request, error):
    """Degrades to the offline summary when a configured model call fails at
    runtime (invalid key, quota exhausted, timeout) instead of aborting."""
    logger.warning(
        "InsightsAgent: model call failed (%s); falling back to offline summary",
        error,
    )
    return _insights_summary_response(callback_context.state)


def _optional_research_tools() -> list:
    """Builds the opt-in research tools that are actually configured.

    Each is added only when its dependencies are actually usable, so a
    missing integration silently degrades to "not available to the agent"
    rather than breaking construction — the whole module has to import and
    run with nothing configured (local dev, tests, this sandbox).
    """
    tools: list = []

    if _model_available():
        # bypass_multi_tools_limit=True: google_search can't be combined with
        # other tools (e.g. the RAG retrieval tool) unless wrapped into its own
        # sub-agent — this flag makes ADK do that wrapping, which also gives it
        # the stable name we filter on (see _GOOGLE_SEARCH_TOOL_NAME).
        tools.append(GoogleSearchTool(bypass_multi_tools_limit=True))

    # The reference corpus always resolves (File Search or local BM25), so the
    # retrieval tool is always available — it simply returns "no passages" until
    # the parent uploads one or the shared UNICEF guide is seeded.
    if corpus.rag_enabled():
        tools.append(_PerClientRagRetrieval())

    return tools


class _PerClientRagRetrieval(BaseRetrievalTool):
    """Retrieves from the shared UNICEF corpus and *this parent's own* corpus,
    subject to the parent's Corpus-tab source toggles.

    A custom ``BaseRetrievalTool`` reads the ``client_id`` from turn state and
    queries only that client's corpus (``nanny/corpus.py`` — Gemini File Search
    or local BM25) — the same per-visitor isolation as the activity log.

    Enforcement of each checkbox happens here, in tool code, before any
    passage reaches the model — not via a prompt instruction:
    - The shared UNICEF corpus is queried only when ``unicef`` is enabled.
    - Passages from the client's own corpus are dropped when their source
      file is disabled in ``uploads`` (default enabled if unlisted, so a
      freshly uploaded file is usable immediately).
    """

    def __init__(self) -> None:
        super().__init__(
            name="search_my_references",
            description=(
                "Search reference materials the parent has enabled — the shared "
                "UNICEF parenting guide and/or their own uploaded references — "
                "for passages relevant to the query. Cite that you drew on this "
                "material when you use it."
            ),
        )

    async def run_async(self, *, args: dict, tool_context) -> str:
        from . import sources as sources_mod
        from .workflow import DEFAULT_CLIENT_ID

        client_id = tool_context.state.get("client_id") or DEFAULT_CLIENT_ID
        enabled_sources = tool_context.state.get("enabled_sources")
        if enabled_sources is None:
            enabled_sources = sources_mod.get_prefs(client_id)
        query = args.get("query") or ""

        passages: list[str] = []

        if enabled_sources.get("unicef", True):
            shared_corpus = corpus.resolve_shared_unicef_corpus()
            passages.extend(
                text for text, _fn in corpus.retrieve(shared_corpus, query) if text
            )

        corpus_handle = corpus.resolve_corpus_name(client_id)
        if corpus_handle:
            upload_prefs = enabled_sources.get("uploads") or {}
            for text, filename in corpus.retrieve(corpus_handle, query):
                if text and upload_prefs.get(filename, True):
                    passages.append(text)

        if not passages:
            return "No relevant passages found in the parent's references."
        return "\n\n---\n\n".join(passages)


_INSIGHTS_INSTRUCTION = """\
You are Nanny's insights assistant. A parent logs their baby's feeds, diaper
changes, and solids. Using ONLY the summary of their own logged data below,
plus whatever evidence tools you have, give a brief, grounded, evidence-based
response.

Activity summary (JSON): {insights_context_json}

Parent's question (an empty value means: proactively surface the one or two
most useful observations or gentle questions from the data): {question}

Rules:
- The summary's 'baby' field gives the child's age and measurements. Weigh
  norms and expectations against that specific age (a 2-month-old and a
  10-month-old differ greatly), and factor in weight/height when relevant.
- Tie every point to the specific numbers in the summary above; don't speak in
  generalities disconnected from what was logged.
- Ground claims in mainstream public-health norms (e.g. CDC/AAP/WHO) and cite
  the source briefly when you use one (e.g. "per CDC/AAP guidance"). If a
  research tool is available and relevant, use it and cite what it returns.
- If a 'search_my_references' tool is available, prefer the parent's own
  uploaded references when the question may be covered by them, and say when an
  answer draws on their material.
- This is general information, NOT medical advice or diagnosis. Frame anything
  health-related as "a pattern worth discussing with your pediatrician," and
  never state or imply a diagnosis.
- If you don't have enough logged data or evidence to say something useful, say
  so plainly instead of inventing a claim.
- Keep it to 2-4 short sentences.
"""


def build_insights_agent() -> LlmAgent:
    """The research-concierge agent (insights path only).

    Terminal node in the graph: writes ``response_text`` and, like the
    Responder, degrades to a template/offline summary when no API key is set.
    """
    tools = _optional_research_tools()
    return LlmAgent(
        name="insights_agent",
        model=_MODEL_NAME,
        mode="single_turn",
        instruction=_INSIGHTS_INSTRUCTION,
        output_key="response_text",
        tools=tools,
        # Security guard runs first; only if it doesn't block does the offline
        # fallback get a chance to short-circuit the real model call. Tool
        # filtering runs regardless (harmless if the offline fallback ends up
        # short-circuiting the turn right after).
        before_model_callback=[
            _insights_security_callback,
            _filter_disabled_tools_callback,
            _insights_offline_fallback_callback,
        ],
        on_model_error_callback=_insights_model_error_callback,
    )
