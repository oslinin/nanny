"""The InsightsAgent: an evidence-grounded research concierge.

A fourth real ``google.adk.agents.LlmAgent`` (alongside the Classifier and
Responder in ``nanny/agents.py``) that consumes a summary of the baby's
activity log and answers the parent's question — or, when there's no question,
proactively surfaces the most useful observation — grounded in reputable
guidance rather than opinion.

Three retrieval sources, layered so the agent is always useful and gets richer
as more is configured (the same opt-in philosophy as ``NANNY_API_TOKEN`` etc.):

1. ``child-guidance`` skill (always on, offline) — curated, cited summaries of
   mainstream public-health guidance (see ``skills/child-guidance/``).
2. Consensus.app via MCP (opt-in, ``NANNY_CONSENSUS_MCP_URL``) — scientific
   consensus over the research literature.
3. Scoped web search via a Google Programmable Search Engine (opt-in,
   ``GOOGLE_CSE_ID`` + ``GOOGLE_CSE_API_KEY``) — pinned in the CSE console to
   reputable sites (cdc.gov, aap.org, healthychildren.org, who.int,
   unicef.org). The ADK built-in ``google_search`` is model-side grounding
   that can't be reliably domain-restricted, which is why this uses a CSE.

With none configured (local dev, tests, this sandbox), the agent still answers
from the log summary + the curated skill; with no API key at all it falls back
to a deterministic summary via ``_summarize_insights`` — exactly how the
Classifier/Responder degrade offline. All optional integrations import their
dependencies lazily inside helpers so this module imports cleanly with no
credentials and no extra packages installed.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from pathlib import Path

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.llm_response import LlmResponse
from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset
from google.genai import types

from .llm import _model_available, _summarize_insights
from .security import screen_text

logger = logging.getLogger("nanny.research")

_MODEL_NAME = os.environ.get("NANNY_GEMINI_MODEL", "gemini-flash-latest")
_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

# Reputable sites the scoped-search tool is meant to cover. The actual
# restriction lives in the Programmable Search Engine's own configuration; this
# list is documentation + a per-query hint.
_GUIDANCE_SITES = (
    "cdc.gov",
    "aap.org",
    "healthychildren.org",
    "who.int",
    "unicef.org",
)


def _text_response(text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)])
    )


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


def _insights_offline_fallback_callback(callback_context, llm_request):
    """Returns a deterministic, log-grounded summary instead of calling Gemini
    when no model backend (AI-Studio key or Vertex) is configured."""
    if _model_available():
        return None
    state = callback_context.state
    context = state.get("insights_context") or {}
    question = state.get("question") or ""
    state["used_llm_response"] = False
    return _text_response(_summarize_insights(context, question))


def _search_reputable_child_health(query: str) -> dict:
    """Search reputable child-health sources for `query` and return citable hits.

    Covers CDC, AAP/HealthyChildren, WHO, and UNICEF via a Google Programmable
    Search Engine. Returns up to five ``{title, link, snippet}`` results the
    agent can cite. Only attached to the agent when ``GOOGLE_CSE_ID`` and
    ``GOOGLE_CSE_API_KEY`` are set.
    """
    api_key = os.environ.get("GOOGLE_CSE_API_KEY")
    cse_id = os.environ.get("GOOGLE_CSE_ID")
    if not (api_key and cse_id):
        return {"results": [], "error": "scoped search is not configured"}
    params = urllib.parse.urlencode(
        {"key": api_key, "cx": cse_id, "q": query, "num": 5}
    )
    url = f"https://www.googleapis.com/customsearch/v1?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # pragma: no cover - network path, not run offline
        logger.warning("scoped search failed: %s", exc)
        return {"results": [], "error": str(exc)}
    results = [
        {
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        }
        for item in data.get("items", [])
    ]
    return {"results": results}


def _optional_research_tools() -> list:
    """Builds the opt-in research tools that are actually configured.

    Each is added only when its env vars are present and its dependencies
    import, so a missing integration silently degrades to "not available to the
    agent" rather than breaking construction — the whole module has to import
    and run with nothing configured (local dev, tests, this sandbox).
    """
    tools: list = []

    consensus_url = os.environ.get("NANNY_CONSENSUS_MCP_URL")
    if consensus_url:
        try:
            from google.adk.tools.mcp_tool import (
                McpToolset,
                StreamableHTTPConnectionParams,
            )

            headers = {}
            api_key = os.environ.get("NANNY_CONSENSUS_API_KEY")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            tools.append(
                McpToolset(
                    connection_params=StreamableHTTPConnectionParams(
                        url=consensus_url, headers=headers or None
                    )
                )
            )
        except Exception as exc:  # pragma: no cover - depends on optional extra
            logger.warning("Consensus MCP not available, skipping: %s", exc)

    if os.environ.get("GOOGLE_CSE_ID") and os.environ.get("GOOGLE_CSE_API_KEY"):
        tools.append(_search_reputable_child_health)

    return tools


_INSIGHTS_INSTRUCTION = """\
You are Nanny's insights assistant. A parent logs their baby's feeds, diaper
changes, and solids. Using ONLY the summary of their own logged data below,
plus whatever evidence tools you have, give a brief, grounded, evidence-based
response.

Activity summary (JSON): {insights_context_json}

Parent's question (an empty value means: proactively surface the one or two
most useful observations or gentle questions from the data): {question}

Rules:
- Tie every point to the specific numbers in the summary above; don't speak in
  generalities disconnected from what was logged.
- Consult the 'child-guidance' skill for mainstream norms before relying on
  general knowledge, and cite the source briefly when you use one (e.g. "per
  CDC/AAP guidance"). If a research tool is available and relevant, use it and
  cite what it returns.
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
    skill = load_skill_from_dir(_SKILLS_DIR / "child-guidance")
    tools = [SkillToolset(skills=[skill]), *_optional_research_tools()]
    return LlmAgent(
        name="insights_agent",
        model=_MODEL_NAME,
        mode="single_turn",
        instruction=_INSIGHTS_INSTRUCTION,
        output_key="response_text",
        tools=tools,
        # Security guard runs first; only if it doesn't block does the offline
        # fallback get a chance to short-circuit the real model call.
        before_model_callback=[
            _insights_security_callback,
            _insights_offline_fallback_callback,
        ],
    )
