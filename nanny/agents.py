"""Real ADK multi-agent nodes: ClassifierAgent and ResponderAgent.

Both are genuine ``google.adk.agents.LlmAgent`` instances — not plain
``FunctionNode``s wrapping a raw model call. ``LlmAgent`` is itself a
``BaseNode`` subclass, so it drops directly into the workflow graph's edges
(see ``nanny/workflow.py``), making this a real ADK multi-agent system rather
than a single deterministic workflow with LLM calls stuffed inside it.

Offline fallback and the security guardrail are both implemented as
``before_model_callback``s: returning a synthetic ``LlmResponse`` from one of
these callbacks skips the real model call entirely while still flowing
through ADK's normal ``output_schema`` / ``output_key`` state-writing path
(``LlmAgent.__maybe_save_output_to_state``), so downstream nodes cannot tell
the difference between a live and a short-circuited response. The first
callback in the list that returns a response wins; later callbacks (and the
real model) are skipped.
"""

from __future__ import annotations

import json
import logging
import os
from enum import StrEnum
from pathlib import Path

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.llm_response import LlmResponse
from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset
from google.genai import types
from pydantic import BaseModel, Field

from .activity import KNOWN_ACTIVITY_TYPES, KNOWN_UNITS, ActivityError
from .llm import _extract_heuristic, _model_available, _synthesize_template
from .security import screen_text

logger = logging.getLogger("nanny.agents")

_MODEL_NAME = os.environ.get("NANNY_GEMINI_MODEL", "gemini-flash-latest")
_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

# Built from activity.py's tuples (single source of truth) rather than
# hardcoded, so the model's structured-output schema — and therefore its
# constrained decoding — always matches what BabyActivity.validate() accepts.
_ActivityTypeEnum = StrEnum("_ActivityTypeEnum", {v: v for v in KNOWN_ACTIVITY_TYPES})
_UnitEnum = StrEnum("_UnitEnum", {v: v for v in KNOWN_UNITS})


class _ExtractedActivity(BaseModel):
    """Constrained JSON schema for ClassifierAgent's structured output."""

    activity_type: _ActivityTypeEnum
    quantity: float = Field(description="Numeric amount; use 1 for simple counts.")
    unit: _UnitEnum
    timestamp: str = Field(
        description="Absolute ISO-8601 timestamp, resolved from any relative time phrase."
    )
    notes: str = Field(
        default="", description="Any extra descriptive detail, else empty string."
    )


def _sentinel_extraction_json(now_iso: str) -> str:
    """A schema-valid placeholder payload for blocked/unrecognized turns.

    Never actually saved — ``classifier_postprocess_node`` in workflow.py
    always checks the ``security_blocked`` / ``heuristic_error`` state flags
    first and routes to the error branch before this content would be used.
    """
    return json.dumps(
        {
            "activity_type": KNOWN_ACTIVITY_TYPES[0],
            "quantity": 0.0,
            "unit": KNOWN_UNITS[0],
            "timestamp": now_iso,
            "notes": "",
        }
    )


def _text_response(text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)])
    )


def _classifier_security_callback(callback_context, llm_request):
    """Screens chat text for prompt injection / secrets before the model call."""
    state = callback_context.state
    text = state.get("chat_text") or ""
    reason = screen_text(text)
    if reason is None:
        return None
    logger.warning("ClassifierAgent: blocked chat input: %s", reason)
    state["security_blocked"] = True
    state["error"] = reason
    state["used_llm_extraction"] = False
    now_iso = state.get("now_iso") or ""
    return _text_response(_sentinel_extraction_json(now_iso))


def _classifier_offline_fallback_callback(callback_context, llm_request):
    """Runs the heuristic extractor instead of calling Gemini when no model
    backend (AI-Studio key or Vertex) is configured."""
    if _model_available():
        return None
    state = callback_context.state
    text = state.get("chat_text") or ""
    now_iso = state.get("now_iso") or ""
    state["used_llm_extraction"] = False
    try:
        activity = _extract_heuristic(text, now_iso=now_iso)
    except ActivityError as exc:
        state["heuristic_error"] = str(exc)
        return _text_response(_sentinel_extraction_json(now_iso))
    return _text_response(json.dumps(activity.to_dict()))


def _responder_offline_fallback_callback(callback_context, llm_request):
    """Renders the template confirmation instead of calling Gemini when no
    model backend (AI-Studio key or Vertex) is configured."""
    if _model_available():
        return None
    state = callback_context.state
    save_result = state.get("save_result") or {}
    state["used_llm_response"] = False
    return _text_response(_synthesize_template(save_result))


_CLASSIFIER_INSTRUCTION = """\
You extract structured infant-care activity records from free text for a
baby activity tracker. Always resolve relative or partial time phrases
("3 PM", "just now", "an hour ago") into an absolute ISO-8601 timestamp using
the current time given below as reference. activity_type must be exactly one
of: {activity_types}. unit must be exactly one of: {units} (oz for
milk/bottle volumes, grams for solids, count for diaper/poop events). Never
invent data that is not present or implied in the text.

Current time (ISO-8601): {{now_iso}}

Message: {{chat_text}}
""".format(activity_types=", ".join(KNOWN_ACTIVITY_TYPES), units=", ".join(KNOWN_UNITS))

_RESPONDER_INSTRUCTION = """\
You are Nanny, a friendly baby-tracking assistant. Write exactly one short,
warm sentence confirming what was logged and the running total for that
activity type today, based on this saved record and running totals (JSON):

{save_result_json}

If the 'care-tips' skill has a tip clearly relevant to this activity type,
you may load it and append one brief tip sentence — otherwise skip it. No
preamble, no more than two sentences total.
"""


def build_classifier_agent() -> LlmAgent:
    """The multi-agent system's entity-extraction agent (chat path only).

    Quick-tap payloads never reach this agent — ``ingest_node`` in
    workflow.py bypasses it entirely for that path.
    """
    return LlmAgent(
        name="classifier_agent",
        model=_MODEL_NAME,
        mode="single_turn",
        instruction=_CLASSIFIER_INSTRUCTION,
        output_schema=_ExtractedActivity,
        output_key="extracted_activity",
        # Security guard runs first; only if it doesn't block does the
        # offline fallback get a chance to short-circuit the real call.
        before_model_callback=[
            _classifier_security_callback,
            _classifier_offline_fallback_callback,
        ],
    )


def build_responder_agent() -> LlmAgent:
    """The multi-agent system's natural-language summary agent."""
    skill = load_skill_from_dir(_SKILLS_DIR / "care-tips")
    return LlmAgent(
        name="responder_agent",
        model=_MODEL_NAME,
        mode="single_turn",
        instruction=_RESPONDER_INSTRUCTION,
        output_key="response_text",
        tools=[SkillToolset(skills=[skill])],
        before_model_callback=[_responder_offline_fallback_callback],
    )
