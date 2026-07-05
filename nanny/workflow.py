"""The Nanny orchestration graph, built on the real Google ADK 2.0 workflow
engine (``google.adk.workflow``).

Mirrors the PRD's Deterministic-First Router-Dispatcher DAG:

    START -> ClassifierNode -> RouterNode -> SaveActivityNode -> ResponderNode
                  \\_____________________________/
                              (error branch, either node) -> ErrorNode

Every node reads and writes the shared ``ctx.state`` (the real ADK session
state, backed by the session service) rather than passing ad hoc arguments,
exactly as the PRD's shared data schema section describes.
"""

from __future__ import annotations

import logging

from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.workflow import START, Edge, Workflow, node

from .activity import ActivityError, BabyActivity
from .llm import extract_activity, synthesize_response
from .store import Store

logger = logging.getLogger("nanny.workflow")


def build_app(store: Store) -> App:
    """Constructs the ADK App wrapping the Nanny workflow graph.

    A single ``Store`` instance (the deterministic datastore) is closed over
    by ``save_activity_node`` — this is the one node in the graph forbidden
    from touching an LLM, matching the PRD's Step 4 requirement.
    """

    @node
    async def classifier_node(ctx: Context) -> None:
        """Step 2: Intent Classification & Entity Extraction.

        Quick-tap payloads are passed through unchanged (Conditional Skip).
        Chat text is routed to LLM extraction (or its offline fallback).
        """
        mode = ctx.state.get("input_mode")
        now_iso = ctx.state.get("now_iso")
        try:
            if mode == "quick_tap":
                payload = ctx.state.get("quick_tap_payload") or {}
                activity = BabyActivity.from_dict(payload)
                activity.validate()
                ctx.state["activity"] = activity.to_dict()
                ctx.state["ingestion_branch"] = "bypass"
                ctx.state["used_llm_extraction"] = False
                ctx.route = "bypass"
            elif mode == "chat":
                text = ctx.state.get("chat_text") or ""
                result = await extract_activity(text, now_iso=now_iso)
                result.activity.validate()
                ctx.state["activity"] = result.activity.to_dict()
                ctx.state["ingestion_branch"] = "extracted"
                ctx.state["used_llm_extraction"] = result.used_llm
                ctx.route = "extracted"
            else:
                raise ActivityError(f"unknown input_mode {mode!r}")
        except ActivityError as exc:
            logger.info("ClassifierNode: rejecting input: %s", exc)
            ctx.state["error"] = str(exc)
            ctx.state["last_status"] = "error"
            ctx.route = "error"

    @node
    async def router_node(ctx: Context) -> None:
        """Step 3: Conditional Routing.

        Deterministic bookkeeping only — declares which ingestion branch
        produced the record, then unconditionally forwards to storage. No
        generative side-effects happen here or after this point until the
        ResponderNode.
        """
        branch = ctx.state.get("ingestion_branch")
        logger.info("RouterNode: dispatching %s-branch record to storage", branch)

    @node
    async def save_activity_node(ctx: Context) -> None:
        """Step 4: Storage Execution — 100% deterministic, no LLM involved."""
        try:
            activity = BabyActivity.from_dict(ctx.state.get("activity") or {})
            result = store.append(activity)
            ctx.state["save_result"] = result.to_dict()
            ctx.state["last_status"] = "ok"
            ctx.route = "saved"
        except ActivityError as exc:
            logger.info("SaveActivityNode: rejecting record: %s", exc)
            ctx.state["error"] = str(exc)
            ctx.state["last_status"] = "error"
            ctx.route = "error"

    @node
    async def responder_node(ctx: Context) -> None:
        """Step 5: Conversational Synthesis — crafts the natural summary."""
        save_result = ctx.state.get("save_result")
        result = await synthesize_response(save_result)
        ctx.state["response_text"] = result.text
        ctx.state["used_llm_response"] = result.used_llm

    @node
    async def error_node(ctx: Context) -> None:
        """Terminal error branch: surfaces a friendly rejection message."""
        err = ctx.state.get("error", "unknown error")
        ctx.state["response_text"] = f"Sorry, I couldn't log that: {err}"
        ctx.state["used_llm_response"] = False

    workflow = Workflow(
        name="nanny_workflow",
        edges=[
            (START, classifier_node),
            Edge(
                from_node=classifier_node,
                to_node=router_node,
                route=["bypass", "extracted"],
            ),
            Edge(from_node=classifier_node, to_node=error_node, route="error"),
            (router_node, save_activity_node),
            Edge(from_node=save_activity_node, to_node=responder_node, route="saved"),
            Edge(from_node=save_activity_node, to_node=error_node, route="error"),
        ],
    )

    return App(root_agent=workflow, name="nanny_app")
