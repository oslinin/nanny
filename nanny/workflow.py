"""The Nanny orchestration graph, built on the real Google ADK 2.0 workflow
engine (``google.adk.workflow``) and real ADK agents (``google.adk.agents``).

    START -> IngestNode --(bypass)--------------------------> RouterNode
                \\--(to_classify)--> ClassifierAgent (LLM) --> ClassifierPostProcessNode --(extracted)--> RouterNode
                \\--(error)-------------------------------------------------------------------------------------> ErrorNode
                \\--(get_history)-------------------------> HistoryNode
                \\--(insights)---> InsightsPrepNode -----> InsightsAgent (LLM)
                                                                              \\--(error)------------------------> ErrorNode
    RouterNode -> SaveActivityNode --(saved)--> ResponderAgent (LLM)
                                    \\--(error)----------------------------> ErrorNode

``InsightsAgent`` (see ``nanny/research.py``) is a fourth real ``LlmAgent``: it
reads a deterministic summary of the log (built by ``InsightsPrepNode``) and
answers the parent's question — or proactively surfaces an observation —
grounded in a curated ``child-guidance`` skill plus opt-in research tools
(Consensus MCP, a scoped guidance search).

``HistoryNode`` is a pure read: deployed on Vertex AI Agent Runtime, the
graph is only reachable through ``stream_query``/``async_stream_query`` (no
distinct REST surface), so reading a client's activity history has to flow
through the same query interface as everything else rather than a separate
endpoint with direct ``Store`` access from the dashboard/bridge layer.

``IngestNode``, ``RouterNode``, ``SaveActivityNode``, and ``ErrorNode`` are
plain deterministic ``FunctionNode``s. ``ClassifierAgent`` and
``ResponderAgent`` (see ``nanny/agents.py``) are real ``google.adk.agents.
LlmAgent`` instances wired directly into the graph — this is a genuine
multi-agent ADK system, not a single workflow with LLM calls stuffed inside
plain function nodes.

Every node reads and writes the shared ``ctx.state`` (the real ADK session
state, backed by the session service) rather than passing ad hoc arguments,
matching the PRD's shared ``BabyActivity`` schema.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.workflow import START, Edge, Workflow, node

from .activity import ActivityError, BabyActivity
from .agents import build_classifier_agent, build_responder_agent
from .llm import build_insights_context
from .research import build_insights_agent
from .store import Store

logger = logging.getLogger("nanny.workflow")

DEFAULT_CLIENT_ID = "default"


def build_app(store_resolver: Callable[[str], Store]) -> App:
    """Constructs the ADK App wrapping the Nanny multi-agent workflow graph.

    ``store_resolver`` maps a per-visitor client id (set into ``ctx.state``
    by the caller, e.g. ``server.py``, before each turn) to that visitor's
    own ``Store`` — each caller gets their own activity log, not one shared
    global log. Resolving the store dynamically per turn (rather than
    closing over one fixed instance) is what lets a single graph/agent set
    serve every client. ``save_activity_node`` is the one node in the graph
    forbidden from touching an LLM, matching the PRD's Step 4 requirement.
    """

    @node
    async def ingest_node(ctx: Context) -> None:
        """Step 1/2 entry: Payload Ingestion + Conditional Skip.

        Quick-tap payloads are validated and passed through unchanged,
        bypassing the LLM entirely. Chat text is routed to ClassifierAgent.
        """
        mode = ctx.state.get("input_mode")
        if mode == "quick_tap":
            try:
                payload = ctx.state.get("quick_tap_payload") or {}
                activity = BabyActivity.from_dict(payload).validate()
                ctx.state["activity"] = activity.to_dict()
                ctx.state["ingestion_branch"] = "bypass"
                ctx.state["used_llm_extraction"] = False
                ctx.route = "bypass"
            except ActivityError as exc:
                logger.info("IngestNode: rejecting quick-tap payload: %s", exc)
                ctx.state["error"] = str(exc)
                ctx.state["last_status"] = "error"
                ctx.route = "error"
        elif mode == "chat":
            # Optimistic default — ClassifierAgent's offline-fallback or
            # security callbacks overwrite this to False if either fires.
            ctx.state["used_llm_extraction"] = True
            ctx.route = "to_classify"
        elif mode == "get_history":
            ctx.route = "get_history"
        elif mode == "insights":
            ctx.route = "insights"
        else:
            ctx.state["error"] = f"unknown input_mode {mode!r}"
            ctx.state["last_status"] = "error"
            ctx.route = "error"

    @node
    async def history_node(ctx: Context) -> None:
        """Read-only: returns the resolved client's activity history.

        A dedicated node — rather than the dashboard/bridge touching
        ``Store`` directly — keeps 100% of Store access on the agent side of
        the Agent Runtime split.
        """
        client_id = ctx.state.get("client_id") or DEFAULT_CLIENT_ID
        store = store_resolver(client_id)
        ctx.state["history"] = [a.to_dict() for a in store.all()]
        ctx.state["last_status"] = "ok"

    @node
    async def insights_prep_node(ctx: Context) -> None:
        """Read-only: summarizes the client's log into state for InsightsAgent.

        Same agent-side-Store rationale as ``history_node`` — the reduction to
        a compact summary happens here (deterministic) so the agent reasons
        over a small structure, and the ``insights_context`` it writes is also
        what the offline fallback grounds a no-LLM reply in.
        """
        client_id = ctx.state.get("client_id") or DEFAULT_CLIENT_ID
        store = store_resolver(client_id)
        now_iso = ctx.state.get("now_iso") or ""
        activities = [a.to_dict() for a in store.all()]
        context = build_insights_context(activities, now_iso=now_iso)
        ctx.state["insights_context"] = context
        ctx.state["insights_context_json"] = json.dumps(context)
        ctx.state.setdefault("question", "")
        ctx.state["last_status"] = "ok"

    classifier_agent = build_classifier_agent()
    insights_agent = build_insights_agent()

    @node
    async def classifier_postprocess_node(ctx: Context) -> None:
        """Deterministic validation of ClassifierAgent's structured output.

        Keeps schema/security enforcement out of the LLM's hands: a node
        forbidden from touching an LLM is the only thing that can route a
        record onward to storage.
        """
        if ctx.state.get("security_blocked"):
            ctx.state["last_status"] = "error"
            ctx.route = "error"
            return
        if ctx.state.get("heuristic_error"):
            ctx.state["error"] = ctx.state["heuristic_error"]
            ctx.state["last_status"] = "error"
            ctx.route = "error"
            return
        try:
            extracted = ctx.state.get("extracted_activity") or {}
            activity = BabyActivity.from_dict(extracted).validate()
        except ActivityError as exc:
            logger.info("ClassifierPostProcessNode: rejecting record: %s", exc)
            ctx.state["error"] = str(exc)
            ctx.state["last_status"] = "error"
            ctx.route = "error"
            return
        ctx.state["activity"] = activity.to_dict()
        ctx.state["ingestion_branch"] = "extracted"
        ctx.route = "extracted"

    @node
    async def router_node(ctx: Context) -> None:
        """Step 3: Conditional Routing.

        Deterministic bookkeeping only — declares which ingestion branch
        produced the record, then unconditionally forwards to storage. No
        generative side-effects happen here or after this point until
        ResponderAgent.
        """
        branch = ctx.state.get("ingestion_branch")
        logger.info("RouterNode: dispatching %s-branch record to storage", branch)

    @node
    async def save_activity_node(ctx: Context) -> None:
        """Step 4: Storage Execution — 100% deterministic, no LLM involved."""
        try:
            client_id = ctx.state.get("client_id") or DEFAULT_CLIENT_ID
            store = store_resolver(client_id)
            activity = BabyActivity.from_dict(ctx.state.get("activity") or {})
            result = store.append(activity)
            ctx.state["save_result"] = result.to_dict()
            ctx.state["save_result_json"] = json.dumps(result.to_dict())
            ctx.state["last_status"] = "ok"
            # Optimistic default — ResponderAgent's offline-fallback callback
            # overwrites this to False if it fires.
            ctx.state["used_llm_response"] = True
            ctx.route = "saved"
        except ActivityError as exc:
            logger.info("SaveActivityNode: rejecting record: %s", exc)
            ctx.state["error"] = str(exc)
            ctx.state["last_status"] = "error"
            ctx.route = "error"

    responder_agent = build_responder_agent()

    @node
    async def error_node(ctx: Context) -> None:
        """Terminal error branch: surfaces a friendly rejection message."""
        err = ctx.state.get("error", "unknown error")
        ctx.state["response_text"] = f"Sorry, I couldn't log that: {err}"
        ctx.state["used_llm_response"] = False

    workflow = Workflow(
        name="nanny_workflow",
        edges=[
            (START, ingest_node),
            Edge(from_node=ingest_node, to_node=router_node, route="bypass"),
            Edge(from_node=ingest_node, to_node=classifier_agent, route="to_classify"),
            Edge(from_node=ingest_node, to_node=history_node, route="get_history"),
            Edge(
                from_node=ingest_node,
                to_node=insights_prep_node,
                route="insights",
            ),
            (insights_prep_node, insights_agent),
            Edge(from_node=ingest_node, to_node=error_node, route="error"),
            (classifier_agent, classifier_postprocess_node),
            Edge(
                from_node=classifier_postprocess_node,
                to_node=router_node,
                route="extracted",
            ),
            Edge(
                from_node=classifier_postprocess_node,
                to_node=error_node,
                route="error",
            ),
            (router_node, save_activity_node),
            Edge(from_node=save_activity_node, to_node=responder_agent, route="saved"),
            Edge(from_node=save_activity_node, to_node=error_node, route="error"),
        ],
    )

    return App(root_agent=workflow, name="nanny_app")
