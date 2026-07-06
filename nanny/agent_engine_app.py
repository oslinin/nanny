"""Wraps the Nanny workflow graph for deployment to Vertex AI Agent Runtime
(Agent Engine).

``vertexai.agent_engines.AdkApp`` accepts our exact ``google.adk.apps.App``
(from ``nanny.workflow.build_app``) directly — no changes needed to the
graph or agents themselves to deploy this way. Once actually deployed on
Agent Runtime, session state is automatically backed by
``VertexAiSessionService`` instead of living in memory — that's ``AdkApp``'s
own built-in behavior, not something this module has to configure.

This module is only usable with real GCP credentials (``AdkApp.set_up()``
unconditionally resolves a project via ``google.auth.default()`` even
without deploying anything, confirmed by trying it — a placeholder project
string is not enough). It cannot run in this sandbox, and the dashboard
(``nanny/server.py``) does not use it for local development either — local
dev keeps the direct ``Runner`` + ``InMemorySessionService`` path unchanged.
This module exists purely for the deploy step, run by a developer with
`gcloud`/ADC credentials on their own machine, and for the deployed
dashboard's remote calls once ``NANNY_AGENT_ENGINE_RESOURCE_NAME`` is set.
"""

from __future__ import annotations

import os

from .stores import get_store
from .workflow import build_app


def build_agent_engine_app():
    """Builds the AdkApp for deployment. Requires real GCP credentials —
    see module docstring."""
    import vertexai
    from vertexai.agent_engines import AdkApp

    vertexai.init(
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east1"),
    )
    return AdkApp(app=build_app(get_store))


def deploy() -> str:
    """Deploys this app to Vertex AI Agent Runtime.

    Run with: `uv run python -m nanny.agent_engine_app`

    Requires `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and
    `GOOGLE_CLOUD_STAGING_BUCKET` env vars plus authenticated `gcloud`/ADC
    credentials — this cannot run in this sandbox (no GCP credentials here).

    Returns:
        The deployed resource name (e.g.
        "projects/.../locations/.../reasoningEngines/...") — set this as
        `NANNY_AGENT_ENGINE_RESOURCE_NAME` on the dashboard's Cloud Run
        service so it calls the deployed agent instead of erroring out (the
        dashboard has no in-process fallback for this path — see
        `nanny/server.py`).
    """
    import vertexai
    from vertexai import agent_engines

    vertexai.init(
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east1"),
        staging_bucket=os.environ["GOOGLE_CLOUD_STAGING_BUCKET"],
    )
    remote_app = agent_engines.create(
        agent_engine=build_agent_engine_app(),
        requirements=["google-adk>=2.3.0"],
        extra_packages=["nanny", "web", "skills"],
    )
    print(f"Deployed: {remote_app.resource_name}")
    return remote_app.resource_name


if __name__ == "__main__":
    deploy()
