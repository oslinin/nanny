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


_REQUIREMENTS = ["google-adk>=2.3.0"]
_EXTRA_PACKAGES = ["nanny", "web", "skills"]


def deploy(dry_run: bool = False) -> str:
    """Deploys this app to Vertex AI Agent Runtime.

    Run with: `uv run python -m nanny.agent_engine_app` (or `--dry-run`).

    Requires `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and
    `GOOGLE_CLOUD_STAGING_BUCKET` env vars plus authenticated `gcloud`/ADC
    credentials — a real deploy cannot run in this sandbox (no GCP credentials
    here). ``dry_run=True`` needs none of that: it just prints the
    ``agent_engines.create(...)`` plan and returns an empty string, so
    `scripts/deploy.sh` can preview the whole pipeline offline.

    Returns:
        The deployed resource name (e.g.
        "projects/.../locations/.../reasoningEngines/...") — set this as
        `NANNY_AGENT_ENGINE_RESOURCE_NAME` on the dashboard's Cloud Run
        service so it calls the deployed agent instead of erroring out (the
        dashboard has no in-process fallback for this path — see
        `nanny/server.py`). Empty string for a dry run.
    """
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east1")
    if dry_run:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "<GOOGLE_CLOUD_PROJECT unset>")
        bucket = os.environ.get(
            "GOOGLE_CLOUD_STAGING_BUCKET", "<GOOGLE_CLOUD_STAGING_BUCKET unset>"
        )
        print("[dry-run] would deploy the agent to Vertex AI Agent Runtime:")
        print(f"  project        = {project}")
        print(f"  location       = {location}")
        print(f"  staging_bucket = {bucket}")
        print(f"  requirements   = {_REQUIREMENTS}")
        print(f"  extra_packages = {_EXTRA_PACKAGES}")
        print("[dry-run] no Vertex resources created.")
        return ""

    import vertexai
    from vertexai import agent_engines

    vertexai.init(
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=location,
        staging_bucket=os.environ["GOOGLE_CLOUD_STAGING_BUCKET"],
    )
    remote_app = agent_engines.create(
        agent_engine=build_agent_engine_app(),
        requirements=_REQUIREMENTS,
        extra_packages=_EXTRA_PACKAGES,
    )
    print(f"Deployed: {remote_app.resource_name}")
    return remote_app.resource_name


if __name__ == "__main__":
    import sys

    deploy(dry_run="--dry-run" in sys.argv[1:])
