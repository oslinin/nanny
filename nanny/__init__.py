"""Nanny — a local baby activity tracker.

A dual-mode tracker that combines deterministic quick-tap logging with a
generative natural-language chat path, orchestrated as a directed acyclic
graph (DAG) that mirrors the Google ADK 2.0 workflow lifecycle described in
the project PRD. Everything runs locally with no cloud dependency.
"""

__all__ = ["activity", "llm", "server", "store", "workflow"]
__version__ = "0.1.0"
