# syntax=docker/dockerfile:1
FROM python:3.11-slim

# uv gives fast, reproducible installs from uv.lock (same tool used in dev).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first so this layer is cached across code-only changes.
# --extra agent-engine bundles google-adk[gcp] (vertexai.agent_engines) so
# NANNY_AGENT_ENGINE_RESOURCE_NAME works out of the box; it's a no-op if you
# never set that env var (the dashboard runs the graph in-process instead).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra agent-engine

# Now copy the rest of the app (nanny/, web/, skills/, main.py) and install it.
COPY . .
RUN uv sync --frozen --no-dev --extra agent-engine

ENV PATH="/app/.venv/bin:${PATH}"
# Cloud Run's ephemeral filesystem resets on restart/redeploy — see README's
# Deployment section for what that means for this path (per-client activity
# logs, one file per X-Nanny-Client-Id under this directory).
ENV NANNY_DATA_DIR=/app/data

EXPOSE 8080
# Cloud Run injects $PORT (usually 8080); default it for local `docker run`.
CMD ["sh", "-c", "uv run uvicorn nanny.server:app --host 0.0.0.0 --port ${PORT:-8080}"]
