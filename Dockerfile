# syntax=docker/dockerfile:1
FROM python:3.11-slim

# uv gives fast, reproducible installs from uv.lock (same tool used in dev).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first so this layer is cached across code-only changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Now copy the rest of the app (nanny/, web/, skills/, main.py) and install it.
COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"
# Cloud Run's ephemeral filesystem resets on restart/redeploy — see README's
# Deployment section for what that means for this path.
ENV NANNY_DATA_PATH=/app/data/activity_log.jsonl

EXPOSE 8080
# Cloud Run injects $PORT (usually 8080); default it for local `docker run`.
CMD ["sh", "-c", "uv run uvicorn nanny.server:app --host 0.0.0.0 --port ${PORT:-8080}"]
