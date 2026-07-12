#!/usr/bin/env bash
#
# One-command deploy for Nanny, driven entirely by .env:
#
#   1. agent      → Vertex AI Agent Runtime  (uv run python -m nanny.agent_engine_app)
#   2. dashboard  → Cloud Run                (gcloud run deploy, + IAM to call the agent)
#   3. frontend   → GitHub Pages             (point docs/index.html at the backend, push)
#
# Each step reads config from .env and writes results it discovers
# (the agent resource name, the Cloud Run URL) back into .env, so the next
# step — or a re-run — picks them up automatically.
#
# Usage:
#   scripts/deploy.sh                    # DRY RUN: print every command, change nothing
#   scripts/deploy.sh --execute          # actually deploy all three steps
#   scripts/deploy.sh dashboard --execute# run one step (agent | dashboard | frontend | all)
#   scripts/deploy.sh --help
#
# Nothing runs against Google Cloud or git unless you pass --execute. The dry
# run needs no credentials, so you can preview the whole pipeline anywhere.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DRY_RUN=1
STEP=all
for arg in "$@"; do
  case "$arg" in
    --execute) DRY_RUN=0 ;;
    --dry-run) DRY_RUN=1 ;;
    agent | dashboard | frontend | all) STEP="$arg" ;;
    -h | --help)
      sed -n '2,/^set -euo/p' "$0" | sed 's/^#\{0,1\} \{0,1\}//; /^set -euo/d'
      exit 0
      ;;
    *)
      echo "unknown argument: $arg (try --help)" >&2
      exit 2
      ;;
  esac
done

# Load .env (KEY=VALUE lines) into the environment.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${GOOGLE_CLOUD_LOCATION:=us-east1}"
: "${NANNY_CLOUD_RUN_SERVICE:=nanny}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "=== DRY RUN (nothing will be executed; pass --execute to deploy) ==="
fi

# --- helpers ----------------------------------------------------------------

# Echo a command, and run it only when executing for real.
run() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then "$@"; fi
}

# Require a var for a real deploy. In a dry run a missing value isn't fatal —
# substitute a visible placeholder and note it, so the whole pipeline still
# previews (e.g. NANNY_SERVICE_URL only exists after step 2 runs for real).
require() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      printf -v "$name" '<%s from .env>' "$name"
      export "${name?}"
      echo "  (note: $name is unset — required for --execute)"
    else
      echo "ERROR: $name is required — set it in .env (see .env.example)." >&2
      exit 1
    fi
  fi
}

# Persist KEY=VALUE into .env (replacing any existing line) so later steps
# and re-runs see it. Only writes for real; in a dry run it just reports.
persist() {
  local key="$1" val="$2"
  echo "  → .env: ${key}=${val}"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    touch .env
    grep -v "^${key}=" .env >.env.tmp 2>/dev/null || true
    mv .env.tmp .env
    echo "${key}=${val}" >>.env
  fi
}

# --- 1. agent → Vertex AI Agent Runtime -------------------------------------

deploy_agent() {
  echo; echo "── 1. Agent → Vertex AI Agent Runtime ──────────────────────────"
  require GOOGLE_CLOUD_PROJECT
  require GOOGLE_CLOUD_STAGING_BUCKET
  run gcloud config set project "$GOOGLE_CLOUD_PROJECT"
  run gcloud services enable aiplatform.googleapis.com --project "$GOOGLE_CLOUD_PROJECT"
  run uv sync --extra agent-engine
  if [[ "$DRY_RUN" -eq 0 ]]; then
    local resource
    resource="$(uv run python -m nanny.agent_engine_app | sed -n 's/^Deployed: //p' | tail -1)"
    if [[ -z "$resource" ]]; then
      echo "ERROR: agent deploy did not print a resource name." >&2
      exit 1
    fi
    persist NANNY_AGENT_ENGINE_RESOURCE_NAME "$resource"
  else
    run uv run python -m nanny.agent_engine_app --dry-run
    echo "  → .env: NANNY_AGENT_ENGINE_RESOURCE_NAME=<printed by the real deploy>"
  fi
}

# --- 2. dashboard → Cloud Run -----------------------------------------------

deploy_dashboard() {
  echo; echo "── 2. Dashboard → Cloud Run ────────────────────────────────────"
  require GOOGLE_CLOUD_PROJECT

  # A shared access-token gates the public API; generate one if none is set.
  if [[ -z "${NANNY_API_TOKEN:-}" ]]; then
    if [[ "$DRY_RUN" -eq 0 ]]; then
      NANNY_API_TOKEN="$(openssl rand -hex 16)"
    else
      NANNY_API_TOKEN="<generated on --execute>"
    fi
    persist NANNY_API_TOKEN "$NANNY_API_TOKEN"
  fi

  run gcloud services enable run.googleapis.com --project "$GOOGLE_CLOUD_PROJECT"

  local env="NANNY_API_TOKEN=${NANNY_API_TOKEN}"
  env+=",GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}"
  env+=",GOOGLE_CLOUD_LOCATION=${GOOGLE_CLOUD_LOCATION}"
  [[ -n "${NANNY_ALLOWED_ORIGINS:-}" ]] && env+=",NANNY_ALLOWED_ORIGINS=${NANNY_ALLOWED_ORIGINS}"
  [[ -n "${NANNY_AGENT_ENGINE_RESOURCE_NAME:-}" ]] && env+=",NANNY_AGENT_ENGINE_RESOURCE_NAME=${NANNY_AGENT_ENGINE_RESOURCE_NAME}"
  [[ -n "${NANNY_STT_ENABLED:-}" ]] && env+=",NANNY_STT_ENABLED=${NANNY_STT_ENABLED}"
  # Reference-corpus backend: auto (default) picks Gemini File Search when
  # GEMINI_API_KEY is set below, else the local BM25 index. Pin it explicitly
  # with NANNY_CORPUS_BACKEND=file_search|local when needed. File Search is the
  # right choice for a deploy — local BM25 lives on the instance's own disk.
  [[ -n "${NANNY_CORPUS_BACKEND:-}" ]] && env+=",NANNY_CORPUS_BACKEND=${NANNY_CORPUS_BACKEND}"
  # Model backend for the in-process path (when no Agent Runtime resource) and
  # the opt-in Google-search grounding, passed through when present in .env.
  [[ -n "${GOOGLE_GENAI_USE_VERTEXAI:-}" ]] && env+=",GOOGLE_GENAI_USE_VERTEXAI=${GOOGLE_GENAI_USE_VERTEXAI}"
  [[ -n "${GEMINI_API_KEY:-}" ]] && env+=",GEMINI_API_KEY=${GEMINI_API_KEY}"
  [[ -n "${GOOGLE_CSE_ID:-}" ]] && env+=",GOOGLE_CSE_ID=${GOOGLE_CSE_ID}"
  [[ -n "${GOOGLE_CSE_API_KEY:-}" ]] && env+=",GOOGLE_CSE_API_KEY=${GOOGLE_CSE_API_KEY}"

  run gcloud run deploy "$NANNY_CLOUD_RUN_SERVICE" --source . \
    --project "$GOOGLE_CLOUD_PROJECT" --region "$GOOGLE_CLOUD_LOCATION" \
    --allow-unauthenticated --min-instances=1 --max-instances=1 \
    --set-env-vars="$env"

  # The dashboard proxies to the Agent Runtime resource, so its service account
  # needs permission to call Vertex — only relevant when the resource is set.
  if [[ -n "${NANNY_AGENT_ENGINE_RESOURCE_NAME:-}" ]]; then
    if [[ "$DRY_RUN" -eq 0 ]]; then
      local sa
      sa="$(gcloud run services describe "$NANNY_CLOUD_RUN_SERVICE" \
        --project "$GOOGLE_CLOUD_PROJECT" --region "$GOOGLE_CLOUD_LOCATION" \
        --format='value(spec.template.spec.serviceAccountName)')"
      run gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" \
        --member="serviceAccount:${sa}" --role="roles/aiplatform.user"
    else
      echo "+ gcloud projects add-iam-policy-binding \$PROJECT \\"
      echo "    --member=serviceAccount:<Cloud Run SA> --role=roles/aiplatform.user"
    fi
  fi

  if [[ "$DRY_RUN" -eq 0 ]]; then
    local url
    url="$(gcloud run services describe "$NANNY_CLOUD_RUN_SERVICE" \
      --project "$GOOGLE_CLOUD_PROJECT" --region "$GOOGLE_CLOUD_LOCATION" \
      --format='value(status.url)')"
    persist NANNY_SERVICE_URL "$url"
  else
    echo "  → .env: NANNY_SERVICE_URL=<Cloud Run URL, printed by the real deploy>"
  fi
}

# --- 3. frontend → GitHub Pages ---------------------------------------------

deploy_frontend() {
  echo; echo "── 3. Frontend → GitHub Pages ──────────────────────────────────"
  require NANNY_SERVICE_URL
  local token="${NANNY_API_TOKEN:-}"

  # Point the static GitHub Pages copy at the deployed backend (idempotent —
  # replaces whatever the two config values currently hold).
  run sed -i.bak -E \
    "s#(window.NANNY_API_BASE = \")[^\"]*(\";)#\\1${NANNY_SERVICE_URL}\\2#" \
    docs/index.html
  run sed -i.bak -E \
    "s#(window.NANNY_API_TOKEN = \")[^\"]*(\";)#\\1${token}\\2#" \
    docs/index.html
  run rm -f docs/index.html.bak

  run git add docs/index.html
  run git commit -m "Point GitHub Pages frontend at the deployed backend"
  run git push

  echo
  echo "One-time GitHub setup (not scriptable from here): repo Settings → Pages"
  echo "→ Source 'Deploy from a branch', branch main, folder /docs."
  echo "Then the frontend is live at https://<user>.github.io/<repo>/."
}

# --- run selected steps -----------------------------------------------------

case "$STEP" in
  agent) deploy_agent ;;
  dashboard) deploy_dashboard ;;
  frontend) deploy_frontend ;;
  all)
    deploy_agent
    deploy_dashboard
    deploy_frontend
    ;;
esac

echo
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "=== dry run complete — re-run with --execute to deploy for real ==="
else
  echo "=== done ==="
fi
