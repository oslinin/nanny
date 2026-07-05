# nanny

Local baby activity tracker built on **Google ADK 2.0** (`google-adk`).

Combines a deterministic quick-tap dashboard with a natural-language chat
interface over one shared local datastore, orchestrated as a directed acyclic
graph using `google.adk.workflow`:

```
START -> ClassifierNode -> RouterNode -> SaveActivityNode -> ResponderNode
              \_____________________________/
                          (error branch, either node) -> ErrorNode
```

- **ClassifierNode** — passes pre-formatted quick-tap JSON straight through
  (no LLM), or extracts a structured record from chat text via Gemini
  (`google.genai`, constrained JSON schema output).
- **RouterNode** — deterministic bookkeeping; declares which branch produced
  the record.
- **SaveActivityNode** — 100% deterministic; appends to a local JSON-lines
  log (`data/activity_log.jsonl`).
- **ResponderNode** — crafts a one-sentence natural confirmation from the
  save transaction metadata.

Every node reads/writes the real ADK session state (`ctx.state`), matching
the PRD's shared `BabyActivity` schema.

## Running

```sh
uv run main.py
```

Then open http://127.0.0.1:8000.

## LLM configuration

Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) to have the ClassifierNode and
ResponderNode call Gemini for real. Without a key, both fall back to a small
offline heuristic so the app is still fully runnable — every response
reports `used_llm_extraction` / `used_llm_response` so you can tell which
path served a given request.

## Development

```sh
uv sync                 # install runtime + dev deps
uv run pytest           # run tests
uv run agents-cli lint  # ruff + codespell + ty, via the ADK CLI toolchain
```

Deployment, CI/CD, and cloud configs are explicitly out of scope for this
project.
