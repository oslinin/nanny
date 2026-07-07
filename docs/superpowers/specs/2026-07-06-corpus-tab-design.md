# Corpus tab — design

## Goal

Give parents control over which evidence sources `InsightsAgent` draws on
when answering a question. Today the retrieval sources are all-or-nothing,
set once at deploy time via env vars (`GOOGLE_CSE_ID`/`GOOGLE_CSE_API_KEY`,
`NANNY_RAG_ENABLED`). Add a **Corpus** tab, alongside Today/History, where a
parent can:

- Turn Google Search (the scoped CSE tool) on or off.
- See a NotebookLM-style list of reference documents — a shared UNICEF
  guide pre-loaded first, then anything they've personally uploaded — and
  check/uncheck which ones the agent may draw on.
- Still upload/delete their own reference files (today's "Your references"
  panel, moved into this tab).

Unchecking a source must be **hard enforcement**: the agent literally
cannot see that source's content for the turn, not just a prompt telling it
not to use it.

The always-on `child-guidance` skill (curated, offline, cites UNICEF/CDC/
AAP/WHO/OpenStax) is unaffected — it has no GCP dependency and isn't part of
this toggle system. The three things gated here are opt-in *extras* layered
on top of it. Consensus.app (`NANNY_CONSENSUS_MCP_URL`) is likewise out of
scope — always on when configured.

## Data model

New `nanny/sources.py`, mirroring `nanny/stores.py`'s per-client file
resolution:

```
data/<client_id>.sources.json
{
  "google_search": true,
  "unicef": true,
  "uploads": { "my-pediatrician-notes.pdf": false }
}
```

- `google_search` / `unicef`: booleans, default `true` when the file or key
  is missing.
- `uploads`: maps a personally-uploaded filename to enabled/disabled;
  a filename not present in the map defaults to enabled (so a freshly
  uploaded file is usable immediately, no extra step).

`nanny/sources.py` exposes:

- `get_prefs(client_id) -> dict` — reads the file, filling in defaults.
- `set_google_search(client_id, enabled: bool) -> dict`
- `set_unicef(client_id, enabled: bool) -> dict`
- `set_upload_enabled(client_id, filename: str, enabled: bool) -> dict`
- `availability(client_id) -> dict` — which sources are actually configured
  server-side (see below), independent of the parent's on/off choice.

## The shared UNICEF corpus

An operator-supplied copy of UNICEF's "The Art of Parenting" guide seeds
**one shared Vertex RAG corpus**, not a per-client one — every parent draws
on the same ingested document rather than each getting their own copy
uploaded to Vertex. The PDF is **not committed to this repo** (it's a
copyrighted file whose license isn't confirmed here); the operator passes its
path to the seed script.

`nanny/corpus.py` gains:

- A fixed display name for the shared corpus (distinct from the
  `nanny-corpus-<client_id>` naming used for personal corpora), so it can
  never collide with — or be reachable through — a client-supplied id.
- `get_or_create_shared_unicef_corpus()` / `resolve_shared_unicef_corpus()`
  and an add-file helper, used only by the seeding script below. **Not**
  exposed through the public `/api/corpus` endpoints — those stay scoped to
  the caller's own `X-Nanny-Client-Id`, so no request can add to, list, or
  delete from the shared corpus.

**Seeding is a one-time, manual, operator-run step** — not automated
ingestion. Fetching the PDF directly from unicef.org returns HTTP 403
(Cloudflare bot protection) from every environment tried while designing
this, including this sandbox, so there is no reliable way to fetch it
programmatically at deploy time; the operator downloads it once and passes
its path.

```
uv run python -m nanny.seed_unicef_corpus "path/to/The Art of Parenting.pdf"
```

Run once per Vertex project, by whoever operates the deployment, using
their own `gcloud`/ADC credentials — same trust boundary as any other
`vertexai.rag` call in this codebase.

## Availability

`nanny/sources.py`'s `availability(client_id)`:

- `google_search`: `GOOGLE_CSE_ID` and `GOOGLE_CSE_API_KEY` both set.
- `unicef`: `corpus.rag_enabled()` **and** the shared corpus has actually
  been seeded (`resolve_shared_unicef_corpus()` returns something) — so the
  row stays hidden until the operator runs the seed script, rather than
  showing a checkbox that silently does nothing.
- `uploads`: `corpus.rag_enabled()` (unchanged from today's `refs-panel`
  gating) — the personal-upload part of the list (and its upload form) only
  appears when RAG is on, though the UNICEF row can appear independently if
  only the shared corpus is seeded.

## Retrieval enforcement

`research.py` changes:

- **Google Search**: unchanged mechanism from today (`_search_reputable_
  child_health`, a plain function tool), but now wrapped so it is *removed
  entirely from the model's tool list* for a turn where `google_search` is
  false in state — a `before_model_callback`-level filter, not a prompt
  instruction. When absent from the tools the model is given, it cannot be
  called, full stop.
- **References** (UNICEF + personal uploads): `_PerClientRagRetrieval` is
  extended into one merged `search_my_references` tool:
  1. If `unicef` is enabled for this client, queries the shared UNICEF
     corpus and includes its passages.
  2. Queries the client's own corpus (if any), then **drops any passage
     whose source file is disabled** in `uploads` before returning
     anything to the model — filtering happens in tool code, before the
     result ever reaches the LLM, so a disabled file's content is never
     visible to it regardless of what the model does.
  3. Concatenates whatever passages survive; if none do (everything
     disabled, or nothing uploaded/seeded), returns the existing "no
     relevant passages" message.
  This tool is attached whenever any RAG source is available at all
  (`unicef` seeded or the client has personal files); the per-document
  filtering inside it is what actually enforces each checkbox.

`workflow.py`'s `insights_prep_node` gains one line loading
`sources.get_prefs(client_id)` into `ctx.state["enabled_sources"]`, read by
both the tool-filtering callback and the retrieval tool above.

## API

- `GET /api/sources` — returns:
  ```json
  {
    "google_search": {"available": true, "enabled": true},
    "documents": [
      {"name": "The Art of Parenting.pdf", "source": "unicef", "enabled": true, "deletable": true},
      {"name": "my-notes.pdf", "source": "upload", "enabled": false, "deletable": true}
    ]
  }
  ```
  The `unicef` row is just a **default entry** in the list, not a special
  fixture: it's included only when `availability(...).unicef` is true *and*
  the client hasn't removed it (`prefs["unicef"]` is true) — once removed it
  drops out of that client's list entirely rather than lingering unchecked,
  the same as a deleted upload disappearing. "Removed" is client-scoped
  (persists `unicef: false` in that client's own prefs file); the shared
  corpus itself is untouched, so another client still sees the row until
  they remove it too. Not every client has to use it. Personal-upload rows
  come from `corpus.list_files(client_id)` merged with the client's
  `uploads` prefs (only present when `availability(...).uploads` is true).
- `POST /api/sources` — partial update, one of:
  - `{"google_search": true}`
  - `{"document": {"source": "unicef", "enabled": false}}`
  - `{"document": {"source": "upload", "name": "my-notes.pdf", "enabled": false}}`

  Persists via `nanny/sources.py` and returns the same shape as the GET.
- `/api/corpus` (upload/list/delete of personal files) is unchanged.

## Frontend

New tab button: `Corpus`, alongside Today/History. New `#view-corpus`
section, replacing today's inline `refs-panel` (which is deleted from the
Today view):

- A **Google Search** row: label, short description ("Live search of
  reputable sites — CDC, AAP, WHO, healthychildren.org"), and a checkbox.
  Omitted if not available.
- A **References** list, NotebookLM-style — one row per document:
  - "The Art of Parenting" with a small "UNICEF" badge, checkbox, and a
    delete button like any other row — it's a default entry a client can
    remove and rely solely on their own uploads instead. Deleting it calls
    `POST /api/sources` (disables it for this client) rather than
    `/api/corpus`'s real file deletion, since the corpus is shared. Omitted
    if not available or already removed by this client.
  - Each personally uploaded file: checkbox + delete button (today's
    existing delete behavior, unchanged — a real `DELETE /api/corpus/{f}`).
  - The existing upload form below the list, shown only when uploads are
    available.
- Toggling any checkbox fires the corresponding `POST /api/sources` call
  immediately and re-renders from the response, matching the existing
  corpus-delete interaction pattern (act on click, no separate save step).
- If nothing is available at all (plain local dev, nothing configured),
  the tab shows a short explanatory note instead of an empty panel.

## Docs

- `README.md`: refresh the endpoint table (`/api/sources` alongside
  `/api/corpus`), the "Bring your own references (RAG)" optional-feature
  bullet to mention the shared UNICEF seed step and per-source toggles, and
  the sentence describing `InsightsAgent`'s tools to mention that the
  parent controls them from the Corpus tab.
- Mermaid orchestration graph: no new nodes (this is entirely within
  `InsightsAgent`'s existing tool set), but the accompanying prose gains a
  short note about per-parent source toggles.

## Out of scope (YAGNI)

- Per-file toggles are not extended to Consensus.app (always on when
  configured — not one of the three requested sources).
- No admin UI for re-seeding or replacing the shared UNICEF corpus; the
  seed script is the only interface.
- No multi-file batch upload — the existing single-file upload form is
  unchanged.
