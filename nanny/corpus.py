"""Per-client parent-controlled reference corpus — hybrid RAG backend.

Keeps the shared UNICEF parenting guide as the default corpus, and lets each
parent upload their *own* references (their copy of a parenting book, a
pediatrician's handout, etc.) that the InsightsAgent can then retrieve from.

Two interchangeable backends sit behind one interface (``add_file`` /
``list_files`` / ``delete_file`` / ``resolve_*`` / ``retrieve``):

1. **Gemini File Search** (``google.genai`` ``file_search_stores``) — Google's
   managed, NotebookLM-style RAG: uploads are chunked + embedded server-side and
   retrieved with citations. Used when a **Gemini Developer API key**
   (``GEMINI_API_KEY``/``GOOGLE_API_KEY``, *not* Vertex) is configured and the
   File Search API is reachable. This is a different surface from Vertex AI RAG
   (``vertexai.rag``), so it sidesteps the Vertex embedding-model restrictions
   that broke the previous backend.

2. **Local BM25** (``rank-bm25`` over text chunks in a JSONL file under
   ``NANNY_DATA_DIR``) — a pure-Python lexical fallback with **no cloud, no
   embeddings, no API key**. Used when there's no usable Gemini key (the
   sandbox, tests, offline), so the corpus feature always works.

Selection is automatic (probe File Search when a key is present, else local) and
can be pinned with ``NANNY_CORPUS_BACKEND=auto|file_search|local``. Either way
``retrieve`` returns ``(text, filename)`` passages, so the retrieval tool and
per-file toggles in ``nanny/research.py`` are backend-agnostic.

Note: the local backend's JSONL lives on the machine that runs the server, so it
is shared with the agent only when the agent runs **in process** (the default
``_LocalRunnerBackend``); the File Search backend has no such constraint since
the store is server-side.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path

logger = logging.getLogger("nanny.corpus")

# Uploads are validated to these; both backends handle them (File Search parses
# server-side, the local backend via pypdf/decode).
ALLOWED_EXTENSIONS = (".txt", ".md", ".pdf")

_RETRIEVAL_MODEL = os.environ.get("NANNY_GEMINI_MODEL", "gemini-flash-latest")


def rag_enabled() -> bool:
    """The corpus is always available — File Search when a key allows it, else
    the local BM25 fallback. Kept as a function so callers read the same way
    they did when this was gated on ``NANNY_RAG_ENABLED``."""
    return True


# --- Backend selection -----------------------------------------------------

# Cached result of the one-time File Search reachability probe (None = not yet
# probed). Reset implicitly per process; tests select a backend via env.
_fs_probe: bool | None = None


def _has_dev_key() -> bool:
    """True when a Gemini *Developer API* key is configured (File Search is not
    available on the Vertex backend)."""
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return False
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _file_search_available() -> bool:
    """Whether to use File Search: a dev key must be present, and (in auto mode)
    the API must actually answer — if it's restricted on this account we fall
    back to local rather than failing every upload."""
    global _fs_probe
    if not _has_dev_key():
        return False
    if _fs_probe is not None:
        return _fs_probe
    try:
        # Cheapest reachable call: list stores. A permission/restriction error
        # here means File Search isn't usable → fall back to local.
        list(_fs_client().file_search_stores.list())
        _fs_probe = True
    except Exception as exc:
        logger.warning("File Search unavailable, using local BM25 corpus: %s", exc)
        _fs_probe = False
    return _fs_probe


def _use_file_search() -> bool:
    override = os.environ.get("NANNY_CORPUS_BACKEND", "auto").strip().lower()
    if override == "file_search":
        return True
    if override == "local":
        return False
    return _file_search_available()


# --- Shared helpers --------------------------------------------------------

_FS_DISPLAY_PREFIX = "nanny-corpus-"
_FS_SHARED_DISPLAY = "nanny-shared-unicef-corpus"


def _safe_client_id(client_id: str) -> str:
    # Same validation as _CLIENT_ID_RE in server.py / sources.py.
    if (
        not client_id
        or len(client_id) > 64
        or not all(c.isalnum() or c in ("-", "_") for c in client_id)
    ):
        return "default"
    return client_id


# ===========================================================================
# Gemini File Search backend
# ===========================================================================

_fs_client_singleton = None


def _fs_client():
    global _fs_client_singleton
    if _fs_client_singleton is None:
        from google import genai

        _fs_client_singleton = genai.Client(
            vertexai=False,
            api_key=os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY"),
        )
    return _fs_client_singleton


def _fs_display(client_id: str) -> str:
    return f"{_FS_DISPLAY_PREFIX}{_safe_client_id(client_id)}"


def _fs_resolve(display: str) -> str | None:
    for store in _fs_client().file_search_stores.list():
        if store.display_name == display:
            return store.name
    return None


def _fs_get_or_create(display: str) -> str:
    existing = _fs_resolve(display)
    if existing:
        return existing
    store = _fs_client().file_search_stores.create(config={"display_name": display})
    return store.name


def _fs_add(display: str, filename: str, data: bytes) -> None:
    store = _fs_get_or_create(display)
    suffix = os.path.splitext(filename)[1] or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        # File Search chunks + embeds server-side; the op resolves when the doc
        # is processed and searchable.
        _fs_client().file_search_stores.upload_to_file_search_store(
            file_search_store_name=store,
            file=tmp.name,
            config={"display_name": filename},
        )


def _fs_list(display: str) -> list[str]:
    store = _fs_resolve(display)
    if not store:
        return []
    names: list[str] = []
    for doc in _fs_client().file_search_stores.documents.list(parent=store):
        name = doc.display_name or doc.name or ""
        if name and name not in names:
            names.append(name)
    return names


def _fs_delete(display: str, filename: str) -> bool:
    store = _fs_resolve(display)
    if not store:
        return False
    for doc in _fs_client().file_search_stores.documents.list(parent=store):
        if doc.display_name == filename:
            _fs_client().file_search_stores.documents.delete(name=doc.name)
            return True
    return False


def _fs_retrieve(store_name: str, query: str, top_k: int) -> list[tuple[str, str]]:
    from google.genai import types

    resp = _fs_client().models.generate_content(
        model=_RETRIEVAL_MODEL,
        contents=query,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    file_search=types.FileSearch(file_search_store_names=[store_name])
                )
            ],
        ),
    )
    passages: list[tuple[str, str]] = []
    for cand in resp.candidates or []:
        meta = getattr(cand, "grounding_metadata", None)
        for chunk in getattr(meta, "grounding_chunks", None) or []:
            ctx = getattr(chunk, "retrieved_context", None)
            text = getattr(ctx, "text", None) if ctx else None
            if text:
                name = getattr(ctx, "document_name", None) or getattr(
                    ctx, "title", None
                )
                passages.append((text, name or ""))
    return passages[:top_k]


# ===========================================================================
# Local BM25 backend
# ===========================================================================

_CHUNK_CHARS = 800
_CHUNK_OVERLAP = 100
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _data_dir() -> Path:
    # Resolved per call (not at import) so tests can point NANNY_DATA_DIR at a
    # tmp dir without reloading the module.
    return Path(
        os.environ.get(
            "NANNY_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")
        )
    )


def _client_path(client_id: str) -> Path:
    return _data_dir() / "corpus" / "clients" / f"{_safe_client_id(client_id)}.jsonl"


def _shared_path() -> Path:
    # A separate subpath from clients/, so no client_id can ever resolve to it.
    return _data_dir() / "corpus" / "shared_unicef.jsonl"


def _extract_text(filename: str, data: bytes) -> str:
    """Parses an uploaded file to plain text. PDFs via pypdf; text/markdown
    decoded directly."""
    if os.path.splitext(filename)[1].lower() == ".pdf":
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return data.decode("utf-8", errors="replace")


def _chunk_text(text: str) -> list[str]:
    """Splits text into ~``_CHUNK_CHARS`` chunks with ``_CHUNK_OVERLAP`` overlap,
    breaking on whitespace so a chunk never ends mid-word."""
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        current.append(word)
        length += len(word) + 1
        if length >= _CHUNK_CHARS:
            chunks.append(" ".join(current))
            overlap: list[str] = []
            olen = 0
            for w in reversed(current):
                if olen >= _CHUNK_OVERLAP:
                    break
                overlap.insert(0, w)
                olen += len(w) + 1
            current = overlap
            length = olen
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _local_add(path: Path, filename: str, data: bytes) -> None:
    chunks = _chunk_text(_extract_text(filename, data))
    if not chunks:
        # Still record the file so it shows in the list (empty/scanned PDF).
        chunks = [filename]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for chunk in chunks:
            f.write(json.dumps({"filename": filename, "text": chunk}) + "\n")


def _local_list(path: Path) -> list[str]:
    seen: list[str] = []
    for row in _read_rows(path):
        name = row.get("filename", "")
        if name and name not in seen:
            seen.append(name)
    return seen


def _local_delete(path: Path, filename: str) -> bool:
    rows = _read_rows(path)
    kept = [r for r in rows if r.get("filename") != filename]
    if len(kept) == len(rows):
        return False
    if kept:
        path.write_text("".join(json.dumps(r) + "\n" for r in kept))
    else:
        path.unlink(missing_ok=True)
    return True


def _local_resolve(path: Path) -> str | None:
    return str(path) if path.exists() and _read_rows(path) else None


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _local_retrieve(handle: str, query: str, top_k: int) -> list[tuple[str, str]]:
    rows = _read_rows(Path(handle))
    if not rows:
        return []
    from rank_bm25 import BM25Okapi

    tokenized = [_tokenize(r.get("text", "")) for r in rows]
    query_tokens = _tokenize(query)
    query_set = set(query_tokens)
    scores = BM25Okapi(tokenized).get_scores(query_tokens)
    # Gate on real term overlap (BM25's IDF collapses to 0/negative on a tiny
    # corpus, so score>0 alone would drop obvious matches); rank the overlapping
    # chunks by overlap count, then BM25 score.
    candidates = [
        (len(query_set.intersection(toks)), scores[i], i)
        for i, toks in enumerate(tokenized)
        if query_set.intersection(toks)
    ]
    candidates.sort(reverse=True)
    return [
        (rows[i].get("text", ""), rows[i].get("filename", ""))
        for _overlap, _score, i in candidates[:top_k]
    ]


# ===========================================================================
# Public interface (backend-agnostic)
# ===========================================================================


def add_file(client_id: str, filename: str, data: bytes) -> None:
    """Indexes one reference file into the client's corpus."""
    if _use_file_search():
        _fs_add(_fs_display(client_id), filename, data)
    else:
        _local_add(_client_path(client_id), filename, data)


def list_files(client_id: str) -> list[str]:
    """Returns the display names of the client's uploaded references."""
    if _use_file_search():
        return _fs_list(_fs_display(client_id))
    return _local_list(_client_path(client_id))


def delete_file(client_id: str, filename: str) -> bool:
    """Removes one reference by name. Returns False if not found."""
    if _use_file_search():
        return _fs_delete(_fs_display(client_id), filename)
    return _local_delete(_client_path(client_id), filename)


def resolve_corpus_name(client_id: str) -> str | None:
    """Returns an opaque handle for this client's corpus (a File Search store
    name or a local file path), or None if the client has uploaded nothing."""
    if _use_file_search():
        return _fs_resolve(_fs_display(client_id))
    return _local_resolve(_client_path(client_id))


def resolve_shared_unicef_corpus() -> str | None:
    """Returns a handle for the shared UNICEF corpus, or None if unseeded
    (see ``nanny/seed_unicef_corpus.py``)."""
    if _use_file_search():
        return _fs_resolve(_FS_SHARED_DISPLAY)
    return _local_resolve(_shared_path())


def add_file_to_shared_unicef_corpus(filename: str, data: bytes) -> None:
    """Indexes a file into the shared UNICEF corpus. Seeding-script only."""
    if _use_file_search():
        _fs_add(_FS_SHARED_DISPLAY, filename, data)
    else:
        _local_add(_shared_path(), filename, data)


def retrieve(handle: str | None, query: str, top_k: int = 5) -> list[tuple[str, str]]:
    """Returns up to ``top_k`` ``(text, filename)`` passages from the corpus at
    ``handle`` most relevant to ``query`` — via File Search grounding or local
    BM25, whichever backend is active. Empty when the handle/query is empty or
    nothing relevant is found."""
    if not handle or not query.strip():
        return []
    if _use_file_search():
        try:
            return _fs_retrieve(handle, query, top_k)
        except Exception as exc:
            logger.warning("File Search retrieval failed: %s", exc)
            return []
    return _local_retrieve(handle, query, top_k)
