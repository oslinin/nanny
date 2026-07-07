"""Per-client parent-controlled reference corpus, backed by Vertex AI RAG.

Keeps the shared ``child-guidance`` skill as the default, and lets each parent
upload their *own* references (their copy of a parenting book, a pediatrician's
handout, etc.) that the InsightsAgent can then retrieve from. Each client id
gets its own managed Vertex RAG corpus — the same per-visitor isolation as the
activity log — named deterministically so both this module (corpus management,
run from the dashboard) and the retrieval tool (run inside the agent) resolve
the same corpus without any shared mapping.

Also manages **one shared** corpus (seeded with UNICEF's "Art of Parenting"
guide, see ``nanny/seed_unicef_corpus.py``) that every client's InsightsAgent
can draw on — distinct from any per-client corpus, and reachable only through
the fixed display name below, never through a client-supplied id.

Gated entirely behind ``NANNY_RAG_ENABLED``: off (the default, and every local
/test/sandbox run) this module makes no Vertex calls at all and the corpus
endpoints report the feature disabled. On (a real Vertex deployment) the
``vertexai.rag`` calls run against Google Cloud with the caller's service
account (ADC) — which is why none of this is exercisable without GCP
credentials, the same constraint as the Agent Runtime deploy.
"""

from __future__ import annotations

import os
import tempfile

# One shared prefix so a client's corpus is found by display name from either
# side of the Agent Runtime split, with no persisted corpus-id mapping.
_DISPLAY_PREFIX = "nanny-corpus-"

# A fixed name distinct from _DISPLAY_PREFIX's per-client pattern above, so no
# client_id can ever collide with — or resolve to — the shared corpus.
_SHARED_UNICEF_DISPLAY_NAME = "nanny-shared-unicef-corpus"

# Vertex RAG parses these natively (no local text extraction needed).
ALLOWED_EXTENSIONS = (".txt", ".md", ".pdf")


def rag_enabled() -> bool:
    """The single flag every other module checks before touching Vertex RAG."""
    return os.environ.get("NANNY_RAG_ENABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def display_name(client_id: str) -> str:
    return f"{_DISPLAY_PREFIX}{client_id}"


def _init_vertex() -> None:
    import vertexai

    # Deliberately not GOOGLE_CLOUD_LOCATION: that's commonly set to "global"
    # for ADK's built-in google_search grounding tool (see nanny/research.py),
    # but Vertex AI RAG isn't available in "global" and needs a real region.
    vertexai.init(
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ.get("NANNY_RAG_LOCATION", "us-east1"),
    )


def resolve_corpus_name(client_id: str) -> str | None:
    """Returns this client's RAG corpus resource name, or None if not created.

    Resolves by matching the deterministic display name, so no corpus-id needs
    to be persisted — the name is a pure function of the client id.
    """
    from vertexai import rag

    _init_vertex()
    wanted = display_name(client_id)
    for corpus in rag.list_corpora():
        if corpus.display_name == wanted:
            return corpus.name
    return None


def _embedding_model_config():
    """Pins the RAG embedding model to text-embedding-005.

    The SDK default, textembedding-gecko, has zero default quota for new
    projects (``429 Quota exceeded ... base model: textembedding-gecko``) —
    text-embedding-005 is the current model and gets real default quota.
    """
    from vertexai import rag

    return rag.RagVectorDbConfig(
        rag_embedding_model_config=rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                publisher_model="publishers/google/models/text-embedding-005",
            )
        )
    )


def get_or_create_corpus(client_id: str) -> str:
    """Returns this client's corpus resource name, creating it on first use."""
    from vertexai import rag

    existing = resolve_corpus_name(client_id)
    if existing:
        return existing
    corpus = rag.create_corpus(
        display_name=display_name(client_id),
        description="Parent-supplied reference material for the Nanny InsightsAgent.",
        backend_config=_embedding_model_config(),
    )
    return corpus.name


# Vertex AI RAG Engine's ingest pipeline appears to invoke a legacy
# textembedding-gecko call for some internal step regardless of the corpus's
# configured embedding model (see _embedding_model_config) — and new projects
# get zero *sustained* quota for it, only a small per-minute burst. Uploading
# one file after another can trip a 429 even though each succeeds in
# isolation once the window resets, so retry with a cooldown.
_QUOTA_RETRY_ATTEMPTS = 5
_QUOTA_RETRY_COOLDOWN_SECONDS = 65


def _upload_file_to_corpus(corpus_name: str, filename: str, data: bytes) -> None:
    """Uploads one file into an already-resolved corpus.

    The bytes are written to a temp path because ``rag.upload_file`` takes a
    filesystem path; Vertex RAG extracts the text (PDF/TXT/MD) on ingest.
    """
    import time

    from vertexai import rag

    suffix = os.path.splitext(filename)[1] or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        for attempt in range(1, _QUOTA_RETRY_ATTEMPTS + 1):
            try:
                rag.upload_file(
                    corpus_name=corpus_name,
                    path=tmp.name,
                    display_name=filename,
                )
                return
            except RuntimeError as e:
                if "Quota exceeded" not in str(e) or attempt == _QUOTA_RETRY_ATTEMPTS:
                    raise
                time.sleep(_QUOTA_RETRY_COOLDOWN_SECONDS)


def add_file(client_id: str, filename: str, data: bytes) -> None:
    """Uploads one reference file into the client's corpus."""
    corpus_name = get_or_create_corpus(client_id)
    _upload_file_to_corpus(corpus_name, filename, data)


def list_files(client_id: str) -> list[str]:
    """Returns the display names of the client's uploaded references."""
    from vertexai import rag

    corpus_name = resolve_corpus_name(client_id)
    if not corpus_name:
        return []
    return [f.display_name for f in rag.list_files(corpus_name=corpus_name)]


def delete_file(client_id: str, filename: str) -> bool:
    """Removes one reference by display name. Returns False if not found."""
    from vertexai import rag

    corpus_name = resolve_corpus_name(client_id)
    if not corpus_name:
        return False
    for f in rag.list_files(corpus_name=corpus_name):
        if f.display_name == filename:
            rag.delete_file(name=f.name, corpus_name=corpus_name)
            return True
    return False


def resolve_shared_unicef_corpus() -> str | None:
    """Returns the shared UNICEF corpus's resource name, or None if it hasn't
    been seeded yet (see ``nanny/seed_unicef_corpus.py``)."""
    from vertexai import rag

    _init_vertex()
    for corpus in rag.list_corpora():
        if corpus.display_name == _SHARED_UNICEF_DISPLAY_NAME:
            return corpus.name
    return None


def get_or_create_shared_unicef_corpus() -> str:
    """Returns the shared UNICEF corpus's resource name, creating it on first
    use. Only called by the one-time seeding script — never from a request
    path, so no client can trigger its creation."""
    from vertexai import rag

    existing = resolve_shared_unicef_corpus()
    if existing:
        return existing
    corpus = rag.create_corpus(
        display_name=_SHARED_UNICEF_DISPLAY_NAME,
        description="Shared UNICEF parenting guidance, available to every parent.",
        backend_config=_embedding_model_config(),
    )
    return corpus.name


def add_file_to_shared_unicef_corpus(filename: str, data: bytes) -> None:
    """Uploads a file into the shared UNICEF corpus. Seeding-script only."""
    corpus_name = get_or_create_shared_unicef_corpus()
    _upload_file_to_corpus(corpus_name, filename, data)
