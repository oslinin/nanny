"""Per-client parent-controlled reference corpus, backed by Vertex AI RAG.

Keeps the shared ``child-guidance`` skill as the default, and lets each parent
upload their *own* references (their copy of a parenting book, a pediatrician's
handout, etc.) that the InsightsAgent can then retrieve from. Each client id
gets its own managed Vertex RAG corpus — the same per-visitor isolation as the
activity log — named deterministically so both this module (corpus management,
run from the dashboard) and the retrieval tool (run inside the agent) resolve
the same corpus without any shared mapping.

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

    vertexai.init(
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east1"),
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


def get_or_create_corpus(client_id: str) -> str:
    """Returns this client's corpus resource name, creating it on first use."""
    from vertexai import rag

    existing = resolve_corpus_name(client_id)
    if existing:
        return existing
    corpus = rag.create_corpus(
        display_name=display_name(client_id),
        description="Parent-supplied reference material for the Nanny InsightsAgent.",
    )
    return corpus.name


def add_file(client_id: str, filename: str, data: bytes) -> None:
    """Uploads one reference file into the client's corpus.

    The bytes are written to a temp path because ``rag.upload_file`` takes a
    filesystem path; Vertex RAG extracts the text (PDF/TXT/MD) on ingest.
    """
    from vertexai import rag

    corpus_name = get_or_create_corpus(client_id)
    suffix = os.path.splitext(filename)[1] or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        rag.upload_file(
            corpus_name=corpus_name,
            path=tmp.name,
            display_name=filename,
        )


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
