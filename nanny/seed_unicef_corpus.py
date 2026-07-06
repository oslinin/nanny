"""One-time, operator-run script: seeds the shared UNICEF Vertex RAG corpus.

Every parent's InsightsAgent draws on one shared corpus rather than each
uploading their own copy of the same guide (see ``nanny/corpus.py`` and the
Corpus-tab design doc). Fetching the PDF directly from unicef.org returns
HTTP 403 (Cloudflare bot protection) from every environment tried, so it's
committed to the repo (``assets/The Art of Parenting.pdf``) and ingested from
there instead of at deploy time.

Run once per Vertex project, using your own gcloud/ADC credentials — the same
trust boundary as any other ``vertexai.rag`` call in this codebase:

    uv run python -m nanny.seed_unicef_corpus
    uv run python -m nanny.seed_unicef_corpus path/to/other.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import corpus

_DEFAULT_PDF = (
    Path(__file__).resolve().parent.parent / "assets" / "The Art of Parenting.pdf"
)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else _DEFAULT_PDF

    if not corpus.rag_enabled():
        print("NANNY_RAG_ENABLED is not set; nothing to seed.", file=sys.stderr)
        return 1
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 1

    corpus.add_file_to_shared_unicef_corpus(path.name, path.read_bytes())
    print(f"seeded the shared UNICEF corpus with {path.name!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
