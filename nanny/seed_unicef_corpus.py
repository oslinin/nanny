"""One-time, operator-run script: seeds the shared UNICEF Vertex RAG corpus.

Every parent's InsightsAgent draws on one shared corpus rather than each
uploading their own copy of the same guide (see ``nanny/corpus.py`` and the
Corpus-tab design doc). The PDF is **not committed to this repo** — the operator
supplies it at seed time: download UNICEF's "The Art of Parenting" guide (or any
parenting reference) and pass its path. (unicef.org returns HTTP 403 to
non-browser fetches, so it can't be pulled programmatically at deploy time.)

Run once per Vertex project, using your own gcloud/ADC credentials — the same
trust boundary as any other ``vertexai.rag`` call in this codebase:

    uv run --extra corpus-seed python -m nanny.seed_unicef_corpus "path/to/The Art of Parenting.pdf"
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import corpus

# A large single upload triggers enough internal embedding calls to trip
# Vertex AI RAG Engine's per-minute quota for its legacy textembedding-gecko
# fallback (see corpus._upload_file_to_corpus) more often than a small one.
# 50-page slices of the 248-page guide ingest reliably; splitting keeps each
# upload's burst small — corpus.py's retry-with-cooldown handles the rest.
# Retrieval works fine across multiple files in one corpus.
_MAX_PAGES_PER_FILE = 50


def _split_pdf(data: bytes) -> list[bytes]:
    import io

    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(io.BytesIO(data))
    parts = []
    for start in range(0, len(reader.pages), _MAX_PAGES_PER_FILE):
        writer = PdfWriter()
        for page in reader.pages[start : start + _MAX_PAGES_PER_FILE]:
            writer.add_page(page)
        buf = io.BytesIO()
        writer.write(buf)
        parts.append(buf.getvalue())
    return parts


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(
            "usage: python -m nanny.seed_unicef_corpus <path-to-pdf>",
            file=sys.stderr,
        )
        return 1
    path = Path(argv[0])

    if not corpus.rag_enabled():
        print("NANNY_RAG_ENABLED is not set; nothing to seed.", file=sys.stderr)
        return 1
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 1

    data = path.read_bytes()
    parts = _split_pdf(data) if path.suffix.lower() == ".pdf" else [data]

    for i, part in enumerate(parts, start=1):
        name = (
            path.name
            if len(parts) == 1
            else f"{path.stem} (part {i} of {len(parts)}){path.suffix}"
        )
        corpus.add_file_to_shared_unicef_corpus(name, part)
        print(f"seeded the shared UNICEF corpus with {name!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
