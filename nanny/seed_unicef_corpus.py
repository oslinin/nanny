"""One-time, operator-run script: seeds the shared UNICEF reference corpus.

Every parent's InsightsAgent draws on one shared corpus rather than each
uploading their own copy of the same guide (see ``nanny/corpus.py``). The PDF is
**not committed to this repo** — the operator supplies it at seed time: download
UNICEF's "The Art of Parenting" guide (or any parenting reference) and pass its
path.

The corpus backend is chosen automatically (see ``nanny/corpus.py``): Gemini
File Search when a Gemini API key is configured, else the local BM25 store. Run
this once per environment:

    uv run python -m nanny.seed_unicef_corpus "path/to/The Art of Parenting.pdf"
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import corpus


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(
            "usage: python -m nanny.seed_unicef_corpus <path-to-pdf>",
            file=sys.stderr,
        )
        return 1
    path = Path(argv[0])
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 1

    corpus.add_file_to_shared_unicef_corpus(path.name, path.read_bytes())
    print(f"seeded the shared UNICEF corpus with {path.name!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
