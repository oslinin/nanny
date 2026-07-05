"""Local dev entrypoint: `uv run main.py`."""

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "nanny.server:app",
        host="127.0.0.1",
        port=int(os.environ.get("NANNY_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
