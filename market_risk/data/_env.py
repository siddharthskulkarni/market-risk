"""Load project .env from repo root."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_project_dotenv() -> Path | None:
    """Search upward from cwd for .env and load it."""
    for parent in [Path.cwd(), *Path.cwd().parents]:
        env_file = parent / ".env"
        if env_file.is_file():
            load_dotenv(env_file)
            return env_file
    load_dotenv()
    return None
