"""Runtime configuration — every path resolved on use, never at import time.

A library must be importable without side effects: ``import autoresearcheval`` has
to work in a process whose working directory contains no corpus, with no environment
set, without touching the filesystem. The CLI scripts this package grew out of
resolved these as module constants at import, which made them unusable as a library
(importing the classifier scanned the cwd for model directories).

Everything here is therefore a function. The environment variables keep the exact
meanings the command-line tools documented.

Packaged data (the ONBOARDING framework, the ARFT guide, the depth exemplar) is
resolved against ``__file__`` rather than through ``importlib.resources``, because
these paths are handed to subprocesses — a spawned analysis session is told to read
ONBOARDING.md by absolute path, so it has to exist on a real filesystem.
"""
from __future__ import annotations

import os
from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data"


def data_path(name: str) -> Path:
    """Absolute path to one packaged data file."""
    return _DATA / name


def onboarding() -> Path:
    """The deep-dive framework handed to every Stage 1 session."""
    return Path(os.environ.get("AAJ_ONBOARDING") or data_path("ONBOARDING.md"))


def exemplar() -> Path:
    """The worked reference analysis Stage 1 uses as its depth standard.

    Missing is tolerated by the caller: the prompt drops the line rather than
    telling a session to read a path that is not there.
    """
    return Path(os.environ.get("AAJ_EXEMPLAR") or data_path("analysis_long.md"))


def arft_guide() -> Path:
    """The operational guide handed to the Stage 2 classifier."""
    return Path(os.environ.get("AAJ_GUIDE") or data_path("arft_guide.md"))


def corpus_dir() -> Path:
    """Where Stage 1 writes and Stage 2 reads: <corpus>/<model>/<task_id>/analysis.md."""
    return Path(os.environ.get("AAJ_CORPUS_DIR", "corpus")).resolve()


def out_dir() -> Path:
    """Where Stage 2 writes classifications and rolled-up statistics."""
    return Path(os.environ.get("AAJ_OUT_DIR", "results")).resolve()


def models() -> list[str]:
    """Model keys discovered under the corpus root.

    Underscore-prefixed directories are bookkeeping, not models: Stage 1 writes its
    run manifests to <corpus>/_batch/.
    """
    root = corpus_dir()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and not p.name.startswith("_"))
