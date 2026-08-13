"""Read a MinerU-parsed paper corpus (zip) into clean, section-sliced text.

Deterministic, no-LLM front-end for the discovery-pattern miner
(``reconstruct/discovery_pattern.py``). Each paper in the corpus is a
``<id>.mineru/`` directory holding a ``*_content_list.json`` (typed blocks:
header / text / equation / image with page_idx + text_level). We group the
``text`` blocks under their nearest ``text_level==1`` header to recover the
paper's logical sections (ABSTRACT / INTRODUCTION / METHODS / RESULTS /
CONCLUSION …), then expose the sections a researcher actually reads to infer
the discovery arc — without ever extracting the 19k-file zip to disk.

This is the §2-layer-1 source (paper + SI) as a *clean* artifact. It produces
no trajectory by itself; it feeds the LLM reconstruction (§5) which is gated as
``pending-soft-verify`` until the external AI Verifier lands (v5 §0.9).
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

# Blocks that are never part of the scientific narrative.
_DROP_TYPES = {"footer", "page_number", "ref_text", "aside_text", "page_footer", "page_header"}

# Canonical section buckets we care about, matched case-insensitively against the
# header text. Order matters: first matching pattern wins.
_SECTION_PATTERNS = [
    ("abstract", re.compile(r"\babstract\b", re.I)),
    ("introduction", re.compile(r"\bintroduction\b|^1\.?\s", re.I)),
    ("methods", re.compile(r"\b(method|computational|experimental|theory|theoretical|model)", re.I)),
    ("results", re.compile(r"\b(result|discussion)", re.I)),
    ("conclusion", re.compile(r"\b(conclusion|summary|outlook)", re.I)),
]


@dataclass
class Paper:
    """One MinerU-parsed paper, sliced into named sections."""
    paper_id: str                       # the .mineru dir stem (DOI-ish or OpenAlex id)
    title: str
    sections: dict[str, str] = field(default_factory=dict)   # bucket -> joined text
    artifact_uri: str = ""              # zip-relative path to the content_list.json

    def section(self, *names: str, cap: int = 6000) -> str:
        """Return the first present section among ``names``, truncated to ``cap`` chars."""
        for nm in names:
            if nm in self.sections and self.sections[nm].strip():
                return self.sections[nm][:cap]
        return ""

    def narrative(self, *, intro_cap: int = 5000, results_cap: int = 6000) -> str:
        """The text a researcher reads to infer the discovery arc (intro→method→results→conclusion)."""
        parts = []
        for nm, cap in (("abstract", 1800), ("introduction", intro_cap), ("methods", 3500),
                        ("results", results_cap), ("conclusion", 2500)):
            t = self.section(nm, cap=cap)
            if t:
                parts.append(f"## {nm.upper()}\n{t}")
        narrative = "\n\n".join(parts)
        # Fallback: some papers carry non-canonical headers (e.g. unnumbered
        # journal styles), so canonical bucketing recovers little. Rather than
        # waste a fully-parsed paper, concatenate all non-title section text.
        if len(narrative) < 600:
            body = " ".join(t for k, t in self.sections.items()
                            if k != "_pre" and k != self.title[:40] and t.strip())
            if len(body) > len(narrative):
                narrative = f"## BODY\n{body[: intro_cap + results_cap + 5000]}"
        return narrative


def _bucket(header: str) -> Optional[str]:
    for name, pat in _SECTION_PATTERNS:
        if pat.search(header):
            return name
    return None


def _slice_sections(blocks: list[dict]) -> tuple[str, dict[str, str]]:
    """Group text blocks under their nearest level-1 header into canonical buckets."""
    title = ""
    raw: dict[str, list[str]] = {}
    cur = "_pre"
    # MinerU's two parse backends tag headers differently: the `vlm` backend
    # marks every heading (incl. the title) as text_level==1; the `pipeline`
    # backend reserves text_level==1 for the title and tags section headers as
    # text_level==2. Treat both 1 and 2 as section breaks so either parse slices.
    header_levels = {b.get("text_level") for b in blocks
                     if b.get("text_level") in (1, 2)
                     and b.get("type") in ("header", "text")}
    section_levels = header_levels or {1}
    for b in blocks:
        btype = b.get("type")
        if btype in _DROP_TYPES:
            continue
        if b.get("text_level") in section_levels and (btype == "header" or btype == "text"):
            header = (b.get("text") or "").strip()
            if header and not title and len(header) > 12 and "cite" not in header.lower():
                title = header                       # first substantive heading ≈ paper title
            cur = _bucket(header) or header[:40]
            continue
        if btype in ("text", "list", "equation"):
            txt = (b.get("text") or "").strip()
            if txt:
                raw.setdefault(cur, []).append(txt)
    sections = {k: " ".join(v) for k, v in raw.items()}
    return title or "(untitled)", sections


class PaperCorpus:
    """Reads MinerU papers from a corpus, either a ``.zip`` (lazily, never extracting
    the whole archive) or an already-extracted directory tree. Pass either path; the
    constructor sniffs which it is. Directory mode avoids re-opening the 1.2G zip on
    every run once the corpus has been unzipped to disk."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._zf: Optional[zipfile.ZipFile] = None
        # map paper_id -> content_list.json reference (zip member name OR fs path).
        # Prefer the vlm parse when a paper is duplicated across parses.
        self._index: dict[str, str] = {}
        if self.path.is_dir():
            self.zip_path = None
            for p in self.path.rglob("*content_list.json"):
                sp = str(p)
                if ".mineru/" not in sp:
                    continue
                pid = sp.split(".mineru/")[0].split("/")[-1]
                if pid not in self._index or "/vlm/" in sp:
                    self._index[pid] = sp
        else:
            self.zip_path = self.path
            self._zf = zipfile.ZipFile(self.path)
            for n in self._zf.namelist():
                if n.endswith("content_list.json") and ".mineru/" in n:
                    pid = n.split(".mineru/")[0].split("/")[-1]
                    if pid not in self._index or "/vlm/" in n:
                        self._index[pid] = n

    def ids(self) -> list[str]:
        return sorted(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def _read(self, ref: str) -> bytes:
        return self._zf.read(ref) if self._zf is not None else Path(ref).read_bytes()

    def get(self, paper_id: str) -> Paper:
        ref = self._index[paper_id]
        blocks = json.loads(self._read(ref))
        title, sections = _slice_sections(blocks)
        return Paper(paper_id=paper_id, title=title, sections=sections, artifact_uri=ref)

    def papers(self, limit: Optional[int] = None) -> Iterator[Paper]:
        for i, pid in enumerate(self.ids()):
            if limit is not None and i >= limit:
                return
            yield self.get(pid)
