"""The two-call public API.

    from autoresearcheval import generate_analysis, label_arft

    analysis = generate_analysis(trajectory)          # Stage 1: trajectory -> analysis.md
    result   = label_arft(analysis, api_key="...")    # Stage 2: analysis.md -> ARFT labels

The two stages are deliberately separate calls rather than one. Stage 1 spawns a fresh
Claude Code session per trajectory — minutes of wall clock, and it needs the ``claude``
CLI installed and authenticated. Stage 2 is a single chat completion against any
OpenAI-compatible endpoint and takes seconds. Most users already have analyses, or want
to inspect and keep them, so folding the two together would hide the expensive half and
throw away the artifact that makes the labels auditable.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import httpx

from . import analysis_qa, classify as _classify, config, generate as _generate
from . import label_qa as _label_qa
from . import patterns as _patterns
from . import traj_tools

__all__ = ["generate_analysis", "label_arft", "pattern_info", "all_patterns"]


# --------------------------------------------------------------------------- taxonomy
def pattern_info(code: str) -> dict:
    """Describe one ARFT code, e.g. ``pattern_info("C.1")``.

    Raises KeyError for a code outside the 45-pattern space, rather than returning an
    empty dict — a typo in a code should not look like a pattern with no description.
    """
    name, definition, stage, pillar = _patterns.PATTERNS[code]
    return {"code": code, "name": name, "definition": definition, "stage": stage,
            "stage_name": _patterns.STAGES[stage], "pillar": pillar,
            "pillar_name": _patterns.PILLARS[pillar][0],
            "root_cause": _patterns.ROOT_CAUSE_WORD[pillar]}


def all_patterns() -> dict:
    """Every ARFT code -> its description, as returned by :func:`pattern_info`."""
    return {code: pattern_info(code) for code in _patterns.PATTERNS}


def _resolve_trajectory(source):
    """Accept a dict, a trajectory JSON, or a directory holding one.

    A directory is checked for the ``traj/*.json`` layout the batch CLI expects first,
    then for bare ``*.json`` at the top level. More than one match is an error rather
    than a silent pick: running them serially here would drop the concurrency, resume
    and QA-retry the batch CLI provides, and picking one arbitrarily would analyze a
    trajectory the caller did not name.
    """
    if not isinstance(source, (str, Path)):
        return dict(source), None

    path = Path(source)
    if path.is_dir():
        found = sorted((path / "traj").glob("*.json")) or sorted(path.glob("*.json"))
        if not found:
            raise FileNotFoundError(
                f"no trajectory JSON under {path} (looked in {path / 'traj'} and {path})")
        if len(found) > 1:
            raise ValueError(
                f"{path} holds {len(found)} trajectories. generate_analysis() does one at "
                f"a time; pass a single file, or use the batch CLI for the whole set:\n"
                f"    aaj-generate --run-dir {path} --concurrency 4 --resume")
        path = found[0]
    return json.loads(path.read_text()), path


# ---------------------------------------------------------------------------- stage 1
def generate_analysis(
    trajectory,
    *,
    task_id: str | None = None,
    retrieval_note: str | None = None,
    gold_note: str | None = None,
    model: str = "claude-opus-4-8",
    effort: str = "high",
    max_turns: int = 80,
    timeout: int = 3600,
    claude_bin: str | None = None,
    workspace: str | Path | None = None,
    keep_workspace: bool = False,
) -> dict:
    """Deep-dive one trajectory into a structured ``analysis.md``.

    Spawns a fresh headless Claude Code session with shell access, hands it the
    trajectory's artifacts plus the ONBOARDING framework and the depth exemplar, and
    returns the analysis it writes together with the QA gate's verdict.

    Args:
        trajectory: a trajectory record as a dict, a path to a JSON file holding one, or
            a directory containing one (``<dir>/traj/*.json`` or ``<dir>/*.json``). It
            needs a ``task_id`` and a log field ``traj_tools.detect_format`` knows
            (Claude Code stream-JSON, Gemini CLI NDJSON, or Codex CLI JSONL).
        task_id: overrides the record's own ``task_id``.
        retrieval_note: what the harness's retrieval tools really did — whether
            WebSearch/WebFetch performed real network I/O or were mocked. Optional: the
            default tells the analyst to work it out from the log and report which it
            concluded. Pass it only when you can state the truth, since a wrong
            assertion here manufactures findings in either direction.
        gold_note: whether ground-truth values are reachable. Optional; the default
            assumes none and grounds every numerical judgment in recomputation, unit and
            magnitude checks, and internal consistency.
        model, effort, max_turns, timeout: passed through to the session.
        claude_bin: path to the ``claude`` CLI; auto-detected when omitted.
        workspace: where to build the per-task workspace. A temporary directory is used
            and removed when omitted, unless ``keep_workspace`` is set.

    Returns:
        ``{"task_id", "analysis", "qa", "path", "workspace", "duration_s", "returncode"}``
        where ``analysis`` is the markdown and ``qa`` is
        :func:`autoresearcheval.analysis_qa.check`'s report. ``qa["ok"]`` being False
        means the analysis came back thinner than the framework's bar, not that the call
        failed — inspect ``qa["problems"]``.

    Raises:
        RuntimeError: the session produced no analysis at all.
    """
    record, traj_path = _resolve_trajectory(trajectory)
    tmp_json = None
    if traj_path is None:                       # given a dict: the workspace extractor
        tmp_json = tempfile.NamedTemporaryFile(  # reads from disk, so stage it there
            "w", suffix=".json", delete=False)
        json.dump(record, tmp_json)
        tmp_json.close()
        traj_path = Path(tmp_json.name)

    tid = task_id or record.get("task_id")
    if not tid:
        raise ValueError("trajectory has no task_id; pass task_id=...")

    reason = record.get("reason", "") or ""
    cat = next((c for c in _generate.KNOWN_CATEGORIES if reason.startswith(c)), "soft")
    t = {"task_id": tid, "reward": record.get("reward"), "reason": reason, "category": cat}

    owns_ws = workspace is None
    ws = Path(tempfile.mkdtemp(prefix=f"aaj-{tid}-")) if owns_ws else Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    out_dir = ws / "_analysis"
    out_dir.mkdir(exist_ok=True)

    # RETRIEVAL_NOTE/GOLD_NOTE are read by build_instruction off the module. Swap them
    # for this call only, so concurrent callers with different harnesses don't bleed
    # into each other any more than the module already allows.
    saved = (_generate.RETRIEVAL_NOTE, _generate.GOLD_NOTE)
    if retrieval_note is not None:
        _generate.RETRIEVAL_NOTE = retrieval_note
    if gold_note is not None:
        _generate.GOLD_NOTE = gold_note
    try:
        traj_tools.extract_workspace(str(traj_path), ws)
        shutil.copy(Path(__file__).resolve().parent / "traj_tools.py", ws / "traj_tools.py")
        (ws / "INSTRUCTION.md").write_text(
            _generate.build_instruction(t, "library", str(out_dir)))
    finally:
        _generate.RETRIEVAL_NOTE, _generate.GOLD_NOTE = saved

    binary = _generate.find_claude_bin(claude_bin)
    read_dirs = {str(config.onboarding().parent)}
    if config.exemplar().exists():
        read_dirs.add(str(config.exemplar().parent))
    cmd = [binary, "--print", "--verbose", "--output-format", "stream-json",
           "--permission-mode", "bypassPermissions", "--max-turns", str(max_turns),
           "--model", model, "--effort", effort,
           *[a for d in sorted(read_dirs) for a in ("--add-dir", d)],
           "-p", ("Read INSTRUCTION.md in this directory FIRST and follow it completely. "
                  "You are fully autonomous; do not ask questions. Produce the analysis.md "
                  "at the path INSTRUCTION.md specifies before stopping.")]
    env = os.environ.copy()
    env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    started = time.time()
    with open(ws / "session.log", "w") as lf:
        try:
            rc = subprocess.run(cmd, cwd=ws, env=env, stdin=subprocess.DEVNULL,
                                stdout=lf, stderr=subprocess.STDOUT,
                                timeout=timeout, text=True).returncode
        except subprocess.TimeoutExpired:
            rc = -1
    duration = round(time.time() - started, 1)

    md = out_dir / "analysis.md"
    if not md.exists():
        raise RuntimeError(
            f"no analysis produced for {tid} (returncode={rc}); session log at "
            f"{ws / 'session.log'}")
    report = analysis_qa.check(md, reason, ws)
    text = md.read_text()
    result = {"task_id": tid, "analysis": text, "qa": report, "path": str(md),
              "workspace": str(ws), "duration_s": duration, "returncode": rc}
    if tmp_json is not None:
        Path(tmp_json.name).unlink(missing_ok=True)
    if owns_ws and not keep_workspace:
        shutil.rmtree(ws, ignore_errors=True)
        result["workspace"] = None
        result["path"] = None
    return result


# ---------------------------------------------------------------------------- stage 2
def label_arft(
    analysis,
    *,
    api_key: str | None = None,
    model: str = "anthropic/claude-sonnet-5",
    base_url: str | None = None,
    task_id: str = "analysis",
    reasoning_tokens: int = 3000,
    max_tokens: int = 16000,
    timeout: int = 300,
    retries: int = 3,
) -> dict:
    """Classify one ``analysis.md`` against the 45 ARFT patterns.

    One chat completion. The model maps an already-completed analysis onto the taxonomy;
    it does not re-judge the trajectory.

    Args:
        analysis: the analysis markdown, or a path to it.
        api_key: falls back to ``ARFT_OPENROUTER_KEY``, then ``~/.openrouter_key``, then
            ``OPENROUTER_API_KEY``.
        model: any model id your endpoint serves.
        base_url: an OpenAI-compatible chat-completions URL. Defaults to
            ``AAJ_ENDPOINT`` if set, else OpenRouter.
        reasoning_tokens: the single biggest quality lever here. Measured against a
            hand-checked reference labelling, reasoning off gives ~40% recall and 3000
            gives ~80%. Do not lower it to save money; the labels stop being usable.

    Returns:
        ``{"task_id", "overall_severity", "summary", "hits", "partials",
        "failure_modes", "total_failures", "iron_rules_cited", "stage_notes",
        "uncovered", "qa", "usage"}``. Each hit and partial carries its ``code``,
        ``name``, ``stage``, ``pillar``, ``root_cause``, ``confidence``, ``evidence``
        and ``why``. ``failure_modes`` maps every code the analysis established to True.
        ``qa["ok"]`` reports the schema-and-polarity gate; False means the label set is
        malformed or cites only the fair-credit section, not that the call failed.

    Raises:
        RuntimeError: the endpoint never returned parseable JSON.
    """
    text = Path(analysis).read_text() if (
        isinstance(analysis, (str, Path)) and len(str(analysis)) < 4096
        and Path(analysis).exists()) else str(analysis)

    key = api_key or _classify.load_key()
    endpoint = base_url or _classify.ENDPOINT
    guide = config.arft_guide().read_text()
    prompt = _classify.build_prompt(
        guide, text, {"task_id": task_id, "source_set": "unknown"}, "library")

    saved_endpoint = _classify.ENDPOINT
    _classify.ENDPOINT = endpoint
    doc = err = None
    usage: dict = {}
    room = max_tokens
    try:
        with httpx.Client() as client:
            for attempt in range(retries + 1):
                doc, usage, err, infra = _classify.call_model(
                    client, key, model, prompt, room, timeout, reasoning_tokens)
                if doc is not None or attempt == retries:
                    break
                if infra == "truncated":
                    room = min(int(room * 1.8), 64000)
                    continue
                if not infra:
                    break
                time.sleep(min(2 ** attempt * 2, 30))
    finally:
        _classify.ENDPOINT = saved_endpoint

    if doc is None:
        raise RuntimeError(f"classification failed: {err}")

    doc.setdefault("task_id", task_id)

    def enrich(items):
        out = []
        for it in items or []:
            code = it.get("code")
            row = dict(it)
            if code in _patterns.PATTERNS:
                row.update({k: v for k, v in pattern_info(code).items()
                            if k in ("name", "stage", "pillar", "root_cause")})
            out.append(row)
        return out

    doc["hits"] = enrich(doc.get("hits"))
    doc["partials"] = enrich(doc.get("partials"))
    doc["failure_modes"] = {h["code"]: True for h in doc["hits"] if h.get("code")}
    doc["total_failures"] = len(doc["hits"])

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(doc, fh, ensure_ascii=False)
        gate_path = fh.name
    try:
        doc["qa"] = _label_qa.check(gate_path)
    finally:
        Path(gate_path).unlink(missing_ok=True)
    doc["usage"] = usage
    return doc
