#!/usr/bin/env python3
"""
traj_tools.py — shared helpers for deep-diving a single agentic-research trajectory
JSON from the realsearch runs.

Two uses:
  1) Driver-side: extract_workspace(json_path, ws) writes the per-task input files.
  2) Session-side: the analysis session imports this to parse the embedded agent log,
     pair tool calls with their results, and RECONSTRUCT the final delivered code
     (full-content write as base + subsequent edits replayed) — ONBOARDING Iron Rule 1,
     "the final delivered artifact is authoritative". This avoids each session
     re-deriving the plumbing.

THREE harnesses are supported, because subset3 spans three CLIs:

  claude   `claude_log`   stream-json JSONL                       6 of 8 subset3 runs
  gemini   `gemini_log`   flattened text + an appended RAW NDJSON tail
  codex    `codex_log`    Codex CLI plain-text transcript

The gemini harness appends its raw stream-json after a `----- RAW stream-json -----`
line (present in 69/70 subset3 gemini trajectories), so gemini is structurally
equivalent to claude, not a text-scraping problem — we parse that tail and fall back
to scraping the flattened prose only when the marker is absent.

Codex is the genuinely lossy one: it has no structured tool events, no search tool,
and writes most files through inline `open(...,'w')` / heredocs that carry no
recoverable "final version" event. Where a capability cannot be honestly provided,
the function returns a marker string starting with NO_RECON_PREFIX rather than
something that looks right but is not — see reconstruct_file().

Every function is read-only w.r.t. the source JSON.
"""
import json
import re
from pathlib import Path

NO_RECON_PREFIX = "<<TRAJ_TOOLS_NO_RECONSTRUCTION"
PARTIAL_PREFIX = "<<TRAJ_TOOLS_PARTIAL_RECONSTRUCTION"

LOG_KEYS = (("claude", "claude_log"), ("gemini", "gemini_log"), ("codex", "codex_log"))
LOG_KEY = dict(LOG_KEYS)
LOG_FILENAME = {"claude": "claude_log.jsonl", "gemini": "agent_log.txt",
                "codex": "agent_log.txt", "codex2": "agent_log.jsonl"}
AGENT_CLI = {"claude": "claude-code", "gemini": "gemini-cli", "codex": "codex-cli",
             "codex2": "codex-cli (native JSONL, NatureBench harness)"}

# Native tool names per harness, so the analysis session can quote real log lines.
TOOL_ALIASES = {
    "shell":      {"claude": "Bash",      "gemini": "run_shell_command", "codex": "exec",
                   "codex2": "command_execution"},
    "write":      {"claude": "Write",     "gemini": "write_file",        "codex": "apply_patch / shell redirect",
                   "codex2": "shell heredoc (see agent_code/, workspace/ is ground truth)"},
    "edit":       {"claude": "Edit",      "gemini": "replace",           "codex": "apply_patch",
                   "codex2": "apply_patch (rare) / shell redirect"},
    "read":       {"claude": "Read",      "gemini": "read_file",         "codex": "(shell cat)",
                   "codex2": "(shell cat, inside command_execution)"},
    "web_search": {"claude": "WebSearch", "gemini": "google_web_search", "codex": "(none)",
                   "codex2": "(none)"},
    "web_fetch":  {"claude": "WebFetch",  "gemini": "web_fetch",         "codex": "(shell curl)",
                   "codex2": "(shell curl, inside command_execution)"},
}


# ----------------------------------------------------------------------------- load
def load_traj(json_path):
    with open(json_path) as f:
        return json.load(f)


def detect_format(traj):
    """Which harness produced this trajectory: claude | gemini | codex."""
    if isinstance(traj, (str, Path)):
        traj = load_traj(traj)
    for fmt, key in LOG_KEYS:
        if traj.get(key):
            return fmt
    for fmt, key in LOG_KEYS:          # empty-but-present still identifies the run
        if key in traj:
            return fmt
    return "claude"                    # legacy default; preserves prior behavior


def detect_log_format(log_path):
    """Sniff the harness from an already-extracted log file.

    NatureBench (nbe_part1 and siblings) container logs open with a docker/CUDA
    banner (plain text, no leading `{`) before the real NDJSON starts — skip
    leading non-JSON lines when looking for the first parseable object, so this
    still works whether or not the banner was stripped at extraction time.
    """
    head = open(log_path, errors="ignore").read(16384)
    first_obj = None
    for ln in head.splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            first_obj = json.loads(ln)
            break
        except Exception:
            continue
    if first_obj is not None:
        o = first_obj
        if "thread_id" in o or o.get("type") in (
                "thread.started", "turn.started", "turn.completed",
                "item.started", "item.completed"):
            return "codex2"
        if "message" in o or o.get("type") in ("system", "assistant", "user", "result"):
            return "claude"
        if "tool_name" in o or "tool_id" in o or "session_id" in o:
            return "gemini"
    if head.startswith("Reading additional input from stdin") or "OpenAI Codex v" in head[:400] \
       or re.search(r"(?m)^exec\n(?:/bin/)?bash -lc? ", head):
        return "codex"
    if re.search(r"(?m)^(?:\S+ )?Tool call: \S+ \{", head) or "----- RAW stream-json -----" in head:
        return "gemini"
    return "claude"


def strip_container_banner(text):
    """Drop the leading docker/CUDA/NGC banner NatureBench containers print to
    stdout before the first NDJSON line, so downstream parsers that assume the
    file already starts with '{' (gemini's bare-jsonl fast path, codex2) work
    unchanged. A no-op on logs that never had a banner."""
    lines = text.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("{"):
            return "".join(lines[i:])
    return text


def extract_workspace(json_path, ws_dir, gemini_stream=None):
    """Write decision.json, report.md, the agent log, and meta.json into ws_dir.

    gemini_stream: optional path to the structured gemini_stream.jsonl from the
    regime framework, used when the embedded log has no RAW tail.
    """
    d = load_traj(json_path)
    ws = Path(ws_dir)
    ws.mkdir(parents=True, exist_ok=True)
    fmt = detect_format(d)
    (ws / "report.md").write_text(d.get("report_md", "") or "")

    raw = d.get(LOG_KEY[fmt], "") or ""
    log_name = LOG_FILENAME[fmt]
    if fmt == "gemini" and "----- RAW stream-json -----" not in raw \
            and gemini_stream and Path(gemini_stream).exists():
        raw = (raw + "\n----- RAW stream-json -----\n"
               + Path(gemini_stream).read_text(errors="ignore"))
    (ws / log_name).write_text(raw)

    with open(ws / "decision.json", "w") as f:
        json.dump(d.get("decision", {}), f, indent=2, ensure_ascii=False)
    meta = {
        "task_id": d.get("task_id"),
        "reward": d.get("reward"),
        "reason": d.get("reason"),
        "dims": d.get("dims"),
        "wall_clock_s": d.get("wall_clock_s"),
        "resource_monitor": d.get("resource_monitor"),
        "observable_text": (d.get("decision", {}) or {}).get("observable"),
        "log_format": fmt,
        "agent_cli": AGENT_CLI[fmt],
        "log_file": log_name,
        "log_chars": len(raw),
    }
    with open(ws / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def _short(s, n=160):
    return (s or "")[:n].replace("\n", " ")


# ============================================================== dialect: claude
def _iter_lines(log_path):
    for line in open(log_path, errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


def _claude_timeline(log_path):
    out, idx = [], 0
    for d in _iter_lines(log_path):
        if d.get("type") != "assistant":
            continue
        for b in d.get("message", {}).get("content", []):
            if b.get("type") != "tool_use":
                continue
            name = b.get("name", "")
            inp = b.get("input", {}) or {}
            if name == "Bash":
                s = (inp.get("command", "") or "")[:160]
            elif name == "WebSearch":
                s = (inp.get("query", "") or "")[:160]
            elif name == "WebFetch":
                s = (inp.get("url", "") or "")[:160]
            elif name in ("Write", "Edit", "Read"):
                s = (inp.get("file_path", "") or "")[:160]
            else:
                s = json.dumps(inp)[:160]
            out.append((idx, name, s.replace("\n", " ")))
            idx += 1
    return out


def _claude_pairs(log_path):
    uses, results = {}, {}
    for d in _iter_lines(log_path):
        t = d.get("type")
        if t == "assistant":
            for b in d.get("message", {}).get("content", []):
                if b.get("type") == "tool_use":
                    uses[b["id"]] = (b.get("name", ""), b.get("input", {}) or {})
        elif t == "user":
            cont = d.get("message", {}).get("content")
            if isinstance(cont, list):
                for b in cont:
                    if b.get("type") == "tool_result":
                        c = b.get("content", "")
                        if isinstance(c, list):
                            c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                        results[b.get("tool_use_id")] = c
    return {uid: (nm, inp, results.get(uid, "")) for uid, (nm, inp) in uses.items()}


def _claude_assistant_text(log_path):
    chunks = []
    for d in _iter_lines(log_path):
        if d.get("type") != "assistant":
            continue
        for b in d.get("message", {}).get("content", []):
            if b.get("type") == "text":
                chunks.append(b.get("text", ""))
    return "\n".join(chunks)


def _claude_write_events(log_path):
    seq = []
    for d in _iter_lines(log_path):
        if d.get("type") != "assistant":
            continue
        for b in d.get("message", {}).get("content", []):
            if b.get("type") != "tool_use" or b.get("name") not in ("Write", "Edit"):
                continue
            inp = b.get("input", {}) or {}
            fp = inp.get("file_path", "")
            if b["name"] == "Write":
                seq.append(("W", fp, inp.get("content", ""), "Write"))
            else:
                seq.append(("E", fp, inp.get("old_string", ""),
                            inp.get("new_string", ""), bool(inp.get("replace_all")), "Edit"))
    return seq


def _claude_searches(log_path):
    out = []
    for _uid, (nm, inp, res) in _claude_pairs(log_path).items():
        if nm in ("WebSearch", "WebFetch"):
            out.append((nm, inp.get("query") or inp.get("url"), res))
    return out


# ============================================================== dialect: gemini
_G_RAW = re.compile(r"(?m)^----- RAW stream-json -----$")
_G_CALL = re.compile(r"(?m)^(?:\S+ )?Tool call: (\S+) (\{.*)$")
_G_RES = re.compile(r"(?m)^Tool result \(([^)]*)\): ?")
_G_SHORT = {"run_shell_command": "command", "google_web_search": "query",
            "web_fetch": "prompt", "write_file": "file_path", "replace": "file_path",
            "read_file": "file_path", "list_directory": "dir_path",
            "glob": "pattern", "grep_search": "pattern", "update_topic": "title"}


def _gemini_events(log_path):
    """Normalized event list [(kind, name, params, result, uid)].

    Prefers the RAW stream-json tail the gemini harness appends; falls back to
    scraping the flattened `Tool call:` / `Tool result (...)` prose.
    """
    text = Path(log_path).read_text(errors="ignore")
    marks = list(_G_RAW.finditer(text))
    lines = None
    if marks:
        lines = text[marks[-1].end():]
    elif text.lstrip().startswith("{"):
        lines = text                       # a bare gemini_stream.jsonl
    if lines is not None:
        objs = []
        for ln in lines.splitlines():
            ln = ln.strip()
            if not ln.startswith("{"):
                continue
            try:
                objs.append(json.loads(ln))
            except Exception:
                continue
        if objs:
            return _gemini_events_from_objs(objs)
    return _gemini_events_from_text(text)


def _gemini_events_from_objs(objs):
    results = {}
    for o in objs:
        if o.get("type") == "tool_result":
            out = o.get("output", "")
            if not isinstance(out, str):
                out = json.dumps(out, ensure_ascii=False)
            results[o.get("tool_id")] = out
    ev = []
    for o in objs:
        t = o.get("type")
        if t == "tool_use":
            uid = o.get("tool_id")
            ev.append(("tool", o.get("tool_name") or "", o.get("parameters") or {},
                       results.get(uid, ""), uid))
        elif t == "message" and o.get("role") == "assistant":
            c = o.get("content", "")
            if isinstance(c, str) and c:
                ev.append(("text", "", {}, c, None))
    return ev


def _gemini_events_from_text(text):
    marks = [(m.start(), m) for m in _G_CALL.finditer(text)]
    ev = []
    for i, (pos, m) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        chunk = text[pos:end]
        try:
            params = json.loads(m.group(2))
        except Exception:
            params = {"_raw": _short(m.group(2), 400)}
        if not isinstance(params, dict):
            params = {"_raw": str(params)}
        rm = _G_RES.search(chunk)
        result = chunk[rm.end():].strip() if rm else ""
        ev.append(("tool", m.group(1), params, result, f"gem-{i}"))
    return ev


def _gemini_timeline(log_path):
    out, idx = [], 0
    for kind, name, params, _res, _uid in _gemini_events(log_path):
        if kind != "tool":
            continue
        key = _G_SHORT.get(name)
        s = params.get(key, "") if key else ""
        if not s:
            s = json.dumps(params, ensure_ascii=False)
        out.append((idx, name, _short(s)))
        idx += 1
    return out


def _gemini_pairs(log_path):
    out = {}
    for i, (kind, name, params, res, uid) in enumerate(_gemini_events(log_path)):
        if kind != "tool":
            continue
        out[uid or f"gem-{i}"] = (name, params, res)
    return out


def _gemini_assistant_text(log_path):
    """Assistant messages arrive as deltas — join a run with '', break on tool calls."""
    chunks, buf = [], []
    for kind, _name, _p, res, _uid in _gemini_events(log_path):
        if kind == "text":
            buf.append(res)
        elif buf:
            chunks.append("".join(buf))
            buf = []
    if buf:
        chunks.append("".join(buf))
    return "\n".join(chunks)


def _gemini_write_events(log_path):
    seq = []
    for kind, name, p, _res, _uid in _gemini_events(log_path):
        if kind != "tool":
            continue
        if name == "write_file":
            seq.append(("W", p.get("file_path", "") or "", p.get("content", ""), "write_file"))
        elif name == "replace":
            seq.append(("E", p.get("file_path", "") or "", p.get("old_string", ""),
                        p.get("new_string", ""), bool(p.get("allow_multiple")), "replace"))
    return seq


_URL = re.compile(r"https?://[^\s\"'\\)>]+")


def _gemini_searches(log_path):
    out = []
    for kind, name, p, res, _uid in _gemini_events(log_path):
        if kind != "tool":
            continue
        if name == "google_web_search":
            out.append((name, p.get("query", ""), res))
        elif name == "web_fetch":
            prompt = p.get("prompt", "") or ""
            m = _URL.search(prompt)
            out.append((name, m.group(0) if m else _short(prompt, 200), res))
    return out


# =============================================================== dialect: codex
# exec
# /bin/bash -lc '<cmd>' in /workspace
#  succeeded in 51902ms:        |    exited 2 in 4175ms:
# <stdout/stderr>
#
# apply patch
# patch: completed
# <path>
# <unified diff, printed twice: preview + applied>
_C_SPLIT = re.compile(r"(?m)^(?:exec\n(?=(?:/bin/)?bash -lc? )|apply patch\n(?=patch: ))")
_C_RES = re.compile(r"(?m)^ (succeeded|exited \d+|failed[^\n]*|timed out[^\n]*) in ([\d.]+)(ms|s):$")
_C_CMD = re.compile(r"(?P<cmd>.*?) in (?P<cwd>/\S*)\n", re.S)
_C_DIFF_HDR = re.compile(r"(?m)^diff --git a/(.+?) b/(.+?)$")
_C_HUNK = re.compile(r"(?m)^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_C_NET = re.compile(r"\b(curl|wget|requests\.(?:get|post)|urllib|httpx|http\.client)\b")
_C_HEREDOC = re.compile(
    r"(?:cat|tee)\s+(?:-a\s+)?>+\s*(?P<path>[^\s<>|;&]+)\s*<<\s*[-]?\s*'?\"?"
    r"(?P<tag>[A-Za-z_][A-Za-z0-9_]*)'?\"?\s*\n(?P<body>.*?)\n[ \t]*(?P=tag)\b", re.S)
_C_SH_WRITE = [
    ("heredoc", re.compile(r"(?:cat|tee)\s+(?:-a\s+)?>+\s*(\S+)\s*<<")),
    ("py-open", re.compile(r"""open\(\s*['"]([^'"]+)['"]\s*,\s*['"][wa]""")),
    ("py-write_text", re.compile(r"""Path\(\s*['"]([^'"]+)['"]\s*\)\s*\.\s*write_text""")),
    ("tee", re.compile(r"\|\s*tee\s+(\S+)")),
    ("redirect", re.compile(r"(?<![>\d])>\s*([^\s>|&;]+\.\w{1,5})\b")),
]
# `> 0.99` inside a printf/format string is not a redirection. Only accept
# candidates that actually look like paths, or the file list fills with junk
# like "0.9", "r=%.3f", "t={:.3f".
_PATHY_EXT = re.compile(r"\.(py|json|jsonl|md|txt|csv|tsv|npy|npz|log|sh|ya?ml|toml|"
                        r"png|pdf|h5|hdf5|pkl|parquet|ipynb|c|cpp|f90|R)$", re.I)


def _looks_like_path(p):
    if not p or len(p) > 300:
        return False
    if any(ch in p for ch in "={}%()\"'`$*?"):
        return False
    if p[0].isdigit() or p.startswith(".") or p.startswith("-"):
        return False
    return "/" in p or bool(_PATHY_EXT.search(p))


def _codex_blocks(log_path):
    """Ordered blocks: {idx, kind, cmd, cwd, status, dur, output, raw}."""
    text = Path(log_path).read_text(errors="ignore")
    starts = [m.start() for m in _C_SPLIT.finditer(text)]
    blocks = []
    for i, pos in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        seg = text[pos:end]
        kind = "exec" if seg.startswith("exec\n") else "patch"
        body = seg.split("\n", 1)[1] if "\n" in seg else ""
        res = list(_C_RES.finditer(body))
        rm = res[0] if res else None
        if kind == "exec":
            cm = _C_CMD.match(body)
            # command must terminate before the result line, else fall back to the
            # first line (a duplicated diff can otherwise be swallowed into the cmd)
            if cm and (not rm or cm.end() <= rm.start()):
                cmd, cwd = cm.group("cmd").strip(), cm.group("cwd")
            else:
                cmd, cwd = body.split("\n", 1)[0].strip(), ""
            out = body[rm.end():] if rm else ""
            # Codex streams asynchronously: a command whose `exec` header was dropped
            # can land its result line inside this block. Those extra results (and
            # their output) stay in `output`; we count them so nothing is silently
            # attributed to the wrong command.
            blocks.append({"idx": i, "kind": "exec", "cmd": cmd, "cwd": cwd,
                           "status": rm.group(1) if rm else "no_result",
                           "dur": (rm.group(2) + rm.group(3)) if rm else "",
                           "extra_results": max(len(res) - 1, 0),
                           "output": out.strip("\n"), "raw": seg})
        else:
            paths = []
            for m in _C_DIFF_HDR.finditer(body):
                p = m.group(2)
                if p not in paths:
                    paths.append(p)
            blocks.append({"idx": i, "kind": "patch", "cmd": "", "cwd": "",
                           "status": "completed" if "patch: completed" in body else "unknown",
                           "dur": "", "paths": paths, "output": body, "raw": seg})
    return blocks


def _codex_timeline(log_path):
    out = []
    for b in _codex_blocks(log_path):
        if b["kind"] == "exec":
            out.append((b["idx"], "exec", _short(b["cmd"])))
        else:
            out.append((b["idx"], "apply_patch", _short(", ".join(b.get("paths", [])) or "(no path)")))
    return out


def _codex_pairs(log_path):
    out = {}
    for b in _codex_blocks(log_path):
        if b["kind"] == "exec":
            out[f"c{b['idx']:04d}"] = ("exec", {"command": b["cmd"], "cwd": b["cwd"],
                                                "_status": b["status"], "_dur": b["dur"]},
                                       b["output"])
        else:
            out[f"c{b['idx']:04d}"] = ("apply_patch", {"paths": b.get("paths", []),
                                                       "_status": b["status"]}, b["output"])
    return out


_C_SUMMARY = re.compile(r"(?m)^\*\*(.+?)\*\*\n(.*?)(?=\n\*\*|\nexec\n|\napply patch\n|\Z)", re.S)


def _codex_assistant_text(log_path):
    """Heuristic: codex emits reasoning as '**Header**' + paragraph, inside exec blocks."""
    text = Path(log_path).read_text(errors="ignore")
    return "\n\n".join(f"**{m.group(1)}**\n{m.group(2).strip()}" for m in _C_SUMMARY.finditer(text))


def _codex_write_events(log_path):
    """('P', path, diff_body, 'apply_patch') for patches; ('S', path, '', channel)
    for shell/python writes whose content is not recoverable as a final version."""
    seq = []
    for b in _codex_blocks(log_path):
        if b["kind"] == "patch":
            for p in b.get("paths", []):
                seq.append(("P", p, b["output"], "apply_patch"))
        else:
            # heredocs carry the file's full content inline -> a real write event
            hd = set()
            for m in _C_HEREDOC.finditer(b["cmd"]):
                p = m.group("path")
                if _looks_like_path(p):
                    hd.add(p)
                    seq.append(("W", p, m.group("body"), "shell:heredoc"))
            for chan, rx in _C_SH_WRITE:
                for m in rx.finditer(b["cmd"]):
                    p = m.group(1)
                    if _looks_like_path(p) and p not in hd:
                        seq.append(("S", p, "", f"shell:{chan}"))
    return seq


def _codex_searches(log_path):
    out = []
    for b in _codex_blocks(log_path):
        if b["kind"] != "exec" or not _C_NET.search(b["cmd"]):
            continue
        m = _URL.search(b["cmd"])
        out.append(("exec:net", m.group(0) if m else _short(b["cmd"], 300), b["output"]))
    return out


# ---- codex apply_patch replay -------------------------------------------------
def _codex_diff_sections(diff_body):
    """[(path, section_text)] keeping the LAST section per path (codex prints each
    diff twice — preview then applied — and they are byte-identical)."""
    hdrs = [(m.start(), m.group(2)) for m in _C_DIFF_HDR.finditer(diff_body)]
    secs = {}
    for i, (pos, path) in enumerate(hdrs):
        end = hdrs[i + 1][0] if i + 1 < len(hdrs) else len(diff_body)
        secs[path] = diff_body[pos:end]
    return list(secs.items())


def _apply_unified_diff(lines, section):
    """Apply one file's hunks to `lines` (list of str). Returns (lines, mismatches)."""
    if re.search(r"(?m)^--- /dev/null$", section):
        lines = []
    hunks = []
    marks = list(_C_HUNK.finditer(section))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(section)
        body = section[m.end():end].split("\n")[1:]
        old_start = int(m.group(1))
        old_count = int(m.group(2)) if m.group(2) is not None else 1
        hunks.append((old_start, old_count, body))
    mismatch = 0
    for old_start, old_count, body in reversed(hunks):     # reverse -> no offset bookkeeping
        before, after = [], []
        for ln in body:
            if ln.startswith("\\"):                        # "\ No newline at end of file"
                continue
            if ln.startswith("-"):
                before.append(ln[1:])
            elif ln.startswith("+"):
                after.append(ln[1:])
            elif ln.startswith(" ") or ln == "":
                before.append(ln[1:] if ln else "")
                after.append(ln[1:] if ln else "")
        i0 = max(old_start - 1, 0)
        cur = lines[i0:i0 + old_count]
        if old_count and cur != before:
            mismatch += 1
        lines[i0:i0 + old_count] = after
    return lines, mismatch


def _codex_reconstruct(log_path, name_substr, evs):
    """Replay a codex write chain for one file: heredoc full writes as bases,
    apply_patch unified diffs replayed on top. Returns (text, confidence, info)."""
    lines, mism, patches, shell_after, created = None, 0, 0, 0, False
    for e in evs:
        if e[0] == "W":                       # heredoc — full content
            lines, mism, patches, shell_after = e[2].split("\n"), 0, 0, 0
            created = True
        elif e[0] == "P":                     # unified diff
            for path, sec in _codex_diff_sections(e[2]):
                if name_substr not in path:
                    continue
                if re.search(r"(?m)^--- /dev/null$", sec):
                    created = True
                if lines is None:
                    lines = []
                lines, m = _apply_unified_diff(lines, sec)
                mism += m
                patches += 1
        else:                                 # 'S' — runtime write, content unknowable
            shell_after += 1
    info = {"patches": patches, "created": created, "mismatches": mism,
            "opaque_writes_after": shell_after}
    if lines is None:
        return None, "unavailable", info
    conf = "exact" if (created and mism == 0 and shell_after == 0) else "partial"
    return "\n".join(lines), conf, info


# ============================================================= dialect: codex2
# NatureBench's Codex CLI emits *native* structured JSONL (thread.started /
# turn.* / item.started+item.completed), unlike subset3's plain-text Codex
# transcript ("codex" above) — a different schema, not just a naming variant,
# so it gets its own fmt tag rather than sharing _codex_* implementations.
def _codex2_events(log_path):
    """Banner-tolerant JSONL reader: skip any leading non-'{' lines (docker/CUDA
    banner), then yield every parseable object in order."""
    text = Path(log_path).read_text(errors="ignore")
    text = strip_container_banner(text)
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            yield json.loads(ln)
        except Exception:
            continue


def _codex2_timeline(log_path):
    out, idx = [], 0
    for o in _codex2_events(log_path):
        it = o.get("item") or {}
        t = it.get("type")
        if o.get("type") != "item.completed" or t not in (
                "reasoning", "agent_message", "command_execution", "todo_list", "error"):
            continue
        if t == "command_execution":
            s = _short(it.get("command", ""))
        elif t in ("reasoning", "agent_message"):
            s = _short(it.get("text", ""))
        elif t == "error":
            s = _short(it.get("message", ""))
        else:
            s = _short(json.dumps(it, ensure_ascii=False))
        out.append((idx, t, s))
        idx += 1
    return out


def _codex2_pairs(log_path):
    """item id -> (item_type, input_dict, result_text). command_execution pairs
    its item.started (command) with its item.completed (aggregated_output);
    reasoning/agent_message/todo_list only ever appear as item.completed."""
    started = {}
    out = {}
    for o in _codex2_events(log_path):
        it = o.get("item") or {}
        iid = it.get("id")
        t = it.get("type")
        if o.get("type") == "item.started" and t == "command_execution":
            started[iid] = it.get("command", "")
        elif o.get("type") == "item.completed":
            if t == "command_execution":
                out[iid] = ("command_execution",
                            {"command": it.get("command", started.get(iid, ""))},
                            it.get("aggregated_output", ""))
            elif t in ("reasoning", "agent_message"):
                out[iid] = (t, {}, it.get("text", ""))
            elif t == "error":
                out[iid] = ("error", {}, it.get("message", ""))
            elif t == "todo_list":
                out[iid] = ("todo_list", {}, json.dumps(it.get("items", it), ensure_ascii=False))
    return out


def _codex2_assistant_text(log_path):
    chunks = []
    for o in _codex2_events(log_path):
        it = o.get("item") or {}
        if o.get("type") == "item.completed" and it.get("type") in ("reasoning", "agent_message"):
            txt = it.get("text", "")
            if txt:
                chunks.append(txt)
    return "\n\n".join(chunks)


def _codex2_write_events(log_path):
    """Best-effort, from the same shell-command text subset3's codex parser
    scans (heredocs, redirects) — 'apply_patch' calls inside command_execution
    are NOT diff-replayed here (unlike subset3's plain-text codex): nbe_part1
    ships the literal final workspace/ on disk, which is strictly better
    ground truth than a replayed diff, so reconstruction is intentionally not
    attempted for this format — see reconstruct_file()'s codex2 branch."""
    seq = []
    for o in _codex2_events(log_path):
        it = o.get("item") or {}
        if o.get("type") != "item.completed" or it.get("type") != "command_execution":
            continue
        cmd = it.get("command", "") or ""
        hd = set()
        for m in _C_HEREDOC.finditer(cmd):
            p = m.group("path")
            if _looks_like_path(p):
                hd.add(p)
                seq.append(("W", p, m.group("body"), "shell:heredoc"))
        for chan, rx in _C_SH_WRITE:
            for m in rx.finditer(cmd):
                p = m.group(1)
                if _looks_like_path(p) and p not in hd:
                    seq.append(("S", p, "", f"shell:{chan}"))
    return seq


def _codex2_searches(log_path):
    out = []
    for o in _codex2_events(log_path):
        it = o.get("item") or {}
        if o.get("type") != "item.completed" or it.get("type") != "command_execution":
            continue
        cmd = it.get("command", "") or ""
        if not _C_NET.search(cmd):
            continue
        m = _URL.search(cmd)
        out.append(("exec:net", m.group(0) if m else _short(cmd, 300), it.get("aggregated_output", "")))
    return out


# ================================================================== dispatch table
_DISPATCH = {
    "claude": (_claude_timeline, _claude_pairs, _claude_assistant_text,
               _claude_write_events, _claude_searches),
    "gemini": (_gemini_timeline, _gemini_pairs, _gemini_assistant_text,
               _gemini_write_events, _gemini_searches),
    "codex":  (_codex_timeline, _codex_pairs, _codex_assistant_text,
               _codex_write_events, _codex_searches),
    "codex2": (_codex2_timeline, _codex2_pairs, _codex2_assistant_text,
               _codex2_write_events, _codex2_searches),
}


# =============================================================== public interface
def parse_timeline(log_path, fmt=None):
    """List of (idx, tool_name, short_input) for every tool call, in order."""
    return _DISPATCH[fmt or detect_log_format(log_path)][0](log_path)


def pair_tool_results(log_path, fmt=None):
    """Map call_id -> (tool_name, input, result_text).

    codex ids are positional (`c0000`) — its transcript carries no tool_use_id.
    """
    return _DISPATCH[fmt or detect_log_format(log_path)][1](log_path)


def assistant_text(log_path, fmt=None):
    """The model's own narration/thinking, concatenated. Heuristic on codex."""
    return _DISPATCH[fmt or detect_log_format(log_path)][2](log_path)


def write_events(log_path, fmt=None):
    """Ordered file-mutation events, tagged by channel:
    ('W', path, content, channel)            full-content write   (claude/gemini)
    ('E', path, old, new, replace_all, chan) in-place edit        (claude/gemini)
    ('P', path, diff_body, 'apply_patch')    unified diff         (codex)
    ('S', path, '', 'shell:...')             shell/python write, content unrecoverable
    """
    return _DISPATCH[fmt or detect_log_format(log_path)][3](log_path)


def search_returns(log_path, fmt=None):
    """[(kind, query_or_url, result_text)] for every retrieval call.

    codex has NO search tool in its transcript — its only observable retrieval is
    network I/O from exec, reported as kind='exec:net'. Do not read that as a
    search-tool transcript.
    """
    return _DISPATCH[fmt or detect_log_format(log_path)][4](log_path)


def written_file_paths(log_path, fmt=None):
    """All distinct file paths ever created or modified by the agent."""
    seen = []
    for ev in write_events(log_path, fmt):
        fp = ev[1]
        if fp and fp not in seen:
            seen.append(fp)
    return seen


def written_file_paths_detail(log_path, fmt=None):
    """[(path, channel)] — same paths, with provenance."""
    out, seen = [], set()
    for ev in write_events(log_path, fmt):
        fp, chan = ev[1], ev[-1]
        if fp and (fp, chan) not in seen:
            seen.add((fp, chan))
            out.append((fp, chan))
    return out


def reconstruct_file_info(log_path, name_substr, fmt=None):
    """Reconstruct the FINAL delivered version of a file whose path contains
    name_substr (ONBOARDING Iron Rule 1 — the final delivered artifact is authoritative).

    Returns {"text", "confidence", "channels", "blocks"} where confidence is
    exact | partial | unavailable | absent.
    """
    f = fmt or detect_log_format(log_path)
    evs = [e for e in write_events(log_path, f) if name_substr in (e[1] or "")]
    if not evs:
        return {"text": "", "confidence": "absent", "channels": [], "blocks": []}
    channels = sorted({e[-1] for e in evs})

    if f == "codex":
        text, conf, info = _codex_reconstruct(log_path, name_substr, evs)
        blocks = [b["idx"] for b in _codex_blocks(log_path)
                  if name_substr in b.get("raw", "")]
        if text is None:
            return {"text": "", "confidence": "unavailable", "channels": channels,
                    "blocks": blocks, "info": info}
        return {"text": text, "confidence": conf, "channels": channels,
                "blocks": blocks, "info": info}

    base, edits = "", []
    for e in evs:
        if e[0] == "W":
            base, edits = e[2], []
        elif e[0] == "E":
            edits.append(e)
    if not base:
        return {"text": "", "confidence": "unavailable", "channels": channels, "blocks": []}
    txt, missing = base, 0
    for _k, _fp, old, new, ra, _chan in edits:
        if old and old not in txt:
            missing += 1
            continue
        txt = txt.replace(old, new) if ra else txt.replace(old, new, 1)
    return {"text": txt, "confidence": "exact" if missing == 0 else "partial",
            "channels": channels, "blocks": [], "info": {"unmatched_edits": missing}}


def reconstruct_file(log_path, name_substr, fmt=None):
    """Thin wrapper over reconstruct_file_info returning a single string.

    On `unavailable` it returns a greppable marker, NOT '' — an empty string would
    read as "the agent never wrote this file", which in an analysis is a false
    finding. On `partial` the text is prefixed with a warning block.
    """
    r = reconstruct_file_info(log_path, name_substr, fmt)
    c = r["confidence"]
    if c == "absent":
        return ""
    if c == "unavailable":
        f = fmt or detect_log_format(log_path)
        action = ("read agent_code/ — it IS the container's real workspace/ at the run's "
                  "end, strictly better ground truth than any log replay could give you"
                  if f == "codex2" else
                  "read the pre-extracted code under agent_code/ if present, or read "
                  "the raw log at those blocks yourself")
        return (f"{NO_RECON_PREFIX} format={f} "
                f"name={name_substr!r}\n"
                f"reason=written via shell redirection / inline python, not through a "
                f"recoverable full-content write\n"
                f"channels_seen={', '.join(r['channels']) or 'none'}\n"
                f"raw_log_blocks={r.get('blocks', [])}\n"
                f"ACTION: {action}. Do NOT claim a reconstructed final version.>>")
    if c == "partial":
        return (f"{PARTIAL_PREFIX} info={r.get('info', {})}\n"
                f"The text below is a best-effort reconstruction; part of the edit chain "
                f"could not be replayed. Cross-check against agent_code/ before drawing "
                f"conclusions.>>\n" + r["text"])
    return r["text"]


# ------------------------------------------------------------------------- CLI use
def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Trajectory extraction / inspection helper")
    ap.add_argument("cmd", choices=["extract", "timeline", "files", "reconstruct",
                                    "searches", "blocks", "text", "format"])
    ap.add_argument("path", help="traj JSON (extract) or the extracted agent log (others)")
    ap.add_argument("--ws", help="workspace dir for extract")
    ap.add_argument("--gemini-stream", help="fallback gemini_stream.jsonl (extract)")
    ap.add_argument("--name", help="filename substring for reconstruct/blocks")
    ap.add_argument("--format", choices=["claude", "gemini", "codex", "codex2"], default=None)
    a = ap.parse_args()
    if a.cmd == "extract":
        print(json.dumps(extract_workspace(a.path, a.ws or ".", a.gemini_stream),
                         indent=2, ensure_ascii=False))
        return
    f = a.format or detect_log_format(a.path)
    if a.cmd == "format":
        print(f)
    elif a.cmd == "timeline":
        for i, n, s in parse_timeline(a.path, f):
            print(f"{i:4} {n:20} {s}")
    elif a.cmd == "files":
        for p, chan in written_file_paths_detail(a.path, f):
            print(f"{p}\t[{chan}]")
    elif a.cmd == "text":
        print(assistant_text(a.path, f))
    elif a.cmd == "reconstruct":
        print(reconstruct_file(a.path, a.name or "", f))
    elif a.cmd == "blocks":
        if f != "codex":
            for i, n, s in parse_timeline(a.path, f):
                print(f"{i:4} {n:20} {s}")
            return
        for b in _codex_blocks(a.path):
            if a.name and a.name not in b.get("raw", ""):
                continue
            print(f"{b['idx']:4} {b['kind']:11} {b['status']:12} {b['dur']:>8} "
                  f"out={len(b['output']):7} {_short(b['cmd'] or ','.join(b.get('paths', [])), 120)}")
    elif a.cmd == "searches":
        if f in ("codex", "codex2"):
            print("## NOTE: codex has no search tool; showing network calls made from exec.")
        rets = search_returns(a.path, f)
        if not rets:
            print("## (no retrieval calls found)")
        for k, q, r in rets:
            print(f"#### {k} | {q}\nLEN {len(r)}\n{r[:600]}\n")


if __name__ == "__main__":
    _cli()
