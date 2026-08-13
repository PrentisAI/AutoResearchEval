#!/usr/bin/env python3
"""
qa_check_analysis.py — quality gate for a generated analysis.md.

Codifies the ONBOARDING.md bar so the driver can decide pass/redo. Six independent
gate classes, calibrated so no single stage can be a stub and no issue can be padding:

  1. totals           — chars / issues / chars-per-issue
  2. section length    — per-section char floors, so no stage is a stub
  3. section breadth   — per-stage issue counts, so every stage is really covered
  4. issue thickness   — the DISTRIBUTION of per-issue length, not just its mean
  5. density           — verifiable anchors per issue, plus anti-padding limits
                         (verbatim-copy share, near-duplicate share). Raising a
                         length floor without a density floor just invites padding.
  6. label well-formedness — every numbered issue needs a parseable
                         `[stage: <A-F,X> | root cause: <grounding|depth|integrity|
                         robustness>]` trailer, since downstream aggregation depends
                         on it. Values must come from the fixed vocabularies below.

Class 5's source-dependent checks need the task workspace (report.md / decision.json
/ execution log / agent_code). Pass ws=... to enable them; without it they are skipped
and reported as "skipped", so the checker still runs standalone on any .md.

Floors adapt to the scoring reason: delivery/infrastructure failures
(no_decision / no_computation / timeout) are naturally thinner and scale to 0.6.
"""
import json
import re
import sys
from pathlib import Path

# ------------------------------------------------------------------ section model
# (key, label, regexes that identify the heading, min chars at full scale)
# The per-section floors and per-stage issue minimums below are the ones this
# framework's ONBOARDING.md and its paper specify; the INSTRUCTION quota given to
# the writer aims higher (the corpus median), these are the floor, not the target.
SECTIONS = [
    ("core",      "Core Verdict",                   [r"Core Verdict"],                    850),
    ("meta",      "Metadata",                        [r"Metadata"],                         720),
    ("arc",       "Trajectory Arc",                   [r"Trajectory Arc"],                   380),
    ("fair",      "Credit Due",                       [r"Credit Due"],                        760),
    ("s1",        "A. Ideation & Planning",           [r"A\.\s*Ideation", r"\bIdeation\b"],   760),
    ("s2",        "B. Retrieval & Synthesis",         [r"B\.\s*Retrieval", r"\bRetrieval\b"], 1020),
    ("s3",        "C. Execution & Implementation",    [r"C\.\s*Execution", r"\bExecution\b"], 2550),
    ("s4",        "D. Analysis & Interpretation",     [r"D\.\s*Analysis", r"\bAnalysis\b"],   600),
    ("s5",        "E. Writing & Documentation",       [r"E\.\s*Writing", r"\bWriting\b"],     640),
    ("s6",        "F. Self-Verification & Review",    [r"F\.\s*Self-Verification", r"Self-Verification", r"\bReview\b"], 640),
    ("sx",        "X. Cross-Stage Dynamics",          [r"X\.\s*Cross-Stage", r"Cross-Stage"], 520),
    ("checklist", "Sentence-by-Sentence Checklist",   [r"Sentence-by-Sentence Checklist", r"\bChecklist\b"], 1100),
    ("grounding", "Numerical Grounding Notes",         [r"Numerical Grounding", r"\bGrounding\b"], 600),
    ("retract",   "Retraction / Correction Log",      [r"Retraction", r"Correction Log"],     340),
    ("verdict",   "One-Line Verdict",                 [r"One-Line Verdict"],                  210),
]
MIN_CHECKLIST_ROWS = 12
# per-stage minimum issue counts at full scale
SECTION_MIN_ISSUES = {"s1": 3, "s2": 4, "s3": 10, "s4": 3, "s5": 3, "s6": 3, "sx": 2}
MIN_ANCHORS = 300
MAX_THIN_FRAC = 0.28
MAX_TINY_FRAC = 0.05
MIN_ANCHOR_FRAC = 0.85
MIN_LABEL_FRAC = 0.85      # gate 6: fraction of issues needing a well-formed trailer
MAX_DUP_RATIO = 0.08       # catches template padding
MAX_COPIED_RATIO = 0.20
STAGE_KEYS = ["s1", "s2", "s3", "s4", "s5", "s6", "sx"]

THIN_REASONS = ("no_decision", "no_computation", "timeout")
THIN_SCALE = 0.6

# Every model/run seems to invent its own marker label style, so rather than keep
# enumerating exact prefixes, match on STRUCTURE. A numbered marker is one of:
#   (a) an opening bracket, ANY short prefix inside it, digits, then a matching
#       closing bracket — "[E-1]" "(E-1)" "(E1)"
#   (b) a KNOWN label word + space + digits + separator, where ":" IS allowed as
#       the separator — "issue 12 —" "Finding 3:". Colon is safe here only because
#       a recognized label word already disambiguates from a bare timestamp.
#   (c) a short, narrow prefix (single Latin letter or "#") directly against the
#       digits (no space), then a separator/closing bracket, WITHOUT colon —
#       "12." "12)" "12]" "12 " "12**" (bold-close) "12|" "E1(***)..." "#18 *...".
#       Colon is excluded here (unlike (b)) because there is no label word to rule
#       out a bare timeline entry like "**05:43:13…**", and a permissive prefix
#       ending at a bare space would false-hit ordinary prose like "in 2026".
# A run may format every finding as a markdown list item ("- **#18 *...**") instead
# of a bare bold-paragraph start — allow one optional bullet marker (-, *, +) before
# the bold delimiter.
# (d) a circled-numeral glyph ("(13) * ..." rendered as "⑬") — these are Unicode
#     category "No" (Other Number), NOT "Nd" (decimal digit), so plain \d+ never
#     matches them; a run that numbers findings this way produced a document with
#     real issues the checker scored as 0 in every section until this branch was
#     added.
_LABEL_WORD = r"(?:[Ii]ssue|[Ff]inding|[Pp]roblem)\s+"
_CIRCLED_NUM = "[①-⒛⓪-⓿❶-➓㉑-㉟㊱-㊿]"
ISSUE_RE = re.compile(
    r"(?m)^\s*(?:[-*+]\s+)?\*{0,2}(?:"
    r"[\[({][^\d\n]{0,10}\d+[\])}]"
    r"|"
    r"(?:" + _LABEL_WORD + r")\d+[.)\]:—(|\s\*]"
    r"|"
    r"(?:[A-Za-z#])?[-–—]?\d+[.)\]—(|\s\*]"
    r"|"
    r"" + _CIRCLED_NUM + r"\s"
    r")"
)
HEADING_RE = re.compile(r"(?m)^##+\s+(.*)$")
CHECKLIST_ROW_RE = re.compile(r"(?m)^\s*\|")

# gate 6: the per-issue [stage | root cause] trailer ONBOARDING.md §3 requires.
STAGE_VOCAB = "A-F X".replace(" ", "")
ROOT_CAUSE_VOCAB = ("grounding", "depth", "integrity", "robustness")
TRAILER_RE = re.compile(
    r"\[\s*stage\s*:\s*([A-FX])\s*\|\s*root\s*cause\s*:\s*"
    r"(grounding|depth|integrity|robustness)\s*\]", re.I)

# anchors = things a reader could go and verify
ANCHOR_RES = [
    re.compile(r"`[^`\n]{2,}`"),                                   # code / value spans
    re.compile(r"\b\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\b"),             # numeric literals
    re.compile(r"\b[\w./-]+\.(?:py|json|jsonl|md|txt|csv|npy|npz|log|sh|ya?ml)\b"),
    re.compile(r"\b[Ll](?:ine)?\s?\d+\b|:\d+\b"),                  # line references
    re.compile(r"10\.\d{4,}/\S+|arXiv:\d{4}\.\d{4,}"),             # DOI / arXiv
]


def _scale(v, thin):
    return int(v * THIN_SCALE) if thin else v


def count_issues(text):
    """A 'finding' line starts (after optional bold **) with a marker label:
    "12."  "**12.**"  "**E-1 …"  "**G1 …"."""
    return len(ISSUE_RE.findall(text)), len(re.findall(r"★", text))


def split_sections(text):
    """{key: body_text} for every recognized section heading, plus '_order'."""
    heads = [(m.start(), m.end(), m.group(1).strip()) for m in HEADING_RE.finditer(text)]
    out, order = {}, []
    for i, (s, e, title) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        body = text[e:end]
        for key, _label, pats, _floor in SECTIONS:
            if key in out:
                continue
            if any(re.search(p, title, re.I) for p in pats):
                out[key] = body
                order.append(key)
                break
    out["_order"] = order
    return out


def issue_segments(body):
    """Per-issue text spans inside one section body."""
    marks = [m.start() for m in ISSUE_RE.finditer(body)]
    segs = []
    for i, s in enumerate(marks):
        e = marks[i + 1] if i + 1 < len(marks) else len(body)
        segs.append(body[s:e].strip())
    return segs


def _anchor_count(s):
    return sum(len(rx.findall(s)) for rx in ANCHOR_RES)


def _load_sources(ws):
    """Concatenate everything the analysis is allowed to quote from."""
    if not ws:
        return None, None
    ws = Path(ws)
    quotable, artifact = [], []
    for name in ("report.md", "decision.json"):
        p = ws / name
        if p.exists():
            t = p.read_text(errors="ignore")
            quotable.append(t)
            artifact.append(t)
    for name in ("agent_log.jsonl", "agent_log.txt", "gemini_stream.jsonl"):
        p = ws / name
        if p.exists():
            quotable.append(p.read_text(errors="ignore"))
    # Injected side resources (gold rubric, the task's own problem statement, the
    # scoring service's real source, harness execution/scoring records) — a session
    # that correctly quotes a gold-observable name or the task's own wording is doing
    # exactly the grounding we want; not loading these penalizes precisely the most
    # diligent analyses.
    for name in ("gold_rubric.json", "task_instruction.md", "task_index.json",
                "traj_tools.py", "INSTRUCTION.md", "result.json", "submissions.jsonl",
                "problem_readme.md", "data_description.md", "verification.md",
                "task_metadata.json"):
        p = ws / name
        if p.exists():
            quotable.append(p.read_text(errors="ignore"))
    for dirname in ("agent_code", "evaluator", "ground_truth"):
        d = ws / dirname
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.stat().st_size < 2_000_000:
                    try:
                        quotable.append(p.read_text(errors="ignore"))
                    except Exception:
                        pass
    return ("\n".join(quotable) if quotable else None,
            "\n".join(artifact) if artifact else None)


def _verbatim_stats(text, quotable, artifact):
    """(traceable_quotes, copied_char_ratio) — grounding vs. padding-by-pasting."""
    traceable = 0
    for m in re.finditer(r"`([^`\n]{20,})`", text):
        if quotable and m.group(1) in quotable:
            traceable += 1
    copied = 0
    if artifact:
        for line in text.splitlines():
            s = line.strip()
            if len(s) >= 120 and s in artifact:
                copied += len(s)
    return traceable, (copied / max(len(text), 1))


def _dup_ratio(text):
    """Share of 60-char normalized shingles that repeat — catches template padding."""
    norm = re.sub(r"\s+", " ", text)
    if len(norm) < 600:
        return 0.0
    shingles = [norm[i:i + 60] for i in range(0, len(norm) - 60, 30)]
    if not shingles:
        return 0.0
    seen, dup = set(), 0
    for sh in shingles:
        if sh in seen:
            dup += 1
        else:
            seen.add(sh)
    return dup / len(shingles)


def check(md_path, reason="", ws=None):
    p = Path(md_path)
    res = {"path": str(p), "exists": p.exists(), "ok": False, "problems": [], "gates": {}}
    if not p.exists():
        res["problems"].append("missing")
        return res
    text = p.read_text(errors="ignore")
    chars = len(text)
    issues, stars = count_issues(text)
    thin = str(reason).startswith(THIN_REASONS)
    res.update(chars=chars, issues=issues, stars=stars, thin_profile=thin)

    prob = res["problems"]

    # ---------------------------------------------------------------- 1. totals
    min_chars = _scale(16000, thin)
    min_issues = _scale(28, thin)
    ratio = chars / max(issues, 1)
    res["ratio"] = round(ratio, 1)
    if chars < min_chars:
        prob.append(f"too short: {chars} < {min_chars}")
    if issues < min_issues:
        prob.append(f"too few issues: {issues} < {min_issues}")
    if ratio < 300:
        prob.append(f"chars/issue {ratio:.0f} < 300")
    res["gates"]["totals"] = {"chars": chars, "min_chars": min_chars, "issues": issues,
                              "min_issues": min_issues, "ratio": round(ratio, 1)}

    # ------------------------------------------------- 2/3. per-section presence
    sec = split_sections(text)
    sec_report = {}
    for key, label, _pats, floor in SECTIONS:
        body = sec.get(key)
        if body is None:
            prob.append(f"missing section:{label}")
            sec_report[key] = {"chars": 0, "issues": 0, "present": False}
            continue
        n = len(body)
        segs = issue_segments(body)
        floor_s = _scale(floor, thin)
        entry = {"chars": n, "min_chars": floor_s, "issues": len(segs), "present": True}
        if n < floor_s:
            prob.append(f"section too short:{label} {n} < {floor_s}")
        need = SECTION_MIN_ISSUES.get(key)
        if need is not None:
            need_s = max(_scale(need, thin), 2)
            entry["min_issues"] = need_s
            if len(segs) < need_s:
                prob.append(f"section too few issues:{label} {len(segs)} < {need_s}")
        sec_report[key] = entry
    # checklist is a table, count rows not numbered issues
    if "checklist" in sec:
        rows = len(CHECKLIST_ROW_RE.findall(sec["checklist"]))
        need_rows = _scale(MIN_CHECKLIST_ROWS, thin)
        sec_report["checklist"]["rows"] = rows
        if rows < need_rows:
            prob.append(f"checklist too few rows: {rows} < {need_rows}")
    res["gates"]["sections"] = sec_report

    # ------------------------------------------------------- 4. issue thickness
    all_segs = []
    for key in STAGE_KEYS:
        if key in sec:
            all_segs += issue_segments(sec[key])
    if not all_segs:
        all_segs = issue_segments(text)
    n_seg = len(all_segs)
    tiny = sum(1 for s in all_segs if len(s) < 100)
    thinish = sum(1 for s in all_segs if len(s) < 200)
    thin_frac = thinish / max(n_seg, 1)
    tiny_frac = tiny / max(n_seg, 1)
    res["gates"]["thickness"] = {"n": n_seg, "tiny(<100)": tiny, "tiny_frac": round(tiny_frac, 3),
                                 "thin(<200)": thinish, "thin_frac": round(thin_frac, 3)}
    if tiny_frac > MAX_TINY_FRAC:
        prob.append(f"{tiny}/{n_seg} issues under 100 chars ({tiny_frac:.0%} > {MAX_TINY_FRAC:.0%})")
    if thin_frac > MAX_THIN_FRAC:
        prob.append(f"thin-issue share {thin_frac:.0%} > {MAX_THIN_FRAC:.0%}")

    # ------------------------------------------------------------- 5. density
    anchors = _anchor_count(text)
    with_anchor = sum(1 for s in all_segs if _anchor_count(s) >= 1)
    anchor_frac = with_anchor / max(n_seg, 1)
    min_anchors = _scale(MIN_ANCHORS, thin)
    dens = {"anchors": anchors, "min_anchors": min_anchors,
            "issues_with_anchor_frac": round(anchor_frac, 3)}
    if anchors < min_anchors:
        prob.append(f"too few verifiable anchors: {anchors} < {min_anchors}")
    if anchor_frac < MIN_ANCHOR_FRAC:
        prob.append(f"only {anchor_frac:.0%} of issues carry a verifiable anchor "
                    f"(<{MIN_ANCHOR_FRAC:.0%})")

    dup = _dup_ratio(text)
    dens["dup_ratio"] = round(dup, 3)
    if dup > MAX_DUP_RATIO:
        prob.append(f"near-duplicate share {dup:.0%} > {MAX_DUP_RATIO:.0%} (template padding)")

    quotable, artifact = _load_sources(ws)
    if quotable is None:
        dens["traceable_quotes"] = "skipped (no ws)"
        dens["copied_ratio"] = "skipped (no ws)"
    else:
        traceable, copied = _verbatim_stats(text, quotable, artifact)
        dens["traceable_quotes"] = traceable
        dens["copied_ratio"] = round(copied, 3)
        need_q = _scale(8, thin)
        if traceable < need_q:
            prob.append(f"only {traceable} verbatim-traceable quotes < {need_q}")
        if copied > MAX_COPIED_RATIO:
            prob.append(f"verbatim-copied share {copied:.0%} > {MAX_COPIED_RATIO:.0%}")
    res["gates"]["density"] = dens

    # ------------------------------------------------- 6. label well-formedness
    labeled = sum(1 for s in all_segs if TRAILER_RE.search(s))
    label_frac = labeled / max(n_seg, 1)
    res["gates"]["labels"] = {"n": n_seg, "labeled": labeled, "label_frac": round(label_frac, 3)}
    if label_frac < MIN_LABEL_FRAC:
        prob.append(f"only {label_frac:.0%} of issues carry a well-formed "
                    f"[stage | root cause] trailer (<{MIN_LABEL_FRAC:.0%})")

    res["ok"] = not prob
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("md_path")
    ap.add_argument("--reason", default="")
    ap.add_argument("--ws", default=None, help="task workspace, enables source-grounded gates")
    a = ap.parse_args()
    r = check(a.md_path, a.reason, a.ws)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    sys.exit(0 if r["ok"] else 1)
