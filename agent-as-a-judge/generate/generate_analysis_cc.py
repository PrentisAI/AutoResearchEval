#!/usr/bin/env python3
"""
generate_analysis_cc.py — spawn ONE fresh headless Claude Code session per trajectory
to produce a comprehensive, ONBOARDING-conformant analysis.md.

Stage 1 of the two-stage pipeline: raw trajectory log -> structured analysis.md
(Stage 2, in ../classify/, turns analysis.md into taxonomy classifications).

Each task:
  1. traj_tools.extract_workspace() -> per-task workspace with decision.json/report.md/
     the agent's raw log/meta.json/traj_tools.py + INSTRUCTION.md
  2. claude -p (fresh session) reads INSTRUCTION.md + ONBOARDING.md + the exemplar,
     deep-dives the log, reruns light code in the workspace, writes analysis.md to
     <corpus>/<model>/<task_id>/analysis.md
  3. QA gate (qa_check_analysis.py); on fail -> status qa_fail, retried next --resume
     pass with the QA failure reasons fed back into the prompt.

BEFORE RUNNING FOR REAL: edit RETRIEVAL_NOTE and GOLD_NOTE below to describe your own
harness/benchmark's retrieval-tool reality and gold-value availability. The originals
here are TODO placeholders — asserting the wrong thing (e.g. claiming a real WebSearch
tool is a shim when it isn't) would inject a false premise into every analysis.

Usage:
  python3 generate_analysis_cc.py --run-dir /path/to/your_model__your_suite \
      --concurrency 4 --resume --model claude-opus-4-8
  # smoke test on 2 tasks:
  python3 generate_analysis_cc.py --run-dir <...> --tasks task_id_1,task_id_2 --concurrency 2
  # see what would run without spawning anything:
  python3 generate_analysis_cc.py --run-dir <...> --dry-run

Expected --run-dir layout: <run-dir>/traj/*.json, one JSON object per trajectory with
at least a `task_id` field and a log field traj_tools.detect_format() can recognize
(see traj_tools.py for the supported formats and their expected keys).

Output corpus root defaults to ./corpus (relative to cwd), overridable via the
AAJ_CORPUS_DIR env var — this is also where ../classify/'s scripts expect to find
their input by default, so the two stages compose without extra flags.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                              # repo root (agent-as-a-judge/)
ONBOARDING = ROOT / "ONBOARDING.md"
EXEMPLAR_LONG = ROOT / "analysis_long.md"
CORPUS_DIR = Path(os.environ.get("AAJ_CORPUS_DIR", "corpus")).resolve()
sys.path.insert(0, str(HERE))
import traj_tools
import qa_check_analysis

DEFAULT_WORKROOT = str(HERE / "_ws")

# ---------------------------------------------------------------------------------
# ADAPT THESE TWO NOTES to your own harness before running for real. They get quoted
# verbatim into every analysis session's instructions.
RETRIEVAL_NOTE = (
    "TODO: describe whether WebSearch/WebFetch (or your harness's equivalent tools) "
    "do real network I/O or are mocked/shimmed in the trajectories you're analyzing. "
    "The analyst needs this to correctly judge retrieval honesty, citation "
    "provenance, and possible answer contamination — guessing wrong in either "
    "direction produces false findings."
)
GOLD_NOTE = (
    "TODO: describe whether ground-truth/gold values are available locally for the "
    "analyst to compare against, or whether it must rely on independent "
    "recomputation, unit/magnitude sanity checks, and internal consistency instead."
)
# ---------------------------------------------------------------------------------

# Optional: if your trajectories carry a `reason` string (e.g. why an automated score
# was low), map recognized prefixes to a note guiding how the analyst should read it.
# Anything not matching falls through to "soft" (the default assumption: whatever
# score exists reflects genuine quality). This whole mechanism is optional — if your
# trajectories have no such concept, every task will just take the "soft" branch,
# which is a harmless no-op note.
KNOWN_CATEGORIES = ("judge_unavailable", "no_decision", "no_computation", "timeout")
REASON_NOTES = {
    "soft": ("Treat the trajectory's score (if any) as a genuine quality signal. "
             "Focus on: does the reported result actually answer the stated "
             "objective, or is the metric mismatched/tautological/an artifact of a "
             "single convenient assumption?"),
    "judge_unavailable": ("The scoring infrastructure was unavailable — this is an "
                          "infrastructure fact, NOT a quality signal. Judge the "
                          "trajectory's science and delivery normally, but do not "
                          "attribute a low/missing score to quality; say explicitly "
                          "that scoring infra failed."),
    "no_decision": ("This trajectory failed to deliver a valid decision/output "
                    "artifact (missing fields, null result, malformed JSON, etc.) — "
                    "a delivery failure, not a science-quality failure. Focus on "
                    "which step broke and why nothing valid was produced."),
}


def find_claude_bin(explicit=None):
    if explicit:
        return explicit
    w = shutil.which("claude")
    if w:
        return w
    for p in (Path.home() / ".local/bin/claude", Path.home() / ".npm-global/bin/claude",
              Path("/usr/local/bin/claude")):
        if p.exists():
            return str(p)
    raise SystemExit("claude CLI not found; pass --claude-bin")


def model_key_from_run(run_dir):
    """e.g. 'my-model__my-suite' -> 'my-model'; falls back to the bare dir name."""
    return Path(run_dir).name.split("__")[0]


def discover(run_dir):
    """Enumerate <run_dir>/traj/*.json. Each file is expected to be one trajectory
    record with at least `task_id`; traj_tools.detect_format() reads whichever log
    field is present to figure out which harness produced it."""
    traj = Path(run_dir) / "traj"
    out = []
    for fp in sorted(traj.glob("*.json")):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        reason = d.get("reason", "") or ""
        cat = next((c for c in KNOWN_CATEGORIES if reason.startswith(c)), "soft")
        fmt = traj_tools.detect_format(d)
        out.append({"task_id": d.get("task_id"), "json": str(fp),
                    "reward": d.get("reward"), "reason": reason, "category": cat,
                    "log_format": fmt,
                    "log_chars": len(d.get(traj_tools.LOG_KEY[fmt], "") or "")})
    return out


def task_dir(model_root, task_id):
    """Bare <model_root>/<task_id>/ — deliberately not a free-form-description
    directory. Stage 2's discover() takes a task's directory name as its task_id
    verbatim (no parsing/splitting), so the two stages only compose without extra
    flags if Stage 1 always writes to the exact task_id, nothing appended."""
    return model_root / task_id


def exemplar_short():
    """An optional second, in-progress exemplar pulled from whatever the corpus has
    already accumulated (any existing analysis.md), so later sessions see a
    same-project reference alongside the fixed EXEMPLAR_LONG. Returns None on a
    fresh corpus — the prompt degrades gracefully when there's nothing yet."""
    if not CORPUS_DIR.exists():
        return None
    for md in sorted(CORPUS_DIR.glob("*/*/analysis.md")):
        return md
    return None


def build_instruction(t, model_key, target_dir, prior_problems=None):
    reason, cat = t["reason"], t["category"]
    strengthen = ""
    if prior_problems:
        strengthen = (
            "\n## ⚠️ Prior attempt did not pass QA — this is a redo, must go deeper\n"
            f"Prior QA verdict: **{'; '.join(prior_problems)}**. Fix specifically:\n"
            "- If 'too few issues': don't pad — go deeper. Run every relevant "
            "`traj_tools.py` subcommand (timeline/files/searches/reconstruct), rebuild "
            "and read the actual final code line by line, turn every checkable "
            "sentence in report.md/decision.json into a row of a claim-by-claim "
            "verdict table (✅/⚠️/❌), and split every bug/metric-mismatch/tautology/"
            "strawman comparison in the execution section into its own paragraph. "
            "Target >=20 substantial issues (25-40 for computational trajectories) "
            "spanning all six stages plus the cross-stage layer (X).\n"
            "- If 'too short': write each issue as a full paragraph (mechanism + why "
            "it matters + fair-credit caveat + line-number/numeric evidence), not a "
            "one-line bullet.\n"
            "- Overwrite the SAME analysis.md — do not create a second directory.\n"
        )
    ex2 = exemplar_short()
    ex2_line = f"- Second exemplar (same framework, from this corpus): `{ex2}`" if ex2 else ""
    reason_note = REASON_NOTES.get(cat, REASON_NOTES["soft"])

    target_line = f"**Write exactly to**: `{target_dir}/analysis.md` (overwrite if it exists)."

    return f"""# Task: write a comprehensive analysis.md for ONE agentic-research trajectory

You are doing a deep-dive analysis of an agentic scientific-discovery trajectory. This
session analyzes exactly **one** trajectory, standalone — do not assume conclusions
from any other trajectory. You are **fully autonomous**; do not ask questions.
{strengthen}
## 0. Required reading (use the Read tool, read all of it before starting)
1. Framework: `{ONBOARDING}` — follow its workflow, depth standard, six-stage-plus-X
   structure, and its "iron rules" exactly.
2. Depth/structure **gold-standard exemplar**: `{EXEMPLAR_LONG}`. Your output's depth
   must match it.
{ex2_line}

## 1. This task's data (all in the current working directory)
- `meta.json` — task_id / reward / reason / any dims or observable description.
  **task_id={t['task_id']}, reward={t['reward']}, reason={reason}.**
- `decision.json` — the agent's final delivered decision (problem/hypothesis/system/
  method/observable/result/conclusion/process_log, if present).
- `report.md` — the agent's delivered report (may include a peer-review section).
- the agent's raw execution log — **your main excavation site.**
- `traj_tools.py` — helper for parsing the log, **use it**, don't reimplement parsing
  from scratch:
  - `python3 traj_tools.py timeline <log>` -> tool-call timeline (order of
    grep -> read -> run -> write)
  - `python3 traj_tools.py files <log>` -> which code files the agent wrote
  - `python3 traj_tools.py reconstruct <log> --name <filename fragment>` ->
    rebuild the final delivered version of a file (Write base + Edit replay; judge
    by the final deliverable, not intermediate drafts)
  - `python3 traj_tools.py searches <log>` -> raw content of every
    WebSearch/WebFetch return
  - or `import traj_tools` and use `pair_tool_results` / `assistant_text` /
    `search_returns` directly.

## 2. How to read the scoring reason
{reason_note}

## 3. Retrieval and ground-truth facts for this corpus (edit these per your harness)
- {RETRIEVAL_NOTE}
- {GOLD_NOTE}

## 4. Output requirements
- Use ONBOARDING's fixed skeleton (**all sections required, headings copied
  verbatim**): a title, `> Core Verdict`, `## Metadata`, `## Trajectory Arc`,
  `## Credit Due`, then `## A. Ideation & Planning`, `## B. Retrieval & Synthesis`,
  `## C. Execution & Implementation` (heaviest), `## D. Analysis & Interpretation`,
  `## E. Writing & Documentation`, `## F. Self-Verification & Review`, then
  `## X. Cross-Stage Dynamics` (error propagation, goal drift, right-for-the-wrong-
  reason outcomes — dynamics that don't belong to a single stage), a
  `## Sentence-by-Sentence Checklist`, `## Numerical Grounding Notes`, a
  `## Retraction / Correction Log`, and a `## One-Line Verdict`.
- Every issue is a **full paragraph**: mechanism (what happened) + why it's harmful +
  fair-credit reading + evidence (line numbers/numbers), and carries a one-line
  trailer `[stage: <A-F,X> | root cause: <grounding|depth|integrity|robustness>]` —
  the two coordinates a downstream classifier needs. Get both right: the stage letter
  must be one of A-F or X, and root cause must be exactly one of the four words above.
- Breadth: computational/buggy trajectories should surface **25-40 issues** across all
  six stages plus X (X: >=2); genuinely limited trajectories (e.g. a single
  catastrophic delivery failure) need **16+ issues, each still deep** rather than
  padding.
- **Calibration floor**: total characters / number of issues **>= 280**; total length
  should generally be >=12000 characters (thinner deliveries may go as low as ~9000).
- Give fair credit too (honest reporting of a negative result, a real mechanistic
  model, genuine self-correction, honestly-flagged failed retrieval) — weigh
  criticism and credit together; don't be fooled by a well-written self-diagnosis
  that was never acted on.

## 5. Where to write it
{target_line}
(Use Bash `mkdir -p` to create the directory, then Write the file. When done, run
`wc -m` on it and count your issues to confirm chars/issues >= 280.)

Start now: run timeline/files/searches to understand the trajectory, reconstruct and
read the key code line by line, do numerical sanity checks / selective re-execution,
then write the full analysis.md following the skeleton above.
"""


INFRA_ERROR_STATUSES = {401, 403, 429, 500, 502, 503, 529}


def _infra_error_status(session_log):
    """If the session's terminal event is a hard API-level failure (bad auth, rate
    limit, upstream 5xx) rather than a content/quality problem, return the HTTP
    status; else None.

    This distinction matters because a transient provider-side blip (seen in
    practice: every concurrent request rejected with 401 'User not found' for
    several minutes, then recovering with no code change on our end) must NOT
    consume the same `attempts` budget as a genuine generation failure — two
    unlucky blips would otherwise permanently give up on a task that never got a
    real shot at producing an analysis.
    """
    p = Path(session_log)
    if not p.exists():
        return None
    last_result = None
    for line in open(p, errors="ignore"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("type") == "result":
            last_result = o
    if last_result and last_result.get("is_error") and last_result.get("api_error_status") in INFRA_ERROR_STATUSES:
        return last_result["api_error_status"]
    return None


def _prior_attempts(model_key, task_id):
    """Accumulated attempt count for a task from the manifest (0 if none)."""
    man = CORPUS_DIR / "_batch" / f"{model_key}_manifest.json"
    if not man.exists():
        return 0
    try:
        rec = json.load(open(man)).get(task_id, {})
        return int(rec.get("attempts", 0))
    except Exception:
        return 0


def run_one(t, args, claude_bin, model_key):
    task_id = t["task_id"]
    model_root = CORPUS_DIR / model_key
    model_root.mkdir(parents=True, exist_ok=True)
    forced = task_id in args._force_set
    tgt = task_dir(model_root, task_id)

    # resume-skip: existing analysis.md that passes QA (unless forced)
    prior_qa = None
    ws = Path(args.workroot) / model_key / task_id
    if (tgt / "analysis.md").exists():
        prior_qa = qa_check_analysis.check(tgt / "analysis.md", t["reason"],
                                           ws if ws.exists() else None)
        if args.resume and not forced and prior_qa["ok"]:
            return {"task_id": task_id, "status": "skipped", "qa": prior_qa, "dir": str(tgt)}

    # per-task attempt cap so stubborn tasks don't burn credits forever
    prior_attempts = _prior_attempts(model_key, task_id)
    if not forced and prior_attempts >= args.max_attempts:
        return {"task_id": task_id, "status": "gave_up", "attempts": prior_attempts,
                "qa": prior_qa or {}, "dir": str(tgt)}

    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    prior_problems = (prior_qa or {}).get("problems") if prior_qa and not prior_qa.get("ok") else None

    traj_tools.extract_workspace(t["json"], ws)
    shutil.copy(HERE / "traj_tools.py", ws / "traj_tools.py")
    (ws / "INSTRUCTION.md").write_text(build_instruction(t, model_key, str(tgt), prior_problems))

    prompt = ("Read INSTRUCTION.md in this directory FIRST and follow it completely. "
              "You are fully autonomous; do not ask questions. Produce the analysis.md at the "
              "path INSTRUCTION.md specifies before stopping.")
    cmd = [claude_bin, "--print", "--verbose", "--output-format", "stream-json",
           "--permission-mode", "bypassPermissions", "--max-turns", str(args.max_turns),
           "--model", args.model, "--effort", args.effort,
           "--add-dir", str(ROOT), "-p", prompt]
    env = os.environ.copy()
    env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    logf = ws / "session.log"
    started = time.time()
    try:
        with open(logf, "w") as lf:
            proc = subprocess.run(cmd, cwd=ws, env=env, stdin=subprocess.DEVNULL,
                                  stdout=lf, stderr=subprocess.STDOUT, timeout=args.task_timeout, text=True)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = -1
    dur = round(time.time() - started, 1)

    md = tgt / "analysis.md"
    if not md.exists():
        infra = _infra_error_status(logf)
        if infra is not None:
            # Don't burn the retry budget on a provider-side outage; keep
            # prior_attempts unchanged so --resume retries it for free next pass.
            return {"task_id": task_id, "status": "infra_error", "returncode": rc, "dur": dur,
                    "reason": t["reason"], "log_format": t.get("log_format"),
                    "api_error_status": infra, "session_log": str(logf),
                    "attempts": prior_attempts}
        return {"task_id": task_id, "status": "failed", "returncode": rc, "dur": dur,
                "reason": t["reason"], "log_format": t.get("log_format"),
                "session_log": str(logf), "attempts": prior_attempts + 1}
    qa = qa_check_analysis.check(md, t["reason"], ws)
    return {"task_id": task_id, "status": "done" if qa["ok"] else "qa_fail",
            "returncode": rc, "dur": dur, "reason": t["reason"],
            "log_format": t.get("log_format"), "dir": str(tgt),
            "qa": qa, "session_log": str(logf), "attempts": prior_attempts + 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="directory containing traj/*.json (one file per trajectory)")
    ap.add_argument("--categories", default="soft,judge_unavailable,no_decision",
                    help="comma list of reason-category buckets to include (see "
                         "KNOWN_CATEGORIES/REASON_NOTES); ignored if --tasks is set")
    ap.add_argument("--dry-run", action="store_true",
                    help="discover and print the task table, spawn nothing")
    ap.add_argument("--tasks", default="", help="comma id subset (overrides --categories)")
    ap.add_argument("--force-tasks", default="", help="comma ids to regenerate even if present")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--model", default=os.environ.get("CLAUDE_CODE_MODEL", "claude-opus-4-8"))
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-turns", type=int, default=80)
    ap.add_argument("--task-timeout", type=int, default=3600)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--claude-bin", default=None)
    ap.add_argument("--workroot", default=DEFAULT_WORKROOT)
    ap.add_argument("--n", type=int, default=0, help="cap number of tasks (0=all)")
    ap.add_argument("--max-attempts", type=int, default=4,
                    help="stop retrying a task after this many attempts (cost guard)")
    args = ap.parse_args()

    claude_bin = find_claude_bin(args.claude_bin) if not args.dry_run else "(dry-run)"
    model_key = model_key_from_run(args.run_dir)
    args._force_set = set(x for x in args.force_tasks.split(",") if x)
    cats = set(args.categories.split(","))
    tasks = discover(args.run_dir)
    if args.tasks:
        want = set(args.tasks.split(","))
        tasks = [t for t in tasks if t["task_id"] in want]
    else:
        tasks = [t for t in tasks if t["category"] in cats]
    if args.n:
        tasks = tasks[:args.n]

    print(f"[gen] run={model_key} corpus={CORPUS_DIR} claude={claude_bin} "
          f"model={args.model} concurrency={args.concurrency} tasks={len(tasks)}", flush=True)
    if args.dry_run:
        import collections
        cc, cf = collections.Counter(), collections.Counter()
        for t in discover(args.run_dir):
            cc[t["category"]] += 1
            cf[t["log_format"]] += 1
        print(f"[dry-run] discovered={sum(cc.values())} categories={dict(cc)} "
              f"formats={dict(cf)} selected={len(tasks)}")
        for t in tasks[:5]:
            print(f"  {t['task_id']} {t['category']:18} {t['log_format']:7} "
                  f"log={t['log_chars']//1000}k reason={t['reason']}")
        return
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futs = {ex.submit(run_one, t, args, claude_bin, model_key): t["task_id"] for t in tasks}
        for fut in as_completed(futs):
            tid = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                # A single worker crash (e.g. the claude binary briefly vanishing
                # mid-self-update) must not discard every other already-completed
                # future in this batch: catch, log, and keep draining as_completed.
                print(f"[worker-error] {tid:12} {type(e).__name__}: {e}", flush=True)
                results.append({"task_id": tid, "status": "failed",
                                 "reason": "worker_exception", "error": str(e)})
                continue
            results.append(r)
            q = r.get("qa", {})
            print(f"[done] {r['task_id']:12} status={r['status']:8} dur={r.get('dur','-')}"
                  f" chars={q.get('chars','-')} issues={q.get('issues','-')}"
                  f" {('PROB='+';'.join(q.get('problems',[])) ) if q.get('problems') else ''}", flush=True)

    man_dir = CORPUS_DIR / "_batch"
    man_dir.mkdir(parents=True, exist_ok=True)
    manifest = man_dir / f"{model_key}_manifest.json"
    prev = json.load(open(manifest)) if manifest.exists() else {}
    for r in results:
        prev[r["task_id"]] = r
    json.dump(prev, open(manifest, "w"), indent=2, ensure_ascii=False)
    n_done = sum(1 for r in results if r["status"] == "done")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    n_qa = sum(1 for r in results if r["status"] == "qa_fail")
    n_fail = sum(1 for r in results if r["status"] == "failed")
    n_infra = sum(1 for r in results if r["status"] == "infra_error")
    print(f"[gen] DONE {model_key} | done={n_done} skipped={n_skip} qa_fail={n_qa} "
          f"failed={n_fail} infra_error={n_infra} | manifest={manifest}", flush=True)
    if n_infra:
        print(f"[gen] WARNING: {n_infra} task(s) hit a provider-side error "
              f"(not counted against --max-attempts; will retry on --resume)", flush=True)


if __name__ == "__main__":
    main()
