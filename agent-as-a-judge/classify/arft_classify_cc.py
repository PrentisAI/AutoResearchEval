#!/usr/bin/env python3
"""
arft_classify_cc.py — label each `analysis.md` in your corpus against ARFT, the
AutoResearch Failure Taxonomy (A.1 … X.8), one fresh headless Claude Code session per
analysis, routed through OpenRouter by `run_all_arft_cc.sh`.

Prefer `arft_classify_api.py` (direct OpenRouter completion, no agentic session) unless
the classifier genuinely needs to go read files beyond the analysis itself — it was
~3.7x cheaper for identical output when both were measured on the same corpus, because
a Claude Code session pays for a system prompt, tool definitions, and multi-turn
re-caching that a plain extraction task doesn't need.

This is a SEPARATE taxonomy from any 29-leaf/undotted classifier you may have lying
around (`legacy/` in this repo, if you keep the historical docs) — the two code sets
COLLIDE IN MEANING (there `C1` might mean "impl bugs"; here `C.1` means "circular
validation"). Never merge their outputs into one table.

Design notes carried over from how this was built and tuned:
  * discover() expects $AAJ_CORPUS_DIR/<model>/<task>/analysis.md (exactly what
    ../generate/generate_analysis_cc.py produces by default) — task ids can be
    anything, not just a fixed prefix pattern.
  * No traj/reward/decision inputs are required. `analysis.md` is expected to be
    self-contained (see ../generate/'s ONBOARDING-conformant output shape, which
    includes a metadata table carrying harness/gold-observable/reward).
  * Sparse output (`hits` / `partials`) rather than 45 inline scores; arft_aggregate.py
    densifies to a 0/1/2 grid (2=HIT, 1=PARTIAL, 0=miss — see arft_patterns.py for why
    that ordering, not 1/2, is deliberate).
  * --max-turns 14 / --effort medium: this is an extraction task over pre-staged
    inputs, not an investigation, so a much smaller turn budget than an authoring task
    needs is enough. Measured: also staging a large per-pattern case-study file (if you
    have one) can push cache_read to ~500k tokens/session because the workspace is
    re-sent every turn — keep the staged workspace small.

Usage (see run_all_arft_cc.sh for the OpenRouter env):
    python3 arft_classify_cc.py --model-key <model> --resume --concurrency 16 \
        --model anthropic/claude-sonnet-5
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                  # repo root (agent-as-a-judge/)
sys.path.insert(0, str(HERE))
import arft_patterns as P                          # noqa: E402
import arft_qa_check as qa                         # noqa: E402

# Bring-your-own-corpus: override with env vars, or just run from the directory where
# you want ./corpus (input) and ./results (output) to live.
CORPUS = Path(os.environ.get("AAJ_CORPUS_DIR", "corpus")).resolve()
OUT_ROOT = Path(os.environ.get("AAJ_OUT_DIR", "results")).resolve()

ARFT_GUIDE = ROOT / "arft_guide.md"

# Auto-discovered from whatever model-named subdirectories exist under CORPUS. Empty
# until you've run Stage 1 (or otherwise populated the corpus) — --model-key will list
# no valid choices until then, which is the correct signal that there's nothing to do.
MODELS = sorted(p.name for p in CORPUS.iterdir() if p.is_dir()) if CORPUS.exists() else []

INFRA_ERROR_STATUSES = {401, 403, 429, 500, 502, 503, 529}


def source_sets():
    """task_id -> source_set, from an OPTIONAL <corpus>/_MANIFEST.tsv (task_id,
    source_set, ...). Purely a provenance label folded into the aggregate stats —
    harmless to omit; every task_id just gets no source_set tag."""
    out = {}
    man = CORPUS / "_MANIFEST.tsv"
    if not man.exists():
        return out
    for line in man.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


SOURCE_SET = source_sets()


def find_claude_bin(explicit=None):
    """Resolve the claude CLI.

    Known absolute paths are checked BEFORE shutil.which: in the pilot, a detached
    background runner resolved `claude` for the first two model dirs and then failed
    with "claude CLI not found" for the rest, because PATH is not reliably inherited
    across the loop. The binary's location is stable, so probe it directly first.
    """
    if explicit:
        return explicit
    for p in (Path.home() / ".local/bin/claude", Path.home() / ".npm-global/bin/claude",
              Path("/usr/local/bin/claude"), Path("/usr/bin/claude")):
        if p.exists():
            return str(p)
    w = shutil.which("claude")
    if w:
        return w
    raise SystemExit("claude CLI not found; pass --claude-bin")


def discover(model_key):
    """Every analysis.md under $AAJ_CORPUS_DIR/<model_key>/<task_id>/.

    Task id is the directory name verbatim — no `(desc)` suffix stripping — so this
    expects the bare-task-id layout Stage 1 (generate_analysis_cc.py) produces.
    """
    out = []
    for md in sorted(glob.glob(str(CORPUS / model_key / "*" / "analysis.md"))):
        d = Path(md).parent
        out.append({"task_id": d.name,
                    "analysis_md": md,
                    "source_set": SOURCE_SET.get(d.name, "unknown")})
    return out


def _infra_error_status(session_log):
    """HTTP status if the session died on a provider-side failure, else None.

    Ported from generate_analysis_cc.py:237. A transient OpenRouter blip (observed:
    every concurrent request 401'ing with "User not found" for minutes, then recovering
    untouched) must not consume the same `attempts` budget as a real content failure.
    """
    p = Path(session_log)
    if not p.exists():
        return None
    last = None
    for line in open(p, errors="ignore"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("type") == "result":
            last = o
    if last and last.get("is_error") and last.get("api_error_status") in INFRA_ERROR_STATUSES:
        return last["api_error_status"]
    return None


def _prior_attempts(model_key, task_id):
    man = OUT_ROOT / f"{model_key}_manifest.json"
    if not man.exists():
        return 0
    try:
        return int(json.load(open(man)).get(task_id, {}).get("attempts", 0))
    except Exception:
        return 0


def build_instruction(t, model_key, prior_problems=None):
    strengthen = ""
    if prior_problems:
        strengthen = (
            "\n## ⚠️ Prior attempt did not pass QA — this is a redo\n"
            f"Prior QA verdict: **{'; '.join(prior_problems)}**. Fix specifically: output "
            "must be **valid JSON**, every hit/partial needs non-empty `evidence` **that "
            "locates a specific section/issue number** in analysis.md, `code` must use "
            "only the 45 valid ARFT codes, and evidence must not come solely from "
            "`## Credit Due` (that's the fair-credit section). Overwrite the same "
            "classification.json.\n")

    return f"""# Task: map one trajectory analysis (analysis.md) onto ARFT and score it
{strengthen}
You are labelling a **deep-dive analysis that has already been written**. It already
contains the evidence, cross-checks, and retractions; **your job is to classify and
count, not to re-judge the trajectory**. Be fully autonomous — do not ask questions,
do not re-investigate.

## 0. Required reading (use the Read tool, **read only these two files**)
1. `arft_guide.md` — **the operational guide, read it in full**. Scoring definitions,
   this corpus's three polarity rules, discrimination rules for easily-confused
   patterns, a Do-NOT-label list, and the full 45-code table are all in there.
2. `analysis.md` — the input for this task.

`arft_codes.txt` is a compact code cheat sheet (the guide already contains the same
content, you usually don't need to read it separately).
**Efficiency requirement: do not browse the directory, do not grep-explore, do not
read any file beyond the two above.** After those two reads, write
`classification.json` directly — the analysis has already done the evidentiary work,
you don't need to investigate further.

## 1. This task
task_id=**{t['task_id']}**, model=**{model_key}**, source_set=**{t['source_set']}**.

## 2. Scoring (per the guide's §0)
- **HIT** → goes in `hits`: the analysis states this failure as **established**, with
  evidence.
- **PARTIAL** → goes in `partials`: the analysis raises it but **with reservations** —
  minor, a single instance, "at risk of" rather than established.
- **Miss** → **the code appears in neither list**: not present, or the analysis
  explicitly clears the agent of it.

You only output these two lists — never write a raw 0/1/2 numeric code. Multi-label is
normal; one mechanism often triggers several codes (a hard-coded constant propping up
the headline is `C.1` + `D.4`; if the agent's own self-review named it and shipped
anyway, add `F.4` too).

## 3. The three easiest ways to get this wrong (condensed from the guide's §1 — follow
   these)
- **Polarity**: `fabricat`/`hallucinat`/`contaminat`-type vocabulary in this corpus is
  **overwhelmingly used to clear the agent** ("no fabricated results were found",
  "external access was clean, no contamination — credit due"). **Mention ≠ hit.**
  Always read the polarity.
- **`## Credit Due` is the fair-credit section**: do not draw evidence from it alone.
  A label sourced only from that section will be rejected by QA.
- **`## Retraction / Correction Log`**: a judgment the analysis itself retracted →
  record as a miss, don't label it.

## 4. Output (write to cwd)
Write `classification.json` (strict JSON, no extra text):
```json
{{"task_id":"{t['task_id']}","model":"{model_key}","source_set":"{t['source_set']}",
 "overall_severity":"high|medium|low|none",
 "summary":"2-4 sentences: what the agent did, main failure/success",
 "hits":[{{"code":"C.1","confidence":0.9,"evidence":"Section C, issue 8 + One-Line Verdict: the 0.984 headline comes from a self-written DGP","why":"matches the C.1 definition because…"}}],
 "partials":[{{"code":"D.3","confidence":0.5,"evidence":"Section D, issue 26: no SEM/significance awareness anywhere","why":"a real gap but the analysis judges it doesn't change the conclusion"}}],
 "iron_rules_cited":[9,6],
 "stage_notes":{{"A":"…","B":"…","C":"…","D":"…","E":"…","F":"…","X":"…"}},
 "uncovered":[{{"mechanism":"…","description":"…","nearest":["C.1","D.1"],"why_each_fails":"C.1 because…; D.1 because…","evidence":"Section …","suggested_new_code":"name: definition"}}]
}}
```
- Every `evidence` is 1-3 sentences and **must be locatable** (section name/issue
  number/short quote). Labels that can't be located get dropped.
- `iron_rules_cited`: every iron-rule number cited anywhere in analysis.md (integer
  array, `[]` if none).
- `stage_notes`: one sentence each for A-F plus one for X; write "ok" if that stage has
  no issues.
- `uncovered`: **judge strictly**. Only use this when a real failure mechanism fits none
  of the 45 codes, and you must individually refute the 2-3 nearest codes. Most
  trajectories should have an empty array here.

When done, self-check with
`python3 -c "import json;json.load(open('classification.json'))"`.
"""


def _model_root(model_key):
    r = OUT_ROOT / model_key
    r.mkdir(parents=True, exist_ok=True)
    return r


def run_one(t, args, claude_bin, model_key):
    task_id = t["task_id"]
    out_root = _model_root(model_key)
    out_json = out_root / f"{task_id}.json"
    forced = task_id in args._force_set

    prior = None
    if out_json.exists():
        prior = qa.check(out_json)
        if args.resume and not forced and prior["ok"]:
            return {"task_id": task_id, "status": "skipped", "qa": prior}

    prior_attempts = _prior_attempts(model_key, task_id)
    if not forced and prior_attempts >= args.max_attempts:
        return {"task_id": task_id, "status": "gave_up", "attempts": prior_attempts,
                "qa": prior or {}}

    ws = Path(args.workroot) / model_key / task_id
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    shutil.copy(t["analysis_md"], ws / "analysis.md")
    shutil.copy(ARFT_GUIDE, ws / "arft_guide.md")
    (ws / "arft_codes.txt").write_text(P.codes_txt())
    # Deliberately minimal staging: just the analysis, the guide, and the code cheat
    # sheet. Measured: staging a large extra reference file drove cache_read to ~500k
    # tokens per session (every turn re-sends the whole workspace context), at
    # ~$1.00/analysis. The guide (arft_guide.md) already carries the definitions and
    # discrimination rules a classifier needs — keep it that way.
    prior_problems = prior.get("problems") if (prior and not prior.get("ok")) else None
    (ws / "INSTRUCTION.md").write_text(build_instruction(t, model_key, prior_problems))

    prompt = ("Read INSTRUCTION.md in this directory FIRST and follow it completely. "
              "You are fully autonomous; do not ask questions. Write classification.json "
              "here before stopping.")
    cmd = [claude_bin, "--print", "--verbose", "--output-format", "stream-json",
           "--permission-mode", "bypassPermissions", "--max-turns", str(args.max_turns),
           "--model", args.model, "--effort", args.effort, "-p", prompt]
    env = os.environ.copy()
    env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    logf = ws / "session.log"
    started = time.time()
    try:
        with open(logf, "w") as lf:
            proc = subprocess.run(cmd, cwd=ws, env=env, stdin=subprocess.DEVNULL,
                                  stdout=lf, stderr=subprocess.STDOUT,
                                  timeout=args.task_timeout, text=True)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = -1
    dur = round(time.time() - started, 1)

    src = ws / "classification.json"
    if not src.exists():
        infra = _infra_error_status(logf)
        if infra:
            # provider-side blip: do NOT burn an attempt
            return {"task_id": task_id, "status": "infra_error", "api_error_status": infra,
                    "returncode": rc, "dur": dur, "attempts": prior_attempts,
                    "session_log": str(logf)}
        return {"task_id": task_id, "status": "failed", "returncode": rc, "dur": dur,
                "attempts": prior_attempts + 1, "session_log": str(logf)}

    shutil.copy(src, out_json)
    qc = qa.check(out_json)
    return {"task_id": task_id, "status": "done" if qc["ok"] else "qa_fail",
            "returncode": rc, "dur": dur, "qa": qc, "attempts": prior_attempts + 1,
            "out": str(out_json)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True, choices=MODELS)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--force-tasks", default="")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--model", default=os.environ.get("CLAUDE_CODE_MODEL",
                                                      "anthropic/claude-sonnet-5"))
    ap.add_argument("--effort", default="medium")   # extraction, not investigation
    ap.add_argument("--max-turns", type=int, default=14)  # pilot: 10 hit error_max_turns
    ap.add_argument("--task-timeout", type=int, default=900)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--claude-bin", default=None)
    ap.add_argument("--workroot", default=str(HERE / "_ws_arft"))
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    claude_bin = find_claude_bin(args.claude_bin)
    mk = args.model_key
    args._force_set = set(x for x in args.force_tasks.split(",") if x)
    tasks = discover(mk)
    if args.tasks:
        want = set(args.tasks.split(","))
        tasks = [t for t in tasks if t["task_id"] in want]
    if args.n:
        tasks = tasks[:args.n]

    print(f"[arft] model={mk} claude={claude_bin} model_arg={args.model} "
          f"conc={args.concurrency} turns={args.max_turns} tasks={len(tasks)}", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futs = {ex.submit(run_one, t, args, claude_bin, mk): t["task_id"] for t in tasks}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:                       # one crash must not lose the batch
                r = {"task_id": futs[fut], "status": "crashed", "error": repr(e)}
            results.append(r)
            q = r.get("qa", {})
            print(f"[done] {r['task_id'][:24]:24} {r['status']:11} dur={r.get('dur','-')} "
                  f"hits={q.get('n_hits','-')} part={q.get('n_partials','-')} "
                  f"uncov={q.get('n_uncovered','-')} "
                  f"{('PROB=' + ';'.join(q.get('problems', []))) if q.get('problems') else ''}",
                  flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manp = OUT_ROOT / f"{mk}_manifest.json"
    prev = json.load(open(manp)) if manp.exists() else {}
    for r in results:
        prev[r["task_id"]] = r
    json.dump(prev, open(manp, "w"), ensure_ascii=False, indent=2)
    print(f"[arft] DONE {mk} | {dict(Counter(r['status'] for r in results))} | "
          f"manifest={manp}", flush=True)


if __name__ == "__main__":
    main()
