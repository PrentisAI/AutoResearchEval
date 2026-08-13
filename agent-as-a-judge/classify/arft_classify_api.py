#!/usr/bin/env python3
"""
arft_classify_api.py — same job as arft_classify_cc.py, executed as a single direct
OpenRouter chat completion instead of a headless Claude Code session.

Why both exist: this is an extraction task, not an investigation. The analysis.md
already contains the evidence; the model only has to map it onto the 45 ARFT codes. A
Claude Code session pays for a system prompt, tool definitions and multi-turn
re-caching on every analysis — measured at $0.669 each (cache_creation alone was
$0.234, and output ran to 16.6k tokens for a JSON of ~19 findings). One-shot
completion measured far cheaper for identical output. Everything downstream is
shared: the same `arft_guide.md`, the same `arft_qa_check` gate, the same output
paths and manifest, the same `arft_aggregate.py`.

Keep arft_classify_cc.py around: it is the right tool if the task ever needs the model
to go read other files (raw trajectories, agent code) rather than just the analysis.

Usage:
    export ARFT_OPENROUTER_KEY=...        # or ~/.openrouter_key
    python3 arft_classify_api.py --model-key <model> --resume --concurrency 24
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arft_patterns as P        # noqa: E402
import arft_qa_check as qa       # noqa: E402
import arft_classify_cc as cc    # noqa: E402  (shared discover/paths/model list)

MODELS = cc.MODELS
OUT_ROOT = cc.OUT_ROOT
ARFT_GUIDE = cc.ARFT_GUIDE

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
RETRY_STATUSES = {408, 409, 429, 500, 502, 503, 504, 529}

_print_lock = threading.Lock()


def load_key():
    k = os.environ.get("ARFT_OPENROUTER_KEY", "").strip()
    if not k:
        f = Path.home() / ".openrouter_key"
        if f.exists():
            k = f.read_text().strip()
    if not k:
        k = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not k:
        raise SystemExit("no OpenRouter key: set ARFT_OPENROUTER_KEY or ~/.openrouter_key")
    return k


SYSTEM = (
    "You are a failure-pattern classifier. You will be given an **already-completed "
    "deep-dive trajectory analysis** (analysis.md) and an operational guide for a "
    "45-pattern taxonomy (ARFT). Your job is to **classify and count, not to "
    "re-judge the trajectory**.\n"
    "Output a single JSON object only — no markdown code fences, no explanatory "
    "preamble or postscript."
)


def build_prompt(guide, analysis, t, model_key):
    return f"""{guide}

---

# Analysis to classify (task_id={t['task_id']}, model={model_key}, source_set={t['source_set']})

{analysis}

---

# Output requirements

Output **one JSON object** strictly following this schema (no code fence, no preamble):

{{"task_id":"{t['task_id']}","model":"{model_key}","source_set":"{t['source_set']}",
 "overall_severity":"high|medium|low|none",
 "summary":"2-4 sentences: what the agent did, main failure/success",
 "hits":[{{"code":"C.1","confidence":0.9,"evidence":"Section C, issue 8: the headline comes from a self-written DGP","why":"matches the C.1 definition because…"}}],
 "partials":[{{"code":"D.3","confidence":0.5,"evidence":"Section D, issue 26: no SEM/significance awareness anywhere","why":"a real gap the analysis judges as not conclusion-changing"}}],
 "iron_rules_cited":[9,6],
 "stage_notes":{{"A":"…","B":"…","C":"…","D":"…","E":"…","F":"…","X":"…"}},
 "uncovered":[]}}

Hard requirements:
- `code` may only use the 45 codes in the guide's code table (dotted form, e.g. `C.1`).
  Put hits in `hits`, qualified concerns in `partials`; **omit codes that miss
  entirely from both lists**. The same code must never appear in both lists.
- Every `evidence` is 1-3 sentences and **must be locatable back into analysis.md**:
  include a section name/issue number/short quote (e.g. `Section C, issue 8`,
  `issue 12`, `Sentence-by-Sentence Checklist row 13`, `One-Line Verdict`).
  Labels that can't be located get dropped.
- **Polarity**: fabrication/contamination-adjacent vocabulary in this kind of
  document is overwhelmingly used to CLEAR the agent. Mention ≠ hit.
- **`## Credit Due` is the fair-credit section** — it must not be your sole evidence
  source.
- Judgments the analysis retracted itself (in `## Retraction / Correction Log`) count
  as a miss.
- `overall_severity`: cannot be `none` if any `hits` are present.
- `iron_rules_cited`: every iron-rule number cited anywhere in analysis.md, as an
  integer array (`[]` if none).
- `uncovered`: judge strictly, most trajectories should have `[]`. If you do fill it
  in, every item needs `mechanism`/`nearest`/`why_each_fails`.
- **JSON validity**: if you need to quote a phrase inside a string value, use a
  distinct quotation style (e.g. Unicode “curly” quotes) rather than a bare ASCII
  `"` — an unescaped `"` inside a string terminates it early and breaks the JSON.
  Do not put a raw newline inside a string value either.
"""


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.M)


def repair_inner_quotes(s):
    """Escape ASCII double quotes used as ad hoc inner quotation marks inside string
    values.

    The single most common malformed-JSON mode observed while building this
    classifier: the model writes `...points to "no effect" — but was...` — a bare `"`
    around a quoted phrase, inside a JSON string, unescaped. That terminates the
    string early and the whole object fails to parse.

    Heuristic: while inside a string, a `"` only really closes it if the next
    non-whitespace character is a structural JSON token (`,` `}` `]` `:`). Anything
    else means it was meant as punctuation, so escape it.
    """
    out, i, n, instr, esc = [], 0, len(s), False, False
    while i < n:
        ch = s[i]
        if not instr:
            out.append(ch)
            if ch == '"':
                instr = True
            i += 1
            continue
        if esc:
            out.append(ch); esc = False; i += 1; continue
        if ch == "\\":
            out.append(ch); esc = True; i += 1; continue
        if ch == '"':
            j = i + 1
            while j < n and s[j] in " \t\r\n":
                j += 1
            if j >= n or s[j] in ",}]:":
                out.append(ch); instr = False
            else:
                out.append('\\"')          # punctuation quote -> escape it
            i += 1
            continue
        if ch in "\n\r":
            out.append("\\n" if ch == "\n" else "\\r")   # raw newline inside a string
            i += 1
            continue
        out.append(ch); i += 1
    return "".join(out)


def extract_json(text):
    """Pull the JSON object out of a completion, tolerating fences and stray prose."""
    if not text:
        return None
    s = _FENCE.sub("", text).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return json.loads(repair_inner_quotes(s))
    except Exception:
        pass
    # fall back to the outermost balanced {...}
    start = s.find("{")
    if start < 0:
        return None
    depth, instr, esc = 0, False, False
    for i, ch in enumerate(s[start:], start):
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
            continue
        if ch == '"':
            instr = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = s[start:i + 1]
                for cand in (blob, repair_inner_quotes(blob)):
                    try:
                        return json.loads(cand)
                    except Exception:
                        continue
                return None
    return None


def call_model(client, key, model, prompt, max_tokens, timeout, reasoning_tokens):
    """-> (parsed_json_or_None, usage_dict, error_string_or_None, is_infra)."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "usage": {"include": True},
        # Reasoning budget is the single biggest quality lever here, measured on a
        # hand-checked case against a Claude Code reference labelling:
        #   reasoning off   ~40% recall  <- drops several genuinely-present patterns
        #   reasoning 3k    ~80% recall
        # Do not disable it to save money; the labels stop being usable.
        "reasoning": {"max_tokens": reasoning_tokens},
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        r = client.post(ENDPOINT, json=body, headers=headers, timeout=timeout)
    except Exception as e:
        return None, {}, f"transport: {e!r}", True
    if r.status_code != 200:
        return None, {}, f"http {r.status_code}: {r.text[:200]}", r.status_code in RETRY_STATUSES
    try:
        d = r.json()
    except Exception as e:
        return None, {}, f"non-json response: {e!r}", True
    if "error" in d and not d.get("choices"):
        msg = str(d["error"])[:200]
        code = (d["error"] or {}).get("code") if isinstance(d.get("error"), dict) else None
        return None, {}, f"api error: {msg}", code in RETRY_STATUSES
    usage = d.get("usage") or {}
    try:
        choice = d["choices"][0]
        text = choice["message"]["content"]
    except Exception:
        return None, usage, f"unexpected shape: {str(d)[:200]}", True

    doc = extract_json(text)
    if doc is not None:
        return doc, usage, None, False

    # Successful HTTP call but unparseable body. Say WHY — an earlier version returned
    # err=None here, which surfaced as `status=failed, error=None` and hid the real
    # cause (the completion was cut off at max_tokens on the longest analyses).
    fin = choice.get("finish_reason") or choice.get("native_finish_reason")
    if fin == "length":
        return None, usage, "truncated: finish_reason=length", "truncated"
    return None, usage, f"unparseable content (finish_reason={fin}): {str(text)[:160]}", False


def cost_of(usage):
    """USD from an OpenRouter usage block (it reports `cost` directly when available)."""
    if usage.get("cost") is not None:
        return float(usage["cost"])
    pin, pout = 2e-6, 10e-6                       # anthropic/claude-sonnet-5 list
    return usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout


def run_one(t, args, key, guide, model_key, client):
    task_id = t["task_id"]
    out_root = OUT_ROOT / model_key
    out_root.mkdir(parents=True, exist_ok=True)
    out_json = out_root / f"{task_id}.json"
    forced = task_id in args._force_set

    prior = None
    if out_json.exists():
        prior = qa.check(out_json)
        if args.resume and not forced and prior["ok"]:
            return {"task_id": task_id, "status": "skipped", "qa": prior}

    prior_attempts = cc._prior_attempts(model_key, task_id)
    if not forced and prior_attempts >= args.max_attempts:
        return {"task_id": task_id, "status": "gave_up", "attempts": prior_attempts,
                "qa": prior or {}}

    analysis = Path(t["analysis_md"]).read_text(errors="ignore")
    prompt = build_prompt(guide, analysis, t, model_key)
    if prior and not prior.get("ok"):
        prompt += ("\n\n# ⚠️ Prior attempt was rejected by QA\nPrior problems: **"
                   + "; ".join(prior["problems"][:8])
                   + "**. Fix them specifically and re-output the complete JSON.\n")

    started = time.time()
    spend = 0.0
    doc = err = None
    max_tokens = args.max_tokens
    for attempt in range(args.http_retries + 1):
        doc, usage, err, infra = call_model(client, key, args.model, prompt,
                                            max_tokens, args.timeout,
                                            args.reasoning_tokens)
        spend += cost_of(usage)
        if doc is not None:
            break
        if attempt == args.http_retries:
            break
        if infra == "truncated":
            # Retrying at the same ceiling would truncate identically. The longest
            # analyses (up to 82k chars) legitimately produce more findings, so give
            # the completion more room instead of just waiting.
            max_tokens = min(int(max_tokens * 1.8), 64000)
            continue
        if not infra:
            break
        time.sleep(min(2 ** attempt * 2, 30))

    dur = round(time.time() - started, 1)
    if doc is None:
        # Provider/transport trouble must not burn the attempt budget; a genuine
        # content failure must.
        infra_like = bool(err) and (err.startswith("transport")
                                    or err.startswith("truncated")
                                    or "http 5" in err or "http 429" in err
                                    or "http 408" in err or "http 409" in err)
        return {"task_id": task_id, "status": "infra_error" if infra_like else "failed",
                "error": err, "dur": dur, "cost": round(spend, 4),
                "attempts": prior_attempts if infra_like else prior_attempts + 1}

    # normalise the fields we control, so a model slip doesn't fail QA on a technicality
    doc.setdefault("task_id", task_id)
    doc["model"] = model_key
    doc["source_set"] = t["source_set"]
    out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    qc = qa.check(out_json)
    return {"task_id": task_id, "status": "done" if qc["ok"] else "qa_fail",
            "dur": dur, "cost": round(spend, 4), "qa": qc,
            "attempts": prior_attempts + 1, "out": str(out_json)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True, choices=MODELS)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--force-tasks", default="")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--model", default=os.environ.get("ARFT_MODEL",
                                                      "anthropic/claude-sonnet-5"))
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--reasoning-tokens", type=int, default=3000)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--http-retries", type=int, default=3)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    key = load_key()
    guide = ARFT_GUIDE.read_text()
    mk = args.model_key
    args._force_set = set(x for x in args.force_tasks.split(",") if x)
    tasks = cc.discover(mk)
    if args.tasks:
        want = set(args.tasks.split(","))
        tasks = [t for t in tasks if t["task_id"] in want]
    if args.n:
        tasks = tasks[:args.n]

    print(f"[arft-api] model={mk} llm={args.model} conc={args.concurrency} "
          f"tasks={len(tasks)}", flush=True)
    results = []
    limits = httpx.Limits(max_connections=args.concurrency + 4,
                          max_keepalive_connections=args.concurrency + 4)
    with httpx.Client(limits=limits) as client:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futs = {ex.submit(run_one, t, args, key, guide, mk, client): t["task_id"]
                    for t in tasks}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"task_id": futs[fut], "status": "crashed", "error": repr(e)}
                results.append(r)
                q = r.get("qa", {})
                with _print_lock:
                    print(f"[done] {r['task_id'][:24]:24} {r['status']:11} "
                          f"dur={r.get('dur','-'):>6} ${r.get('cost',0):.4f} "
                          f"hits={q.get('n_hits','-')} part={q.get('n_partials','-')} "
                          f"{('PROB=' + ';'.join(q.get('problems', [])[:2])) if q.get('problems') else ''}"
                          f"{(' ERR=' + str(r.get('error'))[:90]) if r.get('error') else ''}",
                          flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manp = OUT_ROOT / f"{mk}_manifest.json"
    prev = json.load(open(manp)) if manp.exists() else {}
    for r in results:
        prev[r["task_id"]] = r
    json.dump(prev, open(manp, "w"), ensure_ascii=False, indent=2)
    spent = sum(r.get("cost", 0) or 0 for r in results)
    print(f"[arft-api] DONE {mk} | {dict(Counter(r['status'] for r in results))} | "
          f"spend=${spent:.2f} | manifest={manp}", flush=True)


if __name__ == "__main__":
    main()
