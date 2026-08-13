#!/bin/bash
# Self-healing ARFT classification over your corpus's analysis.md files, via
# OpenRouter and headless Claude Code sessions (prefer run_all_arft_api.sh unless
# you specifically need agentic file-reading). Loops with --resume until every
# analysis has a QA-passing classification.json, then aggregates.
#
# Key handling: never inline a key here. Set ARFT_OPENROUTER_KEY (preferred, so a
# stray ambient OPENROUTER_API_KEY left over from some other workflow can't be
# silently picked up), else OPENROUTER_API_KEY, else ~/.openrouter_key.
set -u
cd "$(dirname "$0")"

# Precedence: explicit ARFT_* > keyfile > ambient env. The ambient OPENROUTER_API_KEY is
# LAST on purpose: as of 2026-08-04 the one exported in this sandbox belongs to an
# unrelated workflow and is dead (401), while ~/.openrouter_key is live. Preferring the
# ambient var would 401 every request — and because 401 is classified as a retryable
# infra blip, the loop below would spin all its passes doing nothing.
KEY="${ARFT_OPENROUTER_KEY:-$(cat ~/.openrouter_key 2>/dev/null)}"
KEY="${KEY:-${OPENROUTER_API_KEY:-}}"
[ -n "$KEY" ] || { echo "no OpenRouter key: set ARFT_OPENROUTER_KEY or ~/.openrouter_key" >&2; exit 1; }

# Preflight: fail fast on a bad key rather than burning passes on 401s.
if ! KEY="$KEY" python3 - <<'PREFLIGHT'
import json, os, sys, urllib.request
k = os.environ["KEY"]
try:
    req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                 headers={"Authorization": f"Bearer {k}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        d = json.load(r)["data"]
except Exception as e:
    print(f"PREFLIGHT FAIL: OpenRouter rejected the key ({e})", file=sys.stderr)
    sys.exit(1)
lim, used = d.get("limit"), d.get("usage")
print(f"preflight ok: key={d.get('label')} usage=${used} "
      f"limit={'unlimited' if lim is None else lim}")
PREFLIGHT
then
  echo "aborting: fix the OpenRouter key before running 800 classifications" >&2; exit 1
fi

export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_API_KEY="$KEY"
export ANTHROPIC_AUTH_TOKEN="$KEY"

MODEL="${MODEL:-anthropic/claude-sonnet-5}"
EFFORT="${EFFORT:-medium}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_TURNS="${MAX_TURNS:-14}"
TASK_TIMEOUT="${TASK_TIMEOUT:-900}"
# Optional: restrict the run, e.g. TASKS=W4388327516,echonet_lvef  (pilot use)
TASKS_ARG=""
[ -n "${TASKS:-}" ] && TASKS_ARG="--tasks $TASKS"

CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"
[ -x "$CLAUDE_BIN" ] || CLAUDE_BIN="$(command -v claude)"
[ -x "$CLAUDE_BIN" ] || { echo "claude CLI not found; set CLAUDE_BIN" >&2; exit 1; }
echo "using claude: $CLAUDE_BIN"

COMMON="--claude-bin $CLAUDE_BIN --resume --model $MODEL --effort $EFFORT --concurrency $CONCURRENCY \
        --max-turns $MAX_TURNS --task-timeout $TASK_TIMEOUT $TASKS_ARG"

# Auto-discovered from whatever model-named subdirectories exist under the corpus
# (same logic arft_classify_cc.py uses) — no hardcoded model list to keep in sync.
MODELS="$(python3 -c 'import arft_classify_cc as c; print(" ".join(c.MODELS))')"
if [ -z "$MODELS" ]; then
  echo "no models found under \$AAJ_CORPUS_DIR (default ./corpus) — nothing to classify" >&2
  exit 1
fi

MAX_PASSES="${MAX_PASSES:-6}"
OUT="${AAJ_OUT_DIR:-results}"
SENTINEL="$OUT/DONE.txt"
mkdir -p "$OUT"
rm -f "$SENTINEL"

echo "########## arft classify (cc): model=$MODEL conc=$CONCURRENCY turns=$MAX_TURNS $(date) ##########"

for pass in $(seq 1 "$MAX_PASSES"); do
  echo "############### ARFT CLASSIFY PASS $pass / $MAX_PASSES  $(date) ###############"
  for mk in $MODELS; do
    python3 arft_classify_cc.py --model-key "$mk" $COMMON
  done
  echo "----- status after pass $pass -----"
  if python3 arft_status.py; then
    echo "ALL CLASSIFIED after pass $pass  $(date)" | tee "$SENTINEL"
    break
  fi
  echo "residual remain; cooling 45s"; sleep 45
done

echo "########## arft classify EXIT $(date) — running aggregate ##########"
python3 arft_aggregate.py || true
python3 arft_status.py || true
