#!/bin/bash
# Self-healing ARFT classification over your corpus's analysis.md files, using the
# direct OpenRouter executor (arft_classify_api.py). This is the default/cheaper path.
#
# Use run_all_arft_cc.sh (Claude Code sessions) instead only if the classifier ever
# needs to go read files beyond the analysis itself — measured ~3.7x more expensive per
# analysis for the same output on a pure classification task.
#
# Key precedence: ARFT_OPENROUTER_KEY > ~/.openrouter_key > OPENROUTER_API_KEY. The
# ambient OPENROUTER_API_KEY is last on purpose — don't let some other tool's exported
# key silently get picked up here.
set -u
cd "$(dirname "$0")"

KEY="${ARFT_OPENROUTER_KEY:-$(cat ~/.openrouter_key 2>/dev/null)}"
KEY="${KEY:-${OPENROUTER_API_KEY:-}}"
[ -n "$KEY" ] || { echo "no OpenRouter key: set ARFT_OPENROUTER_KEY or ~/.openrouter_key" >&2; exit 1; }
export ARFT_OPENROUTER_KEY="$KEY"

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
lim = d.get("limit")
print(f"preflight ok: key={d.get('label')} usage=${d.get('usage')} "
      f"limit={'unlimited' if lim is None else lim}")
PREFLIGHT
then
  echo "aborting: fix the OpenRouter key before running 800 classifications" >&2; exit 1
fi

MODEL="${MODEL:-anthropic/claude-sonnet-5}"
CONCURRENCY="${CONCURRENCY:-24}"
REASONING="${REASONING:-3000}"
MAX_TOKENS="${MAX_TOKENS:-16000}"
TASKS_ARG=""
[ -n "${TASKS:-}" ] && TASKS_ARG="--tasks $TASKS"

COMMON="--resume --model $MODEL --concurrency $CONCURRENCY \
        --reasoning-tokens $REASONING --max-tokens $MAX_TOKENS $TASKS_ARG"

# Auto-discovered from whatever model-named subdirectories exist under the corpus
# (same logic arft_classify_cc.py uses) — no hardcoded model list to keep in sync.
MODELS="$(python3 -c 'import arft_classify_cc as c; print(" ".join(c.MODELS))')"
if [ -z "$MODELS" ]; then
  echo "no models found under \$AAJ_CORPUS_DIR (default ./corpus) — nothing to classify" >&2
  exit 1
fi

MAX_PASSES="${MAX_PASSES:-4}"
OUT="${AAJ_OUT_DIR:-results}"
SENTINEL="$OUT/DONE.txt"
mkdir -p "$OUT"
rm -f "$SENTINEL"

echo "########## arft classify (api): $MODEL conc=$CONCURRENCY reasoning=$REASONING $(date) ##########"

for pass in $(seq 1 "$MAX_PASSES"); do
  echo "############### PASS $pass / $MAX_PASSES  $(date) ###############"
  for mk in $MODELS; do
    python3 arft_classify_api.py --model-key "$mk" $COMMON
  done
  echo "----- status after pass $pass -----"
  if python3 arft_status.py; then
    echo "ALL CLASSIFIED after pass $pass  $(date)" | tee "$SENTINEL"
    break
  fi
  echo "residual remain; cooling 30s"; sleep 30
done

echo "########## EXIT $(date) — aggregating ##########"
python3 arft_aggregate.py || true
python3 arft_verify.py || true
python3 arft_status.py || true
