# AutoResearchEval — Data-Generation Pipeline

Minimal, self-contained code for generating agent research-trajectory data,
covering two task families:

- **open-ended discovery** — mine a paper's *discovery pattern*
  (premise → tension → move), materialize it into an open-ended research task,
  and run a live-oracle rollout.
- **rigor / recompute** — reconstruct verifiable trajectories from computational
  provenance (e.g. mc2d DFT) with deterministic recomputation gates.

## Layout

| Module | Purpose |
|---|---|
| `adapters/` | source ingestion — OpenAlex metadata, MinerU-parsed paper corpus |
| `reconstruct/` | discovery-pattern extraction, move generation, LLM (OpenRouter) client, GT anchoring |
| `harness/` | rollout environment, recompute oracle/verifier, catalysis-QE domain |
| `ir/` | unified trajectory IR + the typed action registry (`ir/actions/`) |
| `export/` | trajectory → SFT/ReAct message export |
| `verify/` | MLIP pre-filter physical checks |
| `examples/` | pipeline entry points (corpus crawl → pattern mine → task export → rollout) |

## Setup

```bash
pip install -r requirements.lock
export OPENROUTER_API_KEY=...   # teacher LLM; never commit keys
```

## Entry points

```bash
python examples/crawl_tiered_corpus.py        # crawl + tier papers (OpenAlex)
python examples/discovery_pattern_mine.py     # papers -> discovery patterns
python examples/materialize_discovery_tasks.py
python examples/export_discovery_tasks.py
python examples/discovery_rollout.py          # live-oracle rollout
python examples/mc2d_reconstruct_admissible.py  # rigor line
```

No API keys, data, or model weights are included. All secrets are read from the
environment / a gitignored `.env`.
