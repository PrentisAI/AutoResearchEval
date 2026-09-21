# Trajectory IR and the action vocabulary

`ir/` is the convergence point of the engine: every source lowers into one
`Trajectory`, and everything downstream (verify, filter, export) reads that one shape.

```
source ──adapter──▶ IR ──reconstruct──▶ verify ──filter──▶ export
```

| Path | What it is |
|---|---|
| [`trajectory.py`](trajectory.py) | the IR itself — `Trajectory`, `Step`, `Action`, `Verification`, `Reward`, and the enums that constrain them |
| [`actions/registry.json`](actions/registry.json) | 26 verifier-bound **execution** actions |
| [`actions/discovery_registry.json`](actions/discovery_registry.json) | 10 atomic **reasoning** moves |
| [`actions/SCHEMA.md`](actions/SCHEMA.md) | the registry schema and how the vocabulary was induced |
| [`../export/to_sft_react.py`](../export/to_sft_react.py) | IR → SFT/ReAct messages with loss masking |

## Action vocabulary

Two registries, induced from data (143 GitHub science agents + real provenance diffs)
rather than designed top-down.

**`registry.json`** — 26 verifier-bound execution actions across 11 categories
(`build_structure`, `run_dft`, `check_convergence`, `triage_failure`, `train_mlip`, …).
Each pins a verifier this repo owns: external tools supply the action space,
verification is ours.

**`discovery_registry.json`** — 10 atomic reasoning moves in three phases:

| Phase | Moves |
|---|---|
| FRAME | `survey_consensus`, `identify_tension`, `formulate_question`, `propose_hypothesis` |
| PROBE | `select_system`, `choose_method`, `run_calculation` |
| RESOLVE | `compare_reference`, `interpret_result`, `draw_conclusion` |

`run_calculation` grounds into the execution actions above; the terminal reward grounds
into the recompute handle described in [`../harness/README.md`](../harness/README.md).
