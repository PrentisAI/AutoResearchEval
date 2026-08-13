# SciData Engine action schema (induced from the agent survey + provenance)

Companion to `registry.json` (same dir). This is the **action-space spec** for the
SciData Engine IR — `id` here == `Action.name` in `ir/trajectory.py`. It defines
how the harvested tool/skill surface normalizes into the IR action vocabulary.
Load it in code via `from ir.actions import REGISTRY, ACTIONS, get`.

## How this was built (data, not vibes)

Per CLAUDE.md §15.3 ("动作词汇从真实数据归纳，不是拍脑袋设计"), the vocabulary is
**induced from two complementary corpora**:

- **Corpus A — 143 GitHub science agents** (this survey): gives the *breadth* of
  what science agents expose to an LLM. 6725 tool entries + 2175 skills →
  frequency/functional analysis (`ANALYSIS.md`).
- **Corpus B — real provenance diffs** (mc2d/ACWF restart chains, Custodian, MLIP
  campaigns): gives the *verifiable recovery* vocabulary scicoder already induces
  (§15.3 named SCF tools, §13 DFT-accel actions). Already in the repo.

A tells us the action *space*; B tells us the action *labels that survive execution
verification*. The draft keeps only the intersection that scicoder can **own a
deterministic verifier for** (§8.3 / §14: external libs = action space, the gate
is always ours).

## What the survey shows (`ANALYSIS.md`)

12 categories emerge across all agents (count = all tool entries):

| category | count | what it is | scicoder verifiable? |
|---|--:|---|---|
| RETRIEVE | 2800 | search/query/get from DB, literature, APIs | mostly NO (literature) / some YES (DB record schema) |
| PREPARE | 563 | build/pack/solvate/equilibrate/write-inputs | YES (parser dry-run, density/overlap, T/P) |
| VALIDATE | 502 | convergence/stability/triage/check | **YES — this is our §6 layer** |
| COMPUTE | 471 | analyze/calculate/fit observables | **YES — recompute from output** |
| CONVERT | 314 | name↔SMILES, parse, format | YES (RDKit round-trip) |
| SIMULATE | 273 | run DFT/MD/docking/relax | **YES — §6 physics gate, the core** |
| VISUALIZE | 227 | plot/render | NO (aux) |
| PREDICT | 199 | train/classify/score (MLIP, ML) | YES (loss finite, MAE vs holdout) |
| REPORT | 110 | summarize/export | NO (aux) |
| EDIT | 90 | fix/adjust/patch inputs | **YES — recovery actions, the gold (§1.3)** |
| DESIGN | 83 | select/propose/plan (active learning) | YES if selection reproducible |
| EVALUATE | 73 | benchmark/compare | YES |

Key reads:
- **RETRIEVE dominates** the wild (search/`get`/DB). For scicoder most of it is
  NON_VERIFIABLE (literature lookup) → context only, **not** dataset actions.
  The verifiable slice is DB-record fetches whose fields we can re-check.
- The **verifiable core for scicoder is SIMULATE + COMPUTE + VALIDATE + EDIT** —
  exactly the run→analyze→check→recover loop the §13 flagship already uses.
- The wild's PREPARE/CONVERT layer is real and we under-represent it today
  (we mostly start from a given structure); worth adding.

## Normalized action entry

Extends the existing `ir.Action{name, params}` — adds a typed, verifier-bound layer
(shape borrowed from Biomni's `{name, required_parameters:[{name,type,default}]}`):

```
action_id      # canonical snake_case == Action.name
category        # one of the 12 (controlled enum)
verb / domain   # comp-chem | md | dft | materials | cross
params[]        # Biomni-style typed list (name, type, required, default)
observation     # what Observation carries back
verifier        # scicoder-OWNED deterministic gate, or NON_VERIFIABLE
verifiable      # true => may enter dataset (§1.1); false => context/aux only
seen_in[]       # provenance: which surveyed repos exemplify it (auditability §1.4)
maps_to         # existing scicoder action/tool, if any
```

`registry.json` seeds **26 actions** across the categories for the
comp-chem / MD / DFT domain (23 verifiable, 13 already map to existing scicoder code).

## Mapping to what already exists in the repo

| existing scicoder | draft action(s) | note |
|---|---|---|
| §15.3 named SCF tools | `damp_charge_mixing`, `custodian_correct`, `triage_failure` | already induced from provenance; survey corroborates (materials-simulation-skills.simulationfailuretriage) |
| §13 DFT-accel space | `run_dft`,`train_mlip`,`validate_mlip`,`run_md`,`select_uncertain`,`custodian_correct` | 1:1; survey adds typed params + `seen_in` |
| harness/ase_tools | `build_structure`,`relax_structure` | survey adds `pack_system`, `equilibrate` we lack |
| harness/l2_tasks | `analyze_dft_output`,`custodian_fix` | survey's COMPUTE/EDIT split matches L2 task types |
| reconstruct/tool_lift | `triage_failure`,`damp_charge_mixing` | the "diagnose (problem only) → recover" split (§15.3) is in both |

## Gaps the survey exposes (candidates to add)

1. **PREPARE layer** — `pack_system` (Packmol), `equilibrate`, `write_md_inputs`.
   MD agents (MDCrow, mdclaw, motus) treat setup as first-class verifiable actions;
   scicoder currently jumps to relax/run. Adds the front of the water-MD trajectory.
2. **COMPUTE leaf actions for MD** — `compute_rdf`, `compute_msd_diffusion`,
   `fit_kinetics` (motus has the cleanest impls: arrhenius_fit, 1st/2nd-order).
   These are deterministically re-checkable → ideal L2 analysis tasks.
3. **CONVERT** — `name_to_smiles`, `compute_descriptor` (chemcrow/RDKit), RDKit
   round-trip = free deterministic verifier.

## Open questions before wiring into `ir/`

- Make `category` a hard enum on `Action`, or keep free-form `name` + a side
  registry? (Side registry is less invasive and matches `tool_lift`'s current style.)
- Phonon/DFPT sub-space (§15.3 TODO) — survey gave no signal; stays provenance-only.
- RETRIEVE: keep verifiable DB fetches as actions, drop literature search to
  context — confirm this is the right §6 boundary.

## Files
- `ANALYSIS.md` / `analysis.json` — frequency + functional analysis (143 repos)
- `registry.json` — the seed registry (26 actions, typed + verifier-bound)
- `FINDINGS.md` — the qualitative survey writeup
- `inventory.json` / `extracted/*.md` — raw per-repo tool/skill inventory
