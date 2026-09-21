# Scoring scaffolding and the catalysis reference domain

Read this if you want to extend the task suite with a new scored domain. Skip it if you
only want to reproduce the diagnosis — that is
[`agent-as-a-judge/`](../agent-as-a-judge/README.md).

| Path | What it does |
|---|---|
| [`discovery_verifier.py`](discovery_verifier.py) | the rubric scorer behind the `reward` / `conclusion_match` / `soft[<observable>]` fields quoted in the paper's case studies |
| [`discovery_env.py`](discovery_env.py) | domain-agnostic discovery episode skeleton (FRAME → DESIGN → EXECUTE → RESOLVE); imports no chemistry |
| [`domains/catalysis_qe.py`](domains/catalysis_qe.py) | computational catalysis as the first domain plugin |
| [`co_pt_oracle.py`](co_pt_oracle.py), [`recompute_tools.py`](recompute_tools.py) | the live recompute oracle (Quantum ESPRESSO) and its calculator-agnostic tool layer |
| [`../verify/mlip_prefilter.py`](../verify/mlip_prefilter.py) | universal-MLIP prefilter, run before paying for DFT |

## The reward is a gate, not a weighted sum

Open-ended tasks expose no objective and are judged on process. Where a task does have a
metric:

```
reward = correctness × (0.5 + 0.25·significance + 0.25·novelty)
```

`correctness ∈ {0,1}` is a re-executed number matching the held-out gold within
tolerance. Wrong or un-run scores **0**, so the soft dimensions only *rank* rollouts that
already passed. The episode gate has the same shape — `sane ∧ decisive ∧ valid`, not
"produced a number".

## Adding a domain

`discovery_env.py` imports no chemistry at all: the skeleton (frame a tension → design an
experiment → read the result → decide if it is resolved) is domain-agnostic, and
catalysis is the first plugin. Adding a second domain means a new oracle, not a new
driver.

Three recompute tiers are wired up behind
[`../pipelines/rollout/discovery_rollout.py`](../pipelines/rollout/discovery_rollout.py):

| Tier | What it is | Admissible? |
|---|---|---|
| `emt` | instant EMT — plumbing / CI only | never (physically meaningless) |
| `mlip` | CHGNet universal MLIP — cheap prefilter | no (not the honest reward) |
| `qe` | real Quantum ESPRESSO 7.5 PBE-PAW | **yes** — minutes per calc, MPI |

```bash
python pipelines/rollout/discovery_rollout.py --calc emt                    # validate the loop, instant
QE_NP=32 QE_NPOOL=4 python pipelines/rollout/discovery_rollout.py --calc qe # real DFT, admissible
```

## Two invariants, enforced in data rather than prose

- **Nothing is verified until an execution check says so.** `Trajectory.is_admissible()`
  returns `verification.passed`; export refuses the rest.
- **Failure branches are kept.** A non-zero exit or a correction is flagged
  `is_failure_branch=True` and retained as error→recovery supervision in the SFT export.

Both live in the trajectory IR — see [`../ir/README.md`](../ir/README.md).
