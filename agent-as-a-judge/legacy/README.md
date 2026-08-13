# Legacy: the 29-leaf taxonomy

This directory holds two reference documents from an earlier version of the failure
taxonomy — the **29-leaf, undotted** code system (`A1`, `A2`, … `X6`):

- `failure_taxonomy_en.md` — the overall framework: 6 lifecycle stages (A–F) crossed
  with 5 root mechanisms (M1–M5), plus a cross-cutting `X` layer.
- `failure_taxonomy_leaf_guide.md` — per-leaf operational definitions (Definition /
  Typical evidence / Confused-with / Do-NOT-label).

## Why this is here, and why it's superseded

The canonical taxonomy for this pipeline is now **ARFT** (the AutoResearch Failure
Taxonomy), a 45-pattern dotted system defined in `../classify/arft_patterns.py` and
`../classify/arft_guide.md`. It was revised from this one after classifying a batch of
real trajectories against it and finding real coverage gaps — five new leaves were
added, one was folded into an existing leaf, and the root-cause axis was redesigned
around four cognitive-obligation pillars instead of five mechanism categories.

**⚠️ Codes collide in meaning between the two systems.** `C1` in this system means
"impl bugs"; `C.1` in ARFT means "Circular Validation & Shortcut Reliance". The dot is
not decorative — never mix labels from the two systems in one table, and don't assume a
leaf with a similar-looking code means the same thing.

## No runnable classifier ships for this taxonomy

The original classifier for the 29-leaf system was tightly coupled to this project's
internal benchmark corpora (hardcoded paths to specific run directories that don't
exist outside that environment) and isn't included here. If you want to classify
against this taxonomy instead of ARFT, you'd adapt
`../classify/arft_classify_cc.py`/`arft_classify_api.py` to use these definitions and
the 29-leaf code list instead of `arft_patterns.py` — the harness (workspace staging,
QA-gated resume, aggregation) is reusable; only the taxonomy content and valid-code list
need to change.
