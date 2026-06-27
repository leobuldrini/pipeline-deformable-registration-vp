# Future improvements

Deferred items found while making the repo self-contained. None block running the pipeline;
the canonical path is correct. Listed roughly by priority. Each "script-functionality" change
should be done carefully (byte-/behavior-equivalence checked), like the CerebrA-template fix.

---

## 1. Remove the CerebrA 25-label eval fallback (silent footgun)
**Where:** `run_transmorph.py`, `run_voxelmorph.py`, `run_nodeo_batch.py` (eval paths),
`common/labels.py` (`EVAL_LABELS_CEREBRA`).

**Current behavior (correct on the canonical path):** with `--atlas-seg fastsurfer_seg`
(npz mode), all three models evaluate on `EVAL_LABELS_30` = the TransMorph 30-label protocol
(the thesis metric). Confirmed via `update_results_csv` note "fastsurfer_seg 30 labels".

**The risk:** `EVAL_LABELS_CEREBRA` (25 labels, missing WM/CSF/Choroid) survives as:
- `else:` fallbacks that fire only when **no** atlas-seg is resolved, and
- the legacy **`.mgz` eval branch** in `run_voxelmorph.py:~1698` (hardcoded `EVAL_LABELS_CEREBRA`),
  unreachable in the refactored pipeline (it feeds npz, not an mgz tree).

If `--atlas-seg` ever fails to load (typo / missing file), the code **silently** downgrades to
25-label Dice instead of erroring → wrong-protocol numbers without warning.

**Fix:** make `--atlas-seg` required for eval; **error** (don't fall back) if eval labels can't
be resolved from it; delete the legacy `.mgz` eval branch and the `EVAL_LABELS_CEREBRA`
references. Verify each model's eval still produces identical 30-label results on a sample.

---

## 2. CerebrA naming in docs/metrics (cosmetic)
**Where:** `common/labels.py` comments, `run_nodeo_batch.py` docstring ("CerebrA Dice/protocol"),
`update_results_csv.py` ("CerebrA 25 labels" note), `prepare_atlas_fastsurfer_seg.py` ("why not
CerebrA" rationale).

These name the *label set / rationale*, not a file dependency (the CerebrA atlas dependency is
already removed). Optional: rename "CerebrA protocol" → "25-label subset" for clarity. The
"why not CerebrA" rationale in `prepare_atlas_fastsurfer_seg.py` is useful — keep.

---

## 3. Inert TransMorph-diff "tombstones"
**Where:** `run_transmorph.py` — `TransMorphDiff = DiffBilinear = None`, `CONFIGS_DIFF = {}`, and
the now-unreachable `transmorph-diff` `elif` branches (the `--model` choices were stripped to
`['transmorph']`).

Harmless (unreachable), kept minimal during the refactor. Could delete the dead branches for
readability. Verify `--model transmorph` path is untouched.

---

## 4. Hardcoded constants in phase-5 scripts
**Where:** `phase5_results/compare_models_for_slicer.py`, `compare_vp_models.py` — some
data/checkpoint paths are baked as module constants.

Harmless until those scripts run (the user supplies real paths). Optional: lift to CLI args /
`$REPO`-relative defaults for full portability.

---

## 5. End-to-end verification on a fresh clone
The pipeline has **not** been run start-to-finish from a clean checkout in the phase layout.
Verified so far: every file `py_compile`s; etapa 1 + 2.1 + 2.3 run on the staged sample;
`prepare_mni_template` output is byte-identical after the CerebrA strip.

**Do:** run the staged ~12-exam sample through etapas 1→5 once (RUNBOOK), fix whatever surfaces,
then this section can be deleted. This is the single thing that turns "should run" into "runs".

---

## 6. External-tool version drift
FastSurfer + FastSurfer-LIT are pinned in README (`0b6c508`, LIT `d23f6d0`), but their upstream
APIs can move. If a fresh install breaks, pin/patch against those commits. (Registration code
is self-contained and patched, so lower risk there.)
