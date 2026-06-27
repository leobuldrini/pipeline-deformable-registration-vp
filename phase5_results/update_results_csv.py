"""Aggregate all model eval results into results_all_models.csv.

Usage:
    python update_results_csv.py                    # s2s-intra (default)
    python update_results_csv.py --mode atlas       # atlas-to-scan
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

TCC = Path(__file__).resolve().parent.parent

# (model, variant) -> result file per mode
RESULT_FILES = {
    "s2s-intra": {
        ("NODEO",       "baseline"): TCC / "result/nodeo_s2s_intra/results.json",
        ("NODEO",       "+VP"):      TCC / "result/nodeo_s2s_intra_vp/results.json",
        ("TransMorph",  "baseline"): TCC / "Transmorph min/checkpoints_baseline_s2s_intra/eval_results_scan_to_scan_intra.json",
        ("TransMorph",  "+VP"):      TCC / "Transmorph min/checkpoints_vp_s2s_intra/eval_results_scan_to_scan_intra.json",
        ("VoxelMorph",  "baseline"): TCC / "checkpoints_s2s_intra/eval_results_scan_to_scan_intra.json",
        ("VoxelMorph",  "+VP"):      TCC / "checkpoints_vp_s2s_intra/eval_results_scan_to_scan_intra.json",
    },
    "atlas": {
        ("NODEO",       "baseline"): TCC / "result/nodeo_baseline/results.json",
        ("NODEO",       "+VP"):      TCC / "result/nodeo_vp/results.json",
        ("TransMorph",  "baseline"): TCC / "Transmorph min/checkpoints_baseline/eval_results.json",
        ("TransMorph",  "+VP"):      TCC / "Transmorph min/checkpoints_vp/eval_results.json",
        ("VoxelMorph",  "baseline"): TCC / "checkpoints_s2s_intra/eval_results_atlas_to_scan.json",
        ("VoxelMorph",  "+VP"):      TCC / "checkpoints_vp_s2s_intra/eval_results_atlas_to_scan.json",
    },
}

CSV_PATHS = {
    "s2s-intra": TCC / "results_all_models.csv",
    "atlas":     TCC / "results_all_models_atlas.csv",
}

FIELDNAMES = [
    "model", "variant", "mode", "n_pairs", "n_labels",
    "dice_mean", "dice_std", "dice_trimmed_mean", "dice_trimmed_std",
    "baseline_dice_mean",
    "neg_jac_pct_mean", "neg_jac_pct_std",
    "stsr_n_raw", "stsr_n_trimmed", "stsr_trimmed_mean", "stsr_trimmed_std",
    "tvcf_eligible", "tvcf_n", "tvcf_trimmed_mean", "tvcf_trimmed_std",
    "lvcr_mean", "lvcr_std",
    "notes",
]


def parse(model, variant, path):
    with open(path) as f:
        d = json.load(f)

    stsr_dist = d.get("stsr_distribution") or {}
    tvcf_dist = d.get("tvcf_distribution") or {}
    lvcr_dist = d.get("lvcr_distribution") or {}
    dice_dist = d.get("dice_distribution") or {}

    per_label = d.get("dice_per_label") or {}
    n_labels = len(per_label) or len(d.get("eval_labels") or [])

    # handle old NODEO format: num_subjects instead of num_pairs, no distributions
    n_pairs = d.get("num_pairs") or d.get("num_subjects")

    # STSR: prefer distribution trimmed stats, fall back to raw mean/std
    stsr_n_raw     = stsr_dist.get("n") or d.get("stsr_n")
    stsr_n_trimmed = stsr_dist.get("n_trimmed")
    if stsr_dist.get("mean_trimmed") is not None:
        stsr_trim_mean = round(stsr_dist["mean_trimmed"], 4)
        stsr_trim_std  = round(stsr_dist["std_trimmed"], 4)
    elif d.get("stsr_mean") is not None:
        stsr_trim_mean = round(d["stsr_mean"], 4)
        stsr_trim_std  = round(d.get("stsr_std", 0), 4)
    else:
        stsr_trim_mean = stsr_trim_std = None

    notes_parts = []
    if n_labels == 30:
        notes_parts.append("fastsurfer_seg 30 labels")
    elif n_labels == 25:
        notes_parts.append("CerebrA 25 labels")
    elif n_labels:
        notes_parts.append(f"{n_labels} labels")

    stsr_type = d.get("stsr_type", "")
    if stsr_type and stsr_type != "dong":
        notes_parts.append(f"stsr_type={stsr_type} (not Dong — incomparable)")
    if not stsr_dist and d.get("stsr_n"):
        notes_parts.append("old format: no trimmed STSR/TVCF/LVCR stats")

    mode = d.get("eval_mode") or d.get("mode") or "atlas-to-scan"
    # NODEO neg_jac was stored as fraction (0-1) in runs before the Apr-26 commit that
    # added the 100× factor. Files newer than that commit already store percentage.
    _NODEO_COMMIT_DATE = datetime(2026, 4, 26, tzinfo=timezone.utc)
    file_mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
    nodeo_needs_scale = model == "NODEO" and file_mtime < _NODEO_COMMIT_DATE

    return {
        "model":             model,
        "variant":           variant,
        "mode":              mode,
        "n_pairs":           n_pairs,
        "n_labels":          n_labels or None,
        "dice_mean":         round(d["dice_mean"], 4),
        "dice_std":          round(d["dice_std"], 4),
        "dice_trimmed_mean": round(dice_dist.get("mean_trimmed", d["dice_mean"]), 4),
        "dice_trimmed_std":  round(dice_dist.get("std_trimmed", d["dice_std"]), 4),
        "baseline_dice_mean":round(d["baseline_dice_mean"], 4),
        "neg_jac_pct_mean":  round(d["neg_jac_pct_mean"] * (100 if nodeo_needs_scale else 1), 4),
        "neg_jac_pct_std":   round(d["neg_jac_pct_std"]  * (100 if nodeo_needs_scale else 1), 4),
        "stsr_n_raw":        stsr_n_raw,
        "stsr_n_trimmed":    stsr_n_trimmed,
        "stsr_trimmed_mean": stsr_trim_mean,
        "stsr_trimmed_std":  stsr_trim_std,
        "tvcf_eligible":     d.get("tvcf_eligible"),
        "tvcf_n":            tvcf_dist.get("n") or d.get("tvcf_n"),
        "tvcf_trimmed_mean": round(tvcf_dist["mean_trimmed"], 4) if tvcf_dist.get("mean_trimmed") else None,
        "tvcf_trimmed_std":  round(tvcf_dist["std_trimmed"], 4) if tvcf_dist.get("std_trimmed") else None,
        "lvcr_mean":         round(lvcr_dist["mean"], 4) if lvcr_dist.get("mean") is not None else None,
        "lvcr_std":          round(lvcr_dist["std"], 4) if lvcr_dist.get("std") is not None else None,
        "notes":             "; ".join(notes_parts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["s2s-intra", "atlas"], default="s2s-intra")
    args = parser.parse_args()

    files = RESULT_FILES[args.mode]
    csv_path = CSV_PATHS[args.mode]
    rows = {}
    updated = []
    skipped = []

    for key, path in files.items():
        model, variant = key
        if not path.exists():
            skipped.append(f"{model} {variant}: {path.name} not found")
            rows[key] = {
                "model": model, "variant": variant,
                "notes": "NEVER_RUN or NEEDS_RERUN",
                **{k: "" for k in FIELDNAMES if k not in ("model", "variant", "notes")},
            }
            continue
        try:
            rows[key] = parse(model, variant, path)
            updated.append(f"{model} {variant}")
        except Exception as e:
            skipped.append(f"{model} {variant}: parse error — {e}")

    sorted_rows = [rows[k] for k in files if k in rows]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted_rows)

    print(f"Written [{args.mode}]: {csv_path}")
    print(f"Updated: {', '.join(updated)}")
    if skipped:
        print(f"Skipped: {'; '.join(skipped)}")


if __name__ == "__main__":
    main()
