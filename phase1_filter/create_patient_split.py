#!/usr/bin/env python3
"""
Create train/val/test split by PATIENT (not by scan) to prevent data leakage.

Takes the top 500 ranked scans. Of the 452 that need LIT, only includes those
where LIT has already finished. The 48 without tumor are always included.

Usage:
    python code/create_patient_split.py \
        --data-dir Voxelmorph/data/yale_phase2_mni_160 \
        --ranking phase1_filter/ranked_4mod.csv \
        --lit-priority code/lit_priority_500.txt \
        --lit-output-dir lit_output \
        --top-k 500
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Create patient-level split from top-500 with LIT status check"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--ranking", default="phase1_filter/ranked_4mod.csv")
    parser.add_argument("--lit-priority", default="code/lit_priority_500.txt")
    parser.add_argument("--lit-output-dir", default="lit_output")
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    split_path = os.path.join(args.data_dir, "data_split.json")
    if os.path.exists(split_path) and not args.force:
        print(f"Split already exists: {split_path}")
        print("Use --force to overwrite")
        return

    # Load top-k from ranking
    with open(args.ranking) as f:
        ranked = list(csv.DictReader(f))
    top_k = [f"{r['patient_id']}_{r['date']}" for r in ranked[:args.top_k]]
    print(f"Top {args.top_k} ranked scans: {len(top_k)}")

    # Load LIT priority list (scans that need LIT)
    with open(args.lit_priority) as f:
        needs_lit = set(line.strip() for line in f if line.strip())

    # Check which LIT scans are done
    lit_done = set()
    lit_pending = set()
    for subject in needs_lit:
        lit_result = os.path.join(args.lit_output_dir, subject,
                                  "inpainting_volumes", "inpainting_result.nii.gz")
        npz_path = os.path.join(args.data_dir, f"{subject}.npz")
        if os.path.exists(lit_result) and os.path.exists(npz_path):
            if os.path.getmtime(npz_path) > os.path.getmtime(lit_result):
                lit_done.add(subject)
            else:
                lit_pending.add(subject)
        else:
            lit_pending.add(subject)

    # Build eligible scan list
    eligible = []
    excluded = []
    for subject in top_k:
        npz_path = os.path.join(args.data_dir, f"{subject}.npz")
        if not os.path.exists(npz_path):
            excluded.append((subject, "no_npz"))
            continue

        if subject in needs_lit:
            if subject in lit_done:
                eligible.append(subject)
            else:
                excluded.append((subject, "lit_pending"))
        else:
            # No tumor — doesn't need LIT, always eligible
            eligible.append(subject)

    no_tumor = len([s for s in eligible if s not in needs_lit])
    lit_included = len([s for s in eligible if s in lit_done])

    print(f"\nEligibility:")
    print(f"  No tumor (always eligible): {no_tumor}")
    print(f"  LIT done:                   {lit_included}")
    print(f"  LIT pending (excluded):     {len(lit_pending)}")
    print(f"  Total eligible:             {len(eligible)}")

    if len(eligible) < 10:
        print("ERROR: Too few eligible scans. Wait for more LIT to finish.")
        return

    # Group by patient
    patients = defaultdict(list)
    for subject in eligible:
        parts = subject.rsplit("_", 1)
        patient_id = parts[0] if len(parts) == 2 else subject
        patients[patient_id].append(subject)

    patient_ids = sorted(patients.keys())
    print(f"\nPatients: {len(patient_ids)}")

    # Shuffle patients and split
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(patient_ids))

    n = len(patient_ids)
    n_train = int(n * args.ratios[0])
    n_val = int(n * args.ratios[1])

    train_pids = [patient_ids[i] for i in indices[:n_train]]
    val_pids = [patient_ids[i] for i in indices[n_train:n_train + n_val]]
    test_pids = [patient_ids[i] for i in indices[n_train + n_val:]]

    train_files = sorted(f"{s}.npz" for pid in train_pids for s in patients[pid])
    val_files = sorted(f"{s}.npz" for pid in val_pids for s in patients[pid])
    test_files = sorted(f"{s}.npz" for pid in test_pids for s in patients[pid])

    # Verify no overlap
    assert not set(train_pids) & set(val_pids), "Train/val overlap!"
    assert not set(train_pids) & set(test_pids), "Train/test overlap!"
    assert not set(val_pids) & set(test_pids), "Val/test overlap!"

    print(f"\nSplit (by patient, no leakage):")
    print(f"  Train: {len(train_pids)} patients, {len(train_files)} scans")
    print(f"  Val:   {len(val_pids)} patients, {len(val_files)} scans")
    print(f"  Test:  {len(test_pids)} patients, {len(test_files)} scans")
    print(f"  Total: {len(eligible)} scans")

    saved = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }
    with open(split_path, "w") as f:
        json.dump(saved, f, indent=2)
    print(f"\nSaved to {split_path}")


if __name__ == "__main__":
    main()
