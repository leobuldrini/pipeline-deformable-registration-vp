#!/usr/bin/env python3
"""
Rebuild data_split.json for fastsurfer_preprocessed_mni_160 using only
the top 1k most isovolumetric scans.

Reuses the same patient-level splitting logic from
Voxelmorph/run_voxelmorph.py::get_or_create_split (80/10/10, seed=42).

Excluded .npz files are moved to a sibling directory
(fastsurfer_preprocessed_mni_160_excluded/).
"""

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict

import numpy as np


def subject_id_from_npz(filename):
    """Extract patient ID from npz filename.

    e.g. YG_0B4NV6E3KEZQ_2015-09-29.npz -> YG_0B4NV6E3KEZQ
    """
    stem = filename.replace(".npz", "")
    return stem.rsplit("_", 1)[0]


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild data_split.json from top-1k isovolumetric scans")
    parser.add_argument("--top-csv",
                        default="code/voxel_spacing_top1k_mni_160.csv",
                        help="CSV with top-1k ranked scans")
    parser.add_argument("--data-dir",
                        default="Voxelmorph/data/fastsurfer_preprocessed_mni_160",
                        help="Directory with .npz files")
    parser.add_argument("--excluded-dir",
                        default="Voxelmorph/data/fastsurfer_preprocessed_mni_160_excluded",
                        help="Directory to move excluded .npz files to")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        help="Train/val/test ratios (default: 0.8 0.1 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split (default: 42)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without moving files")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    excluded_dir = os.path.abspath(args.excluded_dir)

    # 1. Read top-1k scan names
    top_subjects = set()
    with open(args.top_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            top_subjects.add(row["subject"])  # e.g. YG_XXX_2015-09-29
    print(f"Top-1k scans loaded: {len(top_subjects)}")

    # 2. List all .npz files in data_dir
    all_npz = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))
    print(f"Total .npz in data_dir: {len(all_npz)}")

    # Partition into keep / exclude
    keep_npz = []
    exclude_npz = []
    for fname in all_npz:
        stem = fname.replace(".npz", "")
        if stem in top_subjects:
            keep_npz.append(fname)
        else:
            exclude_npz.append(fname)

    print(f"  Keep:    {len(keep_npz)}")
    print(f"  Exclude: {len(exclude_npz)}")

    # 3. Group kept scans by patient (same logic as get_or_create_split)
    subj_files = defaultdict(list)
    for fname in keep_npz:
        pid = subject_id_from_npz(fname)
        subj_files[pid].append(fname)

    subjects = sorted(subj_files.keys())
    print(f"\nPatients represented in top-1k: {len(subjects)}")

    # 4. Shuffle patients and split (mirrors get_or_create_split)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(n * args.ratios[0])
    n_val = int(n * args.ratios[1])

    train_subjs = subjects[:n_train]
    val_subjs = subjects[n_train:n_train + n_val]
    test_subjs = subjects[n_train + n_val:]

    split = {
        "train": [f for s in train_subjs for f in subj_files[s]],
        "val":   [f for s in val_subjs   for f in subj_files[s]],
        "test":  [f for s in test_subjs  for f in subj_files[s]],
    }

    print(f"\nNew split ({len(train_subjs)}/{len(val_subjs)}/{len(test_subjs)} patients):")
    print(f"  Train: {len(split['train'])} scans")
    print(f"  Val:   {len(split['val'])} scans")
    print(f"  Test:  {len(split['test'])} scans")
    print(f"  Total: {sum(len(v) for v in split.values())} scans")

    if args.dry_run:
        print("\n[DRY RUN] No files moved or written.")
        return

    # 5. Write new data_split.json
    split_path = os.path.join(data_dir, "data_split.json")
    # Back up old split
    if os.path.exists(split_path):
        backup = split_path + ".bak"
        shutil.copy2(split_path, backup)
        print(f"\nBacked up old split to {backup}")

    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"Wrote new data_split.json to {split_path}")

    # 6. Move excluded .npz files
    os.makedirs(excluded_dir, exist_ok=True)
    moved = 0
    for fname in exclude_npz:
        src = os.path.join(data_dir, fname)
        dst = os.path.join(excluded_dir, fname)
        shutil.move(src, dst)
        moved += 1
        if moved % 500 == 0:
            print(f"  Moved {moved}/{len(exclude_npz)}...")
    print(f"Moved {moved} excluded .npz files to {excluded_dir}")


if __name__ == "__main__":
    main()
