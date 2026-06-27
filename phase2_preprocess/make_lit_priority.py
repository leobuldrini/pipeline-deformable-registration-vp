#!/usr/bin/env python3
"""
Generate the LIT priority list = exams that have a tumor and therefore need
FastSurfer-LIT inpainting (Etapa 3). Exams with an empty tumor_mask are skipped.

Run after Etapa 2.4 (segment_tumors.py adds the `tumor_mask` key to each .npz).

Usage:
    python phase2_preprocess/make_lit_priority.py \
        --npz-dir data/yale_phase2_mni_160 \
        --output phase1_filter/lit_priority.txt

Output: one subject id per line (e.g. YG_XXX_2015-09-29), tumor-bearing exams only.
Consumed by phase3_inpaint/run_fastsurfer_lit.py (--priority-list) and
phase1_filter/create_patient_split.py (--lit-priority).
"""

import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="List tumor-bearing exams (non-empty tumor_mask) that need LIT")
    parser.add_argument("--npz-dir", required=True,
                        help="Directory of preprocessed .npz (with tumor_mask key)")
    parser.add_argument("--output", default="phase1_filter/lit_priority.txt",
                        help="Output text file (one subject id per line)")
    args = parser.parse_args()

    npz_files = sorted(f for f in os.listdir(args.npz_dir) if f.endswith(".npz"))
    if not npz_files:
        print(f"ERROR: no .npz found in {args.npz_dir}")
        return

    needs_lit = []
    no_tumor = []
    missing_key = []
    for fname in npz_files:
        subject = fname[:-len(".npz")]
        with np.load(os.path.join(args.npz_dir, fname)) as data:
            if "tumor_mask" not in data:
                missing_key.append(subject)
                continue
            if int(data["tumor_mask"].sum()) > 0:
                needs_lit.append(subject)
            else:
                no_tumor.append(subject)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for subject in needs_lit:
            f.write(subject + "\n")

    print(f"Scanned {len(npz_files)} exams in {args.npz_dir}")
    print(f"  needs LIT (tumor present): {len(needs_lit)}")
    print(f"  no tumor (skip LIT):       {len(no_tumor)}")
    if missing_key:
        print(f"  WARNING: no tumor_mask key (run Etapa 2.4 first): {len(missing_key)}")
    print(f"Wrote {len(needs_lit)} subjects to {args.output}")


if __name__ == "__main__":
    main()
