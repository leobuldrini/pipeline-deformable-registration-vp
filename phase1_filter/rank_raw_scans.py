#!/usr/bin/env python3
"""
Rank ALL raw PRE scans in Dataset/MRI/ by voxel spacing distance from
isotropic (1.0, 1.0, 1.0)mm. Then filter the top N scans that have all
4 modalities (PRE, POST, T2, FLAIR).

Outputs:
  - raw_voxel_ranking_full.csv: complete ranking of all PRE scans
  - ranked_4mod.csv: top-k with all 4 modalities + full paths
"""

import argparse
import csv
import os
from pathlib import Path

import nibabel as nib
import numpy as np


def read_spacing(nifti_path: str) -> tuple[float, ...] | None:
    """Read voxel spacing from a NIfTI header (fast, no data loaded)."""
    try:
        img = nib.load(nifti_path)
        return tuple(np.round(img.header.get_zooms()[:3], 4))
    except Exception:
        return None


def euclidean_distance_from_iso(spacing: tuple[float, ...]) -> float:
    """Euclidean distance of spacing vector from (1.0, 1.0, 1.0)."""
    return float(np.sqrt(sum((s - 1.0) ** 2 for s in spacing)))


def find_modality_path(date_dir: Path, prefix: str, modality: str) -> str | None:
    """Find a modality file matching the scan prefix in the date directory."""
    for f in date_dir.iterdir():
        if f.name.endswith(f"_{modality}.nii.gz") and f.name.startswith(prefix):
            if "_resampled" not in f.name:
                return str(f)
    return None


def scan_all_raw_pre(mri_root: str) -> list[dict]:
    """Walk Dataset/MRI/ and collect info from every *_PRE.nii.gz scan."""
    mri_root = Path(mri_root)
    scans = []

    patient_dirs = sorted(d for d in mri_root.iterdir() if d.is_dir())

    for i, patient_dir in enumerate(patient_dirs, 1):
        patient_id = patient_dir.name

        for date_dir in sorted(patient_dir.iterdir()):
            if not date_dir.is_dir():
                continue

            # Find PRE files (exclude resampled)
            pre_files = [
                f for f in date_dir.iterdir()
                if f.name.endswith("_PRE.nii.gz") and "_resampled" not in f.name
            ]

            for pre_file in pre_files:
                # Extract prefix (everything before _PRE.nii.gz)
                prefix = pre_file.name.replace("_PRE.nii.gz", "")

                spacing = read_spacing(str(pre_file))
                if spacing is None:
                    continue

                # Check sibling modalities
                post_path = find_modality_path(date_dir, prefix, "POST")
                t2_path = find_modality_path(date_dir, prefix, "T2")
                flair_path = find_modality_path(date_dir, prefix, "FLAIR")

                scans.append({
                    "patient_id": patient_id,
                    "date": date_dir.name,
                    "prefix": prefix,
                    "pre_path": str(pre_file),
                    "spacing_x": spacing[0],
                    "spacing_y": spacing[1],
                    "spacing_z": spacing[2],
                    "euclidean_dist": euclidean_distance_from_iso(spacing),
                    "has_post": post_path is not None,
                    "has_t2": t2_path is not None,
                    "has_flair": flair_path is not None,
                    "post_path": post_path or "",
                    "t2_path": t2_path or "",
                    "flair_path": flair_path or "",
                })

        if i % 200 == 0:
            print(f"  Scanned {i}/{len(patient_dirs)} patients "
                  f"({len(scans)} PRE scans found)...")

    return scans


def rank_and_filter(scans: list[dict], top_k: int) -> tuple[list[dict], list[dict]]:
    """Sort by euclidean_dist ascending, filter top_k with all 4 modalities."""
    # Sort all scans
    scans.sort(key=lambda s: s["euclidean_dist"])
    for i, s in enumerate(scans, 1):
        s["rank"] = i

    # Filter: walk ranking, collect until top_k with 4 modalities
    filtered = []
    for s in scans:
        if s["has_post"] and s["has_t2"] and s["has_flair"]:
            filtered.append(s)
            if len(filtered) >= top_k:
                break

    return scans, filtered


def write_csv(rows: list[dict], path: str, fieldnames: list[str]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(scans: list[dict], filtered: list[dict]):
    dists = [s["euclidean_dist"] for s in scans]
    n_4mod = sum(1 for s in scans
                 if s["has_post"] and s["has_t2"] and s["has_flair"])

    print(f"\n{'=' * 70}")
    print("SCAN RANKING SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total PRE scans found:     {len(scans)}")
    print(f"Scans with all 4 mods:     {n_4mod}")
    print(f"Selected (top-k filtered): {len(filtered)}")

    print(f"\nEuclidean distance from (1,1,1) — all scans:")
    print(f"  Min:    {min(dists):.4f}")
    print(f"  Max:    {max(dists):.4f}")
    print(f"  Mean:   {np.mean(dists):.4f}")
    print(f"  Median: {np.median(dists):.4f}")

    if filtered:
        f_dists = [s["euclidean_dist"] for s in filtered]
        print(f"\nEuclidean distance — selected scans:")
        print(f"  Min:    {min(f_dists):.4f}")
        print(f"  Max:    {max(f_dists):.4f}")
        print(f"  Mean:   {np.mean(f_dists):.4f}")
        print(f"  Median: {np.median(f_dists):.4f}")

    # Modality availability
    print(f"\nModality availability (all scans):")
    print(f"  POST:  {sum(1 for s in scans if s['has_post']):>6d} "
          f"({100*sum(1 for s in scans if s['has_post'])/len(scans):.1f}%)")
    print(f"  T2:    {sum(1 for s in scans if s['has_t2']):>6d} "
          f"({100*sum(1 for s in scans if s['has_t2'])/len(scans):.1f}%)")
    print(f"  FLAIR: {sum(1 for s in scans if s['has_flair']):>6d} "
          f"({100*sum(1 for s in scans if s['has_flair'])/len(scans):.1f}%)")

    # Top 5 selected
    print(f"\nTop 5 selected scans:")
    for s in filtered[:5]:
        print(f"  #{s['rank']:>5d}  {s['patient_id']}  {s['date']}  "
              f"({s['spacing_x']:.3f}, {s['spacing_y']:.3f}, {s['spacing_z']:.3f})  "
              f"dist={s['euclidean_dist']:.4f}")

    # Worst 5 selected
    if len(filtered) >= 5:
        print(f"\nLast 5 selected scans (worst isotropy in selection):")
        for s in filtered[-5:]:
            print(f"  #{s['rank']:>5d}  {s['patient_id']}  {s['date']}  "
                  f"({s['spacing_x']:.3f}, {s['spacing_y']:.3f}, {s['spacing_z']:.3f})  "
                  f"dist={s['euclidean_dist']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Rank raw PRE scans by isotropy and filter top-k with 4 modalities"
    )
    parser.add_argument(
        "--mri-dir",
        default="Dataset/MRI",
        help="Root directory of raw MRI data",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Number of top scans to select (with all 4 modalities)",
    )
    parser.add_argument(
        "--output-dir",
        default="code",
        help="Directory for output CSVs",
    )
    args = parser.parse_args()

    print(f"Scanning {args.mri_dir} for PRE scans...")
    scans = scan_all_raw_pre(args.mri_dir)
    print(f"Found {len(scans)} PRE scans total")

    print(f"\nRanking and filtering top {args.top_k} with 4 modalities...")
    full_ranking, filtered = rank_and_filter(scans, args.top_k)

    # Write full ranking
    full_fields = [
        "rank", "patient_id", "date", "prefix", "pre_path",
        "spacing_x", "spacing_y", "spacing_z", "euclidean_dist",
        "has_post", "has_t2", "has_flair",
        "post_path", "t2_path", "flair_path",
    ]
    full_csv = os.path.join(args.output_dir, "raw_voxel_ranking_full.csv")
    write_csv(full_ranking, full_csv, full_fields)
    print(f"Full ranking written to {full_csv}")

    # Write filtered top-k
    filtered_fields = [
        "rank", "patient_id", "date", "prefix",
        "pre_path", "post_path", "t2_path", "flair_path",
        "spacing_x", "spacing_y", "spacing_z", "euclidean_dist",
        "has_post", "has_t2", "has_flair",
    ]
    filtered_csv = os.path.join(args.output_dir, "ranked_4mod.csv")
    write_csv(filtered, filtered_csv, filtered_fields)
    print(f"Filtered top-{args.top_k} written to {filtered_csv}")

    print_summary(full_ranking, filtered)


if __name__ == "__main__":
    main()
