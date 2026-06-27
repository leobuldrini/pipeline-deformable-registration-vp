#!/usr/bin/env python3
"""
Generate a TSV job list for FastSurfer batch processing from the top-1k ranking.

Reads ranked_4mod.csv (output of rank_raw_scans.py) and emits a TSV
with one row per scan: patient_id, date, pre_path (absolute).

Usage:
    python code/generate_fastsurfer_jobs_v2.py \
        --input-csv phase1_filter/ranked_4mod.csv \
        --output-tsv code/fastsurfer_jobs_v2.tsv
"""

import argparse
import csv
import os


def main():
    parser = argparse.ArgumentParser(
        description="Generate FastSurfer job list from top-1k ranking"
    )
    parser.add_argument(
        "--input-csv", default="phase1_filter/ranked_4mod.csv",
        help="Top-1k CSV from rank_raw_scans.py",
    )
    parser.add_argument(
        "--output-tsv", default="code/fastsurfer_jobs_v2.tsv",
        help="Output TSV job list",
    )
    args = parser.parse_args()

    with open(args.input_csv, newline="") as f:
        scans = list(csv.DictReader(f))

    jobs = []
    skipped = []

    for row in scans:
        pre_path = os.path.abspath(row["pre_path"])
        if not os.path.exists(pre_path):
            skipped.append((row["patient_id"], row["date"], "file not found"))
            continue
        jobs.append({
            "patient_id": row["patient_id"],
            "date": row["date"],
            "pre_path": pre_path,
        })

    # Write TSV
    with open(args.output_tsv, "w", newline="") as f:
        f.write("patient_id\tdate\tpre_path\n")
        for job in jobs:
            f.write(f"{job['patient_id']}\t{job['date']}\t{job['pre_path']}\n")

    print(f"Generated {len(jobs)} jobs from {len(scans)} scans")
    print(f"Output: {args.output_tsv}")

    if skipped:
        print(f"Skipped {len(skipped)} scans:")
        for pid, date, reason in skipped:
            print(f"  {pid}_{date}: {reason}")


if __name__ == "__main__":
    main()
