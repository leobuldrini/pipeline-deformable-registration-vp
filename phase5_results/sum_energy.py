"""
sum_energy.py — Summarise total Wh consumed during training from power_log.csv.

Each row in the CSV is a 1-second power sample (gpu_w, cpu_w in watts).
Energy per sample: W × (1 s / 3600) = Wh.
"""

import csv
import sys
from pathlib import Path

POWER_LOG = Path(__file__).parent / "checkpoints" / "power_log.csv"
INTERVAL_S = 1.0  # sampling interval used by PowerMonitor


def sum_energy(filepath: Path = POWER_LOG) -> dict:
    gpu_total_wh = 0.0
    cpu_total_wh = 0.0
    gpu_samples = 0
    cpu_samples = 0
    total_samples = 0

    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_samples += 1
            if row["gpu_w"]:
                gpu_total_wh += float(row["gpu_w"]) * INTERVAL_S / 3600
                gpu_samples += 1
            if row["cpu_w"]:
                cpu_total_wh += float(row["cpu_w"]) * INTERVAL_S / 3600
                cpu_samples += 1

    combined_wh = gpu_total_wh + cpu_total_wh
    duration_h = total_samples * INTERVAL_S / 3600

    return {
        "gpu_wh": round(gpu_total_wh, 4),
        "cpu_wh": round(cpu_total_wh, 4),
        "total_wh": round(combined_wh, 4),
        "gpu_samples": gpu_samples,
        "cpu_samples": cpu_samples,
        "duration_h": round(duration_h, 4),
    }


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else POWER_LOG
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    stats = sum_energy(path)
    print(f"Power log : {path}")
    print(f"Duration  : {stats['duration_h']:.4f} h  ({stats['gpu_samples']} samples)")
    print(f"GPU       : {stats['gpu_wh']:.4f} Wh")
    print(f"CPU       : {stats['cpu_wh']:.4f} Wh")
    print(f"Total     : {stats['total_wh']:.4f} Wh")
