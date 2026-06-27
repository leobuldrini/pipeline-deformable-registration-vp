"""
power_monitor.py — Persistent power monitor for PyTorch training.

- Streams readings to a CSV (one line per sample, flushed immediately)
- Resumes/appends to an existing file after a power loss or crash
- Computes a live summary from the full file history on resume
- Atomic-safe: each line is independently valid; no data lost on crash

Usage:
    monitor = PowerMonitor("run_001_power.csv", interval=1.0)
    monitor.start(epoch=0, step=0)
    # ... training ...
    monitor.stop()
    print(monitor.summary())
"""

import csv
import os
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("[PowerMonitor] pynvml not found — GPU power will not be recorded.")
    print("               Install with: pip install pynvml --break-system-packages")

RAPL_PATH = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
CSV_FIELDS_TRAIN = ["timestamp", "epoch", "step", "gpu_w", "cpu_w"]
CSV_FIELDS_BATCH = ["timestamp", "scan_index", "subject", "gpu_w", "cpu_w"]


class PowerMonitor:
    def __init__(self, filepath: str = "power_log.csv", gpu_index: int = 0,
                 interval: float = 1.0, mode: str = "train"):
        """
        Args:
            filepath:  Path to the CSV log file. Appended to if it already exists.
            gpu_index: NVML GPU index (0 for single-GPU systems).
            interval:  Sampling interval in seconds.
            mode:      "train" for epoch/step tracking, "batch" for scan_index/subject tracking.
        """
        self.filepath = Path(filepath)
        self.interval = interval
        self.gpu_index = gpu_index
        self.mode = mode
        self._fields = CSV_FIELDS_BATCH if mode == "batch" else CSV_FIELDS_TRAIN

        self._running = False
        self._thread = None
        self._epoch = 0
        self._step = 0
        self._subject = ""
        self._lock = threading.Lock()

        # NVML handle
        self._nvml_handle = None
        if NVML_AVAILABLE:
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            name = pynvml.nvmlDeviceGetName(self._nvml_handle)
            print(f"[PowerMonitor] GPU detected: {name}")

        # RAPL availability
        self._rapl_ok = os.path.exists(RAPL_PATH)
        if not self._rapl_ok:
            print(f"[PowerMonitor] RAPL not readable at {RAPL_PATH}")
            print("               Run: sudo chmod -R a+r /sys/class/powercap/intel-rapl/")

        # Resume: count existing rows
        self._rows_on_resume = self._count_existing_rows()
        if self._rows_on_resume > 0:
            print(f"[PowerMonitor] Resuming — found {self._rows_on_resume} existing readings in '{self.filepath}'")
        else:
            print(f"[PowerMonitor] Starting fresh log at '{self.filepath}'")

        # Open file in append mode; write header only if new
        self._file = open(self.filepath, "a", newline="", buffering=1)  # line-buffered
        self._writer = csv.DictWriter(self._file, fieldnames=self._fields)
        if self._rows_on_resume == 0:
            self._writer.writeheader()
            self._file.flush()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self, epoch: int = 0, step: int = 0):
        """Start background sampling. Call at the beginning of each epoch."""
        if self._running:
            self.stop()
        with self._lock:
            self._epoch = epoch
            self._step = step
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update_step(self, step: int):
        """Call inside your training loop to record the current step."""
        with self._lock:
            self._step = step

    def update_scan(self, scan_index: int, subject: str = ""):
        """Call inside batch processing loops to record current scan."""
        with self._lock:
            self._step = scan_index
            self._subject = subject

    def stop(self):
        """Stop sampling and flush the file."""
        self._running = False
        if self._thread:
            self._thread.join()
        self._file.flush()

    def close(self):
        """Stop sampling and close the file. Call at end of training."""
        self.stop()
        self._file.close()
        if NVML_AVAILABLE:
            pynvml.nvmlShutdown()

    def summary(self) -> dict:
        """
        Compute summary stats from the *entire* CSV file (including pre-resume rows).
        Safe to call at any time, even mid-training.
        """
        gpu_vals, cpu_vals = [], []
        try:
            with open(self.filepath, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["gpu_w"]:
                        gpu_vals.append(float(row["gpu_w"]))
                    if row["cpu_w"]:
                        cpu_vals.append(float(row["cpu_w"]))
        except FileNotFoundError:
            return {}

        def stats(vals):
            if not vals:
                return {"avg": None, "max": None, "min": None}
            return {
                "avg": round(sum(vals) / len(vals), 2),
                "max": round(max(vals), 2),
                "min": round(min(vals), 2),
            }

        gpu_s = stats(gpu_vals)
        cpu_s = stats(cpu_vals)

        total_samples = len(gpu_vals)
        duration_s = total_samples * self.interval
        energy_wh = None
        if gpu_s["avg"] is not None and cpu_s["avg"] is not None:
            energy_wh = round((gpu_s["avg"] + cpu_s["avg"]) * duration_s / 3600, 4)

        return {
            "total_samples": total_samples,
            "duration_min": round(duration_s / 60, 2),
            "gpu": gpu_s,
            "cpu": cpu_s,
            "total_energy_wh": energy_wh,
        }

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _loop(self):
        while self._running:
            ts = datetime.utcnow().isoformat(timespec="seconds")
            with self._lock:
                epoch = self._epoch
                step = self._step
                subject = self._subject

            gpu_w = self._read_gpu_w()
            cpu_w = self._read_cpu_w()

            gpu_str = round(gpu_w, 2) if gpu_w is not None else ""
            cpu_str = round(cpu_w, 2) if cpu_w is not None else ""

            if self.mode == "batch":
                row = {
                    "timestamp": ts,
                    "scan_index": step,
                    "subject": subject,
                    "gpu_w": gpu_str,
                    "cpu_w": cpu_str,
                }
            else:
                row = {
                    "timestamp": ts,
                    "epoch": epoch,
                    "step": step,
                    "gpu_w": gpu_str,
                    "cpu_w": cpu_str,
                }
            self._writer.writerow(row)
            self._file.flush()   # <-- ensures data hits disk every sample
            os.fsync(self._file.fileno())  # <-- survives power loss

            time.sleep(self.interval)

    def _read_gpu_w(self):
        if not self._nvml_handle:
            return None
        try:
            return pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
        except pynvml.NVMLError:
            return None

    def _read_cpu_w(self):
        if not self._rapl_ok:
            return None
        try:
            e1 = int(Path(RAPL_PATH).read_text())
            time.sleep(0.05)
            e2 = int(Path(RAPL_PATH).read_text())
            return (e2 - e1) / 50_000  # uJ over 0.05s → watts
        except Exception:
            return None

    def _count_existing_rows(self) -> int:
        if not self.filepath.exists():
            return 0
        try:
            with open(self.filepath, newline="") as f:
                return max(0, sum(1 for _ in f) - 1)  # subtract header
        except Exception:
            return 0


# ------------------------------------------------------------------ #
#  Example: plug into a PyTorch training loop                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="power_log.csv", help="CSV output path")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval (s)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10, help="Fake steps per epoch")
    args = parser.parse_args()

    monitor = PowerMonitor(filepath=args.log, interval=args.interval)

    try:
        for epoch in range(args.epochs):
            monitor.start(epoch=epoch, step=0)
            print(f"\n── Epoch {epoch} ──")

            for step in range(args.steps):
                monitor.update_step(step)
                # simulate training work
                time.sleep(0.5)
                print(f"  step {step}", end="\r")

            monitor.stop()
            s = monitor.summary()
            print(f"\n  GPU {s['gpu']['avg']}W avg / {s['gpu']['max']}W peak"
                  f"  |  CPU {s['cpu']['avg']}W avg"
                  f"  |  {s['total_energy_wh']} Wh total")

    except KeyboardInterrupt:
        print("\n[PowerMonitor] Interrupted — data saved.")
    finally:
        monitor.close()
        print(f"\nFinal summary: {monitor.summary()}")
        print(f"Full log: {args.log}")
