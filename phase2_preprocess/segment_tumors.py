#!/usr/bin/env python3
"""
Batch tumor segmentation using extracted BraTS25_1 nnU-Net weights (GPU local).

For each scan in the top-1k CSV:
  1. Co-register POST, T2, FLAIR to orig_nu.mgz (conformed 256³ space)
  2. Run nnU-Net inference on GPU (~5s/scan)
  3. Save conformed-space tumor mask as NIfTI (for FastSurfer-LIT later)
  4. Warp tumor mask to MNI using saved ANTs affine transform
  5. Center-crop to match volume shape
  6. Add tumor_mask (uint8) key to existing .npz

Usage:
    python preprocessing/segment_tumors.py \
        --csv phase1_filter/ranked_4mod.csv \
        --fastsurfer-dir fastsurfer_output/phase2 \
        --npz-dir Voxelmorph/data/yale_phase2_mni_160 \
        --transforms-dir Voxelmorph/data/yale_phase2_mni/transforms \
        --template Atlas/mni_icbm152_t1_padded.nii.gz \
        --tumor-masks-dir tumor_masks_conformed \
        --crop-shape 160 192 224
"""

import argparse
import csv
import os
import shutil
import sys
import tempfile
from pathlib import Path

import ants
import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from power_monitor import PowerMonitor

# nnU-Net environment variables (must be set before import)
BRATS_LOCAL_DIR = Path(__file__).resolve().parent.parent / "brats_local"
os.environ["nnUNet_results"] = str(BRATS_LOCAL_DIR / "results")
os.environ["nnUNet_raw"] = str(BRATS_LOCAL_DIR / "raw")
os.environ["nnUNet_preprocessed"] = str(BRATS_LOCAL_DIR / "preprocessed")


def create_predictor():
    """Initialize nnU-Net predictor ONCE for all scans."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
    )

    model_dir = (
        BRATS_LOCAL_DIR / "results" / "Dataset101_submission"
        / "nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres"
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=("all",),
        checkpoint_name="checkpoint_final.pth",
    )

    return predictor


def find_orig_nu(fastsurfer_dir: str, patient_id: str, date: str) -> str | None:
    """Find orig_nu.mgz for a subject in fastsurfer_output."""
    subject_name = f"{patient_id}_{date}"
    for cohort in ["prevalent", "incident", ""]:
        if cohort:
            candidate = Path(fastsurfer_dir) / cohort / subject_name / "mri" / "orig_nu.mgz"
        else:
            candidate = Path(fastsurfer_dir) / subject_name / "mri" / "orig_nu.mgz"
        if candidate.exists():
            return str(candidate)
    return None


def prepare_nnunet_input(
    orig_nu_path: str,
    post_path: str,
    t2_path: str,
    flair_path: str,
    tmpdir: str,
) -> Path:
    """Co-register modalities to orig_nu.mgz and prepare nnU-Net input."""
    input_dir = Path(tmpdir) / "input"
    input_dir.mkdir(parents=True)
    case_name = "BraTS-MET-00000-000"

    ref = ants.image_read(orig_nu_path)

    # Channel 0: T1 native (orig_nu.mgz)
    ants.image_write(ref, str(input_dir / f"{case_name}_0000.nii.gz"))

    # Co-register other modalities to orig_nu space
    for path, suffix in [(post_path, "_0001"), (t2_path, "_0002"), (flair_path, "_0003")]:
        mov = ants.image_read(path)
        reg = ants.registration(fixed=ref, moving=mov, type_of_transform="Rigid")
        ants.image_write(
            reg["warpedmovout"],
            str(input_dir / f"{case_name}{suffix}.nii.gz"),
        )

    return input_dir


def run_nnunet(predictor, input_dir: Path, output_dir: Path):
    """Run nnU-Net inference on prepared input."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor.predict_from_files(
        list_of_lists_or_source_folder=str(input_dir),
        output_folder_or_list_of_truncated_output_files=str(output_dir),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )
    result = output_dir / "BraTS-MET-00000-000.nii.gz"
    return result if result.exists() else None


def warp_mask_to_mni(
    tumor_mask_path: str,
    fwd_mat_path: str,
    template_path: str,
) -> np.ndarray:
    """Warp tumor mask from conformed space to MNI using saved affine."""
    tumor = ants.image_read(tumor_mask_path)
    template = ants.image_read(template_path)
    warped = ants.apply_transforms(
        fixed=template,
        moving=tumor,
        transformlist=[fwd_mat_path],
        interpolator="nearestNeighbor",
    )
    return warped.numpy()


def center_crop(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Center-crop array to target shape (same logic as crop_to_transmorph.py)."""
    starts = [(s - t) // 2 for s, t in zip(arr.shape, target_shape)]
    slices = tuple(slice(st, st + t) for st, t in zip(starts, target_shape))
    return arr[slices]


def add_tumor_mask_to_npz(npz_path: str, tumor_mask: np.ndarray):
    """Add tumor_mask key to existing .npz file."""
    data = dict(np.load(npz_path))
    data["tumor_mask"] = tumor_mask.astype(np.uint8)
    np.savez(npz_path, **data)


def process_scan(
    row: dict,
    predictor,
    fastsurfer_dir: str,
    transforms_dir: str,
    npz_dir: str,
    template_path: str,
    tumor_masks_dir: str,
    crop_shape: tuple,
) -> dict:
    """Process a single scan: segment tumor, warp to MNI, add to .npz."""
    patient_id = row["patient_id"]
    date = row["date"]
    subject = f"{patient_id}_{date}"

    result = {"subject": subject, "status": "error", "tumor_voxels": 0}

    # Find orig_nu.mgz
    orig_nu = find_orig_nu(fastsurfer_dir, patient_id, date)
    if orig_nu is None:
        result["status"] = "no_fastsurfer"
        return result

    # Check for saved ANTs transform
    fwd_mat = os.path.join(transforms_dir, f"{subject}_fwd.mat")
    if not os.path.exists(fwd_mat):
        result["status"] = "no_transform"
        return result

    # Check .npz exists
    npz_path = os.path.join(npz_dir, f"{subject}.npz")
    if not os.path.exists(npz_path):
        result["status"] = "no_npz"
        return result

    tmpdir = tempfile.mkdtemp(prefix=f"tumor_{subject}_")
    try:
        # 1. Co-register modalities to orig_nu.mgz
        input_dir = prepare_nnunet_input(
            orig_nu, row["post_path"], row["t2_path"], row["flair_path"], tmpdir
        )

        # 2. Run nnU-Net
        output_dir = Path(tmpdir) / "output"
        seg_path = run_nnunet(predictor, input_dir, output_dir)
        if seg_path is None:
            result["status"] = "inference_failed"
            return result

        # 3. Save conformed-space tumor mask (for FastSurfer-LIT later)
        conformed_dst = os.path.join(tumor_masks_dir, f"{subject}_tumor.nii.gz")
        shutil.copy2(str(seg_path), conformed_dst)

        # Check if any tumor was found
        tumor_data = nib.load(str(seg_path)).get_fdata()
        n_tumor = int(np.sum(tumor_data > 0))
        result["tumor_voxels"] = n_tumor

        if n_tumor == 0:
            # No tumor — save empty mask to .npz
            sample = np.load(npz_path)
            empty_mask = np.zeros(sample["vol"].shape, dtype=np.uint8)
            add_tumor_mask_to_npz(npz_path, empty_mask)
            result["status"] = "no_tumor"
            return result

        # 4. Warp tumor mask to MNI
        tumor_mni = warp_mask_to_mni(str(seg_path), fwd_mat, template_path)

        # 5. Center-crop to match volume shape
        tumor_cropped = center_crop(tumor_mni, crop_shape)

        # 6. Add to .npz
        add_tumor_mask_to_npz(npz_path, tumor_cropped)

        result["status"] = "ok"
        return result

    except Exception as e:
        result["status"] = f"error: {e}"
        return result
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Batch tumor segmentation with BraTS25_1 (GPU local)"
    )
    parser.add_argument("--csv", required=True, help="Path to ranked_4mod.csv")
    parser.add_argument("--fastsurfer-dir", required=True,
                        help="Root of fastsurfer_output/")
    parser.add_argument("--npz-dir", required=True,
                        help="Directory with preprocessed .npz files")
    parser.add_argument("--transforms-dir", required=True,
                        help="Directory with saved ANTs _fwd.mat files")
    parser.add_argument("--template", required=True,
                        help="MNI template NIfTI (padded, skull-stripped)")
    parser.add_argument("--tumor-masks-dir", default="tumor_masks_conformed",
                        help="Output dir for conformed-space tumor masks")
    parser.add_argument("--crop-shape", nargs=3, type=int, default=[160, 192, 224],
                        help="Target crop shape (must match .npz volumes)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index in CSV (for resuming)")
    parser.add_argument("--power-log", default="segment_tumors_power.csv",
                        help="Power monitoring CSV output")
    args = parser.parse_args()

    os.makedirs(args.tumor_masks_dir, exist_ok=True)
    crop_shape = tuple(args.crop_shape)

    # Load scan list
    with open(args.csv) as f:
        scans = list(csv.DictReader(f))
    print(f"Loaded {len(scans)} scans from {args.csv}")

    # Initialize predictor ONCE
    print("Initializing nnU-Net predictor...")
    predictor = create_predictor()

    # Power monitoring
    monitor = PowerMonitor(filepath=args.power_log, interval=1.0, mode="batch")
    monitor.start()

    # Process scans
    stats = {"ok": 0, "no_tumor": 0, "no_fastsurfer": 0,
             "no_transform": 0, "no_npz": 0, "error": 0}

    try:
        for i, row in enumerate(scans[args.start:], start=args.start):
            subject = f"{row['patient_id']}_{row['date']}"
            monitor.update_scan(i, subject)

            # Skip if already processed
            npz_path = os.path.join(args.npz_dir, f"{subject}.npz")
            if os.path.exists(npz_path):
                try:
                    d = np.load(npz_path)
                    if "tumor_mask" in d:
                        stats["ok"] += 1
                        continue
                except Exception:
                    pass

            print(f"[{i+1}/{len(scans)}] {subject}...", end=" ", flush=True)

            result = process_scan(
                row, predictor, args.fastsurfer_dir, args.transforms_dir,
                args.npz_dir, args.template, args.tumor_masks_dir, crop_shape,
            )

            status = result["status"].split(":")[0]
            stats[status] = stats.get(status, 0) + 1

            if result["status"] == "ok":
                print(f"tumor={result['tumor_voxels']} voxels")
            elif result["status"] == "no_tumor":
                print("no tumor found")
            else:
                print(result["status"])

    except KeyboardInterrupt:
        print("\n[Interrupted] Data saved.")
    finally:
        monitor.stop()
        s = monitor.summary()
        monitor.close()

    # Summary
    print(f"\n{'=' * 50}")
    print("TUMOR SEGMENTATION SUMMARY")
    print(f"{'=' * 50}")
    for k, v in sorted(stats.items()):
        print(f"  {k:>15s}: {v}")
    print(f"  {'total':>15s}: {sum(stats.values())}")
    if s.get("total_energy_wh") is not None:
        print(f"\n  Energy: {s['total_energy_wh']} Wh over {s['duration_min']:.1f} min")
        print(f"  GPU: {s['gpu']['avg']}W avg / {s['gpu']['max']}W peak")
        print(f"  CPU: {s['cpu']['avg']}W avg")


if __name__ == "__main__":
    main()
