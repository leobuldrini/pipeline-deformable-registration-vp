#!/usr/bin/env python3
"""
FastSurfer-LIT pipeline: lesion inpainting + re-segmentation for improved labels.

Only processes scans where tumor segmentation found tumors (non-empty mask).

For each scan with tumors:
  1. Run LIT inpainting: orig_nu.mgz + tumor_mask → inpainted T1
  2. Re-run FastSurfer --seg_only --no_biasfield on inpainted T1
  3. ANTs affine re-register improved seg to MNI
  4. Replace 'seg' key in .npz (keep original 'vol' and 'tumor_mask')

Usage:
    python preprocessing/run_fastsurfer_lit.py \
        --tumor-masks-dir tumor_masks_conformed \
        --fastsurfer-dir fastsurfer_output/phase2 \
        --fastsurfer-bin FastSurfer/run_fastsurfer.sh \
        --template Atlas/mni_icbm152_t1_padded.nii.gz \
        --npz-dir Voxelmorph/data/yale_phase2_mni_160 \
        --lit-output-dir lit_output \
        --crop-shape 160 192 224
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import ants
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from power_monitor import PowerMonitor


# Labels to merge (same as preprocess_fastsurfer.py)
LH_CORTICAL = list(range(1002, 1036))
RH_CORTICAL = list(range(2002, 2036))


def find_scans_with_tumors(tumor_masks_dir: str) -> list[dict]:
    """Find all conformed-space tumor masks that are non-empty."""
    scans = []
    for mask_file in sorted(glob.glob(os.path.join(tumor_masks_dir, "*_tumor.nii.gz"))):
        name = os.path.basename(mask_file).replace("_tumor.nii.gz", "")
        # Quick check: file size > threshold means non-empty
        img = nib.load(mask_file)
        data = img.get_fdata()
        n_tumor = int(np.sum(data > 0))
        if n_tumor > 0:
            parts = name.rsplit("_", 1)
            if len(parts) == 2:
                patient_id, date = parts
            else:
                patient_id, date = name, ""
            scans.append({
                "subject": name,
                "patient_id": patient_id,
                "date": date,
                "tumor_mask_path": mask_file,
                "tumor_voxels": n_tumor,
            })
    return scans


def find_orig_nu(fastsurfer_dir: str, subject: str) -> str | None:
    """Find orig_nu.mgz for a subject."""
    for cohort in ["prevalent", "incident", ""]:
        if cohort:
            candidate = Path(fastsurfer_dir) / cohort / subject / "mri" / "orig_nu.mgz"
        else:
            candidate = Path(fastsurfer_dir) / subject / "mri" / "orig_nu.mgz"
        if candidate.exists():
            return str(candidate)
    return None


def convert_mgz_to_nifti(mgz_path: str, nifti_path: str):
    """Convert MGZ to NIfTI using nibabel."""
    img = nib.load(mgz_path)
    nib.save(img, nifti_path)


def run_lit_inpainting(
    t1_nifti: str,
    tumor_mask_nifti: str,
    output_dir: str,
    dilate: int = 2,
) -> str | None:
    """Run LIT inpainting via containerized script or local install.

    Returns path to inpainted T1 NIfTI.
    """
    # Try local lit-inpainting command first
    cmd = [
        "lit-inpainting",
        "--input_image", t1_nifti,
        "--lesion_mask", tumor_mask_nifti,
        "--output_directory", output_dir,
        "--dilate", str(dilate),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"\n  LIT stderr: {result.stderr[-500:]}")
            return None
    except FileNotFoundError:
        # Try containerized version
        lit_script = Path(__file__).resolve().parent.parent / "LIT" / "run_lit_containerized.sh"
        if lit_script.exists():
            cmd = [
                str(lit_script),
                "--input_image", t1_nifti,
                "--lesion_mask", tumor_mask_nifti,
                "--output_directory", output_dir,
                "--dilate", str(dilate),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        else:
            raise FileNotFoundError(
                "Neither 'lit-inpainting' command nor LIT/run_lit_containerized.sh found. "
                "Install neurolit (pip install neurolit) or clone https://github.com/Deep-MI/LIT"
            )

    # Find inpainted result
    result = os.path.join(output_dir, "inpainting_volumes", "inpainting_result.nii.gz")
    if os.path.exists(result):
        return result
    # Fallback: search for it
    for f in glob.glob(os.path.join(output_dir, "**", "inpainting_result.nii.gz"), recursive=True):
        return f
    return None


def run_fastsurfer_seg(
    t1_path: str,
    subject_id: str,
    output_dir: str,
    fastsurfer_bin: str,
    device: str = "cuda",
) -> str | None:
    """Run FastSurfer --seg_only --no_biasfield on a T1 image."""
    cmd = [
        fastsurfer_bin,
        "--t1", t1_path,
        "--sid", subject_id,
        "--sd", output_dir,
        "--seg_only",
        "--no_biasfield",
        "--vox_size", "1",
        "--no_hypothal",
        "--no_cc",
        "--device", device,
        "--viewagg_device", device,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n  FastSurfer stderr: {result.stderr[-500:]}")
        return None

    seg_path = os.path.join(output_dir, subject_id, "mri", "aparc.DKTatlas+aseg.deep.mgz")
    if not os.path.exists(seg_path):
        print(f"\n  FastSurfer output missing: {seg_path}")
        # Print the deep-seg log to see why
        log_path = os.path.join(output_dir, subject_id, "scripts", "deep-seg.log")
        if os.path.exists(log_path):
            with open(log_path) as f:
                lines = f.readlines()
            print(f"  deep-seg.log (last 20 lines):")
            for line in lines[-20:]:
                print(f"    {line.rstrip()}")
    return seg_path if os.path.exists(seg_path) else None


def register_seg_to_mni(
    seg_mgz_path: str,
    mask_mgz_path: str,
    orig_nu_path: str,
    template_path: str,
) -> tuple[np.ndarray, list[str]]:
    """ANTs affine register inpainted FastSurfer seg to MNI space.

    Uses same parameters as preprocess_fastsurfer.py.
    Returns (warped_seg_array, fwd_transforms).
    """
    # Load and skull-strip the inpainted orig_nu
    orig = ants.image_read(orig_nu_path)
    mask = ants.image_read(mask_mgz_path)
    brain = orig * ants.threshold_image(mask, 1, mask.max())
    template = ants.image_read(template_path)

    # Affine registration (same params as preprocess_fastsurfer.py)
    result = ants.registration(
        fixed=template,
        moving=brain,
        type_of_transform="Affine",
        aff_metric="mattes",
        aff_iterations=(2100, 1200, 1200, 10),
        aff_shrink_factors=(6, 4, 2, 1),
        aff_smoothing_sigmas=(3, 2, 1, 0),
    )

    # Warp segmentation
    seg_ants = ants.image_read(seg_mgz_path)
    seg_warped = ants.apply_transforms(
        fixed=template,
        moving=seg_ants,
        transformlist=result["fwdtransforms"],
        interpolator="nearestNeighbor",
    )

    seg_arr = seg_warped.numpy().astype(np.int32)

    # Merge cortical labels (same as preprocess_fastsurfer.py)
    for label in LH_CORTICAL:
        seg_arr[seg_arr == label] = 3
    for label in RH_CORTICAL:
        seg_arr[seg_arr == label] = 42

    # Clean up transforms
    for tf in result.get("fwdtransforms", []):
        if os.path.exists(tf):
            os.remove(tf)
    for tf in result.get("invtransforms", []):
        if os.path.exists(tf):
            os.remove(tf)

    return seg_arr


def center_crop(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Center-crop array to target shape."""
    starts = [(s - t) // 2 for s, t in zip(arr.shape, target_shape)]
    slices = tuple(slice(st, st + t) for st, t in zip(starts, target_shape))
    return arr[slices]


def update_seg_in_npz(npz_path: str, new_seg: np.ndarray):
    """Replace 'seg' key in .npz, keeping all other keys."""
    data = dict(np.load(npz_path))
    data["seg"] = new_seg.astype(np.int32)
    np.savez(npz_path, **data)


def process_scan(
    scan: dict,
    fastsurfer_dir: str,
    fastsurfer_bin: str,
    template_path: str,
    npz_dir: str,
    lit_output_dir: str,
    crop_shape: tuple,
) -> str:
    """Process one scan through LIT + FastSurfer + re-registration."""
    subject = scan["subject"]

    # Find orig_nu.mgz
    orig_nu = find_orig_nu(fastsurfer_dir, subject)
    if orig_nu is None:
        return "no_fastsurfer"

    # Find mask.mgz (in same dir as orig_nu)
    mask_mgz = os.path.join(os.path.dirname(orig_nu), "mask.mgz")
    if not os.path.exists(mask_mgz):
        return "no_mask"

    npz_path = os.path.join(npz_dir, f"{subject}.npz")
    if not os.path.exists(npz_path):
        return "no_npz"

    tmpdir = tempfile.mkdtemp(prefix=f"lit_{subject}_")
    try:
        # Convert mgz to nifti for LIT
        t1_nifti = os.path.join(tmpdir, "orig_nu.nii.gz")
        convert_mgz_to_nifti(orig_nu, t1_nifti)

        # Use full tumor mask (all labels > 0, including edema)
        # Rationale: Pollak et al. 2025 recommends oversegmentation over undersegmentation
        # and including "all abnormal tissue" in the mask
        tumor_mask = scan["tumor_mask_path"]

        # 6a: LIT inpainting
        lit_out = os.path.join(lit_output_dir, subject)
        os.makedirs(lit_out, exist_ok=True)
        inpainted = run_lit_inpainting(t1_nifti, tumor_mask, lit_out)
        if inpainted is None:
            return "lit_failed"

        # 6b: Re-run FastSurfer on inpainted T1
        fs_out = os.path.join(tmpdir, "fastsurfer")
        seg_path = run_fastsurfer_seg(
            os.path.abspath(inpainted), f"{subject}_lit", fs_out,
            os.path.abspath(fastsurfer_bin)
        )
        if seg_path is None:
            return "fastsurfer_failed"

        # Get mask.mgz from new FastSurfer output
        new_mask = os.path.join(fs_out, f"{subject}_lit", "mri", "mask.mgz")
        new_orig = os.path.join(fs_out, f"{subject}_lit", "mri", "orig_nu.mgz")

        # 6c: Re-register to MNI and update .npz
        seg_mni = register_seg_to_mni(
            seg_path,
            new_mask if os.path.exists(new_mask) else mask_mgz,
            new_orig if os.path.exists(new_orig) else orig_nu,
            template_path,
        )

        # Center-crop
        seg_cropped = center_crop(seg_mni, crop_shape)

        # Update .npz
        update_seg_in_npz(npz_path, seg_cropped)

        return "ok"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"error: {e}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="FastSurfer-LIT: lesion inpainting + re-segmentation"
    )
    parser.add_argument("--tumor-masks-dir", required=True,
                        help="Dir with conformed-space tumor masks (*_tumor.nii.gz)")
    parser.add_argument("--fastsurfer-dir", required=True,
                        help="Root of fastsurfer_output/")
    parser.add_argument("--fastsurfer-bin", default="FastSurfer/run_fastsurfer.sh",
                        help="Path to FastSurfer run script")
    parser.add_argument("--template", required=True,
                        help="MNI template NIfTI")
    parser.add_argument("--npz-dir", required=True,
                        help="Dir with .npz files to update")
    parser.add_argument("--lit-output-dir", default="lit_output",
                        help="Dir for LIT inpainting outputs")
    parser.add_argument("--crop-shape", nargs=3, type=int, default=[160, 192, 224])
    parser.add_argument("--priority-list", default=None,
                        help="Text file with subject names to process (one per line). "
                             "If provided, only these subjects are processed.")
    parser.add_argument("--power-log", default="fastsurfer_lit_power.csv",
                        help="Power monitoring CSV output")
    args = parser.parse_args()

    crop_shape = tuple(args.crop_shape)

    # Find scans with tumors
    print(f"Scanning {args.tumor_masks_dir} for non-empty tumor masks...")
    scans = find_scans_with_tumors(args.tumor_masks_dir)
    print(f"Found {len(scans)} scans with tumors")

    # Filter by priority list if provided
    if args.priority_list:
        with open(args.priority_list) as f:
            priority = set(line.strip() for line in f if line.strip())
        scans = [s for s in scans if s["subject"] in priority]
        print(f"Filtered to {len(scans)} scans from priority list")

    if not scans:
        print("Nothing to process.")
        return

    # Power monitoring
    monitor = PowerMonitor(filepath=args.power_log, interval=1.0, mode="batch")
    monitor.start()

    # Process
    stats = {}
    try:
        for i, scan in enumerate(scans, 1):
            monitor.update_scan(i, scan["subject"])

            # Resume: skip if LIT output already exists for this scan
            lit_done = os.path.join(args.lit_output_dir, scan["subject"],
                                    "inpainting_volumes", "inpainting_result.nii.gz")
            npz_path = os.path.join(args.npz_dir, f"{scan['subject']}.npz")
            if os.path.exists(lit_done) and os.path.exists(npz_path):
                # Check if seg was already updated (file mtime of npz > lit)
                if os.path.getmtime(npz_path) > os.path.getmtime(lit_done):
                    stats["skipped"] = stats.get("skipped", 0) + 1
                    continue

            print(f"[{i}/{len(scans)}] {scan['subject']} "
                  f"({scan['tumor_voxels']} tumor voxels)...", end=" ", flush=True)

            status = process_scan(
                scan, args.fastsurfer_dir, args.fastsurfer_bin,
                args.template, args.npz_dir, args.lit_output_dir, crop_shape,
            )

            status_key = status.split(":")[0]
            stats[status_key] = stats.get(status_key, 0) + 1
            print(status)

    except KeyboardInterrupt:
        print("\n[Interrupted] Data saved.")
    finally:
        monitor.stop()
        s = monitor.summary()
        monitor.close()

    # Summary
    print(f"\n{'=' * 50}")
    print("FASTSURFER-LIT SUMMARY")
    print(f"{'=' * 50}")
    for k, v in sorted(stats.items()):
        print(f"  {k:>20s}: {v}")
    print(f"  {'total':>20s}: {sum(stats.values())}")
    if s.get("total_energy_wh") is not None:
        print(f"\n  Energy: {s['total_energy_wh']} Wh over {s['duration_min']:.1f} min")
        print(f"  GPU: {s['gpu']['avg']}W avg / {s['gpu']['max']}W peak")
        print(f"  CPU: {s['cpu']['avg']}W avg")


if __name__ == "__main__":
    main()
