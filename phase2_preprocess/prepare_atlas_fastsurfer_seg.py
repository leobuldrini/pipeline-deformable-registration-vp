#!/usr/bin/env python3
"""Generate a FreeSurfer-based atlas segmentation for the MNI template.

Replaces the CerebrA-based atlas_seg with a FastSurfer segmentation that
includes all 30 TransMorph evaluation labels (WM, CSF, Choroid Plexus, etc.),
which CerebrA does not provide.

Pipeline:
  1. Run FastSurfer --seg_only on the MNI-ICBM152 T1 template
  2. ANTs affine register the conformed output back to MNI template space
     (same parameters as preprocess_fastsurfer.py for subjects)
  3. Merge DKT cortical parcels (1002-1035 -> 3, 2002-2035 -> 42)
  4. Pad (193, 229, 193) -> (224, 256, 224) (same as prepare_mni_template.py)
  5. Center-crop to (160, 192, 224)
  6. Save as atlas_seg NIfTI

Why not CerebrA:
  CerebrA (Manera et al. 2020) segments 102 cortical+subcortical regions but
  does NOT label White Matter, CSF, or Choroid Plexus. TransMorph (Chen et al.
  2022) evaluates on 30 structures that include these labels. Using CerebrA
  causes 5/30 evaluation labels to be permanently absent (Dice=0).

Usage:
    python preprocessing/prepare_atlas_fastsurfer_seg.py

    # If FastSurfer was already run on the MNI template:
    python preprocessing/prepare_atlas_fastsurfer_seg.py --skip-fastsurfer

    # Custom output paths:
    python preprocessing/prepare_atlas_fastsurfer_seg.py \
        --output-seg Atlas/fastsurfer_seg_160x192x224.nii.gz \
        --output-seg-padded Atlas/fastsurfer_seg_padded.nii.gz
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import ants
import nibabel as nib
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths to raw MNI-ICBM152 2009c template files
MNI_T1_PATH = os.path.join(
    PROJECT_ROOT,
    "Atlas/mni_icbm152_nlin_sym_09c_nifti/"
    "mni_icbm152_nlin_sym_09c/"
    "mni_icbm152_t1_tal_nlin_sym_09c.nii",
)
MNI_MASK_PATH = os.path.join(
    PROJECT_ROOT,
    "Atlas/mni_icbm152_nlin_sym_09c_nifti/"
    "mni_icbm152_nlin_sym_09c/"
    "mni_icbm152_t1_tal_nlin_sym_09c_mask.nii",
)

# Padded MNI template (target for ANTs registration)
MNI_PADDED_PATH = os.path.join(PROJECT_ROOT, "Atlas/mni_icbm152_t1_padded.nii.gz")

# FastSurfer binary
FASTSURFER_BIN = os.path.join(PROJECT_ROOT, "FastSurfer/run_fastsurfer.sh")

# Padding from (193, 229, 193) -> (224, 256, 224)
# Must match prepare_mni_template.py exactly
PAD = ((15, 16), (13, 14), (15, 16))

# Target shape after center-crop (TransMorph input)
TARGET_SHAPE = (160, 192, 224)

# DKT cortical parcel merge (same as preprocess_fastsurfer.py)
LH_CORTICAL = list(range(1002, 1036))
RH_CORTICAL = list(range(2002, 2036))

# TransMorph 30 evaluation labels (Chen et al. 2022, IXI dataset)
TRANSMORPH_30 = [
    16,  # Brain-Stem
    10,  # Left-Thalamus
    49,  # Right-Thalamus
    8,   # Left-Cerebellum-Cortex
    47,  # Right-Cerebellum-Cortex
    2,   # Left-Cerebral-White-Matter
    41,  # Right-Cerebral-White-Matter
    7,   # Left-Cerebellum-White-Matter
    46,  # Right-Cerebellum-White-Matter
    12,  # Left-Putamen
    51,  # Right-Putamen
    28,  # Left-VentralDC
    60,  # Right-VentralDC
    13,  # Left-Pallidum
    52,  # Right-Pallidum
    11,  # Left-Caudate
    50,  # Right-Caudate
    4,   # Left-Lateral-Ventricle
    43,  # Right-Lateral-Ventricle
    17,  # Left-Hippocampus
    53,  # Right-Hippocampus
    14,  # 3rd-Ventricle
    15,  # 4th-Ventricle
    18,  # Left-Amygdala
    54,  # Right-Amygdala
    3,   # Left-Cerebral-Cortex
    42,  # Right-Cerebral-Cortex
    24,  # CSF
    31,  # Left-choroid-plexus
    63,  # Right-choroid-plexus
]


def center_crop(arr, target_shape):
    """Center-crop a 3D array."""
    starts = tuple((s - t) // 2 for s, t in zip(arr.shape, target_shape))
    slices = tuple(slice(s, s + t) for s, t in zip(starts, target_shape))
    return arr[slices]


def run_fastsurfer(t1_path, output_dir, subject_id="mni_template"):
    """Run FastSurfer --seg_only on the MNI template."""
    cmd = [
        FASTSURFER_BIN,
        "--t1", os.path.abspath(t1_path),
        "--sid", subject_id,
        "--sd", os.path.abspath(output_dir),
        "--seg_only",
        "--no_biasfield",
        "--vox_size", "1",
        "--no_hypothal",
        "--no_cc",
    ]

    print(f"Running FastSurfer on MNI template...")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  stdout: {result.stdout[-1000:]}")
        print(f"  stderr: {result.stderr[-1000:]}")
        raise RuntimeError("FastSurfer failed")

    # Expected outputs
    seg_path = os.path.join(output_dir, subject_id, "mri", "aparc.DKTatlas+aseg.deep.mgz")
    orig_path = os.path.join(output_dir, subject_id, "mri", "orig.mgz")
    mask_path = os.path.join(output_dir, subject_id, "mri", "mask.mgz")

    for p in [seg_path, orig_path, mask_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected FastSurfer output missing: {p}")

    return seg_path, orig_path, mask_path


def register_to_mni(seg_path, orig_path, mask_path, template_path):
    """ANTs affine register FastSurfer seg back to MNI template space.

    Uses the same parameters as preprocess_fastsurfer.py so the atlas seg
    is in the exact same space as subject segmentations.
    """
    print("ANTs affine registration to MNI template...")
    template = ants.image_read(template_path)
    orig = ants.image_read(orig_path)
    mask = ants.image_read(mask_path)

    # Skull-strip (same as preprocess_fastsurfer.py)
    brain = orig * ants.threshold_image(mask, 1, mask.max())

    # Affine registration (identical parameters to preprocess_fastsurfer.py)
    result = ants.registration(
        fixed=template,
        moving=brain,
        type_of_transform="Affine",
        aff_metric="mattes",
        aff_iterations=(2100, 1200, 1200, 10),
        aff_shrink_factors=(6, 4, 2, 1),
        aff_smoothing_sigmas=(3, 2, 1, 0),
    )

    # Warp segmentation with nearest-neighbor
    seg_ants = ants.image_read(seg_path)
    seg_warped = ants.apply_transforms(
        fixed=template,
        moving=seg_ants,
        transformlist=result["fwdtransforms"],
        interpolator="nearestNeighbor",
    )

    seg_arr = seg_warped.numpy().astype(np.int32)

    # Clean up ANTs temp files
    for tf in result.get("fwdtransforms", []):
        if os.path.exists(tf):
            os.remove(tf)
    for tf in result.get("invtransforms", []):
        if os.path.exists(tf):
            os.remove(tf)

    return seg_arr


def merge_cortical(seg):
    """Merge DKT cortical parcels into L/R Cerebral Cortex.

    Same operation as preprocess_fastsurfer.py and run_fastsurfer_lit.py.
    Standard in VoxelMorph/TransMorph evaluation — volumetric registration
    cannot reliably align fine cortical parcels (requires surface registration).
    """
    seg = seg.copy()
    for label in LH_CORTICAL:
        seg[seg == label] = 3   # Left-Cerebral-Cortex
    for label in RH_CORTICAL:
        seg[seg == label] = 42  # Right-Cerebral-Cortex
    return seg


def verify(seg, name):
    """Print label statistics and check TransMorph-30 coverage."""
    labels = sorted(set(np.unique(seg).astype(int)) - {0})
    print(f"\n=== Verification: {name} ===")
    print(f"  Shape: {seg.shape}")
    print(f"  Labels ({len(labels)}): {labels}")
    print(f"  Nonzero voxels: {(seg > 0).sum()}")

    present = [l for l in TRANSMORPH_30 if l in labels]
    missing = [l for l in TRANSMORPH_30 if l not in labels]
    print(f"  TransMorph-30 coverage: {len(present)}/30")
    if missing:
        label_names = {
            2: "L-WM", 3: "L-Cortex", 4: "L-Lat-Vent", 5: "L-Inf-Lat-Vent",
            7: "L-Cereb-WM", 8: "L-Cereb-Ctx", 10: "L-Thalamus", 11: "L-Caudate",
            12: "L-Putamen", 13: "L-Pallidum", 14: "3rd-Vent", 15: "4th-Vent",
            16: "Brain-Stem", 17: "L-Hippocampus", 18: "L-Amygdala", 24: "CSF",
            26: "L-Accumbens", 28: "L-VentralDC", 31: "L-Choroid",
            41: "R-WM", 42: "R-Cortex", 43: "R-Lat-Vent", 44: "R-Inf-Lat-Vent",
            46: "R-Cereb-WM", 47: "R-Cereb-Ctx", 49: "R-Thalamus", 50: "R-Caudate",
            51: "R-Putamen", 52: "R-Pallidum", 53: "R-Hippocampus", 54: "R-Amygdala",
            58: "R-Accumbens", 60: "R-VentralDC", 63: "R-Choroid", 85: "Optic-Chiasm",
        }
        for l in missing:
            print(f"    MISSING: {l} ({label_names.get(l, '?')})")
    else:
        print(f"  All 30 TransMorph labels present!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate FastSurfer-based atlas segmentation for MNI template"
    )
    parser.add_argument("--skip-fastsurfer", action="store_true",
                        help="Skip FastSurfer if already run (reuse existing output)")
    parser.add_argument("--fastsurfer-output", type=str,
                        default=os.path.join(PROJECT_ROOT, "fastsurfer_atlas"),
                        help="FastSurfer output directory (default: fastsurfer_atlas/)")
    parser.add_argument("--subject-id", type=str, default="mni_template",
                        help="FastSurfer subject ID (default: mni_template)")
    parser.add_argument("--output-seg", type=str,
                        default=os.path.join(PROJECT_ROOT,
                                             "Atlas/fastsurfer_seg_160x192x224.nii.gz"),
                        help="Output path for cropped atlas seg (160x192x224)")
    parser.add_argument("--output-seg-padded", type=str,
                        default=os.path.join(PROJECT_ROOT,
                                             "Atlas/fastsurfer_seg_padded.nii.gz"),
                        help="Output path for padded atlas seg (224x256x224)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1: Run FastSurfer on MNI template
    # ------------------------------------------------------------------
    fs_dir = os.path.abspath(args.fastsurfer_output)
    sid = args.subject_id

    seg_mgz = os.path.join(fs_dir, sid, "mri", "aparc.DKTatlas+aseg.deep.mgz")
    orig_mgz = os.path.join(fs_dir, sid, "mri", "orig.mgz")
    mask_mgz = os.path.join(fs_dir, sid, "mri", "mask.mgz")

    if args.skip_fastsurfer:
        for p in [seg_mgz, orig_mgz, mask_mgz]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"--skip-fastsurfer but missing: {p}\n"
                    f"Run without --skip-fastsurfer first.")
        print(f"Reusing FastSurfer output from {fs_dir}/{sid}/")
    else:
        # Convert MNI template to NIfTI in a temp dir (FastSurfer expects NIfTI or MGZ)
        seg_mgz, orig_mgz, mask_mgz = run_fastsurfer(
            MNI_T1_PATH, fs_dir, sid
        )
    print(f"  seg: {seg_mgz}")
    print(f"  orig: {orig_mgz}")
    print(f"  mask: {mask_mgz}")

    # ------------------------------------------------------------------
    # Step 2: ANTs affine register back to MNI padded template
    # ------------------------------------------------------------------
    seg_mni = register_to_mni(seg_mgz, orig_mgz, mask_mgz, MNI_PADDED_PATH)
    print(f"  Registered seg shape: {seg_mni.shape}")  # should be (224, 256, 224)

    # ------------------------------------------------------------------
    # Step 3: Merge cortical parcels
    # ------------------------------------------------------------------
    seg_merged = merge_cortical(seg_mni)
    verify(seg_merged, "after merge (224x256x224)")

    # ------------------------------------------------------------------
    # Step 4: Save padded version (224, 256, 224)
    # ------------------------------------------------------------------
    # Use same affine as the padded MNI template
    padded_nii = nib.load(MNI_PADDED_PATH)
    affine = padded_nii.affine

    os.makedirs(os.path.dirname(args.output_seg_padded), exist_ok=True)
    nib.save(
        nib.Nifti1Image(seg_merged.astype(np.int32), affine),
        args.output_seg_padded,
    )
    print(f"\nSaved padded seg: {args.output_seg_padded}")

    # ------------------------------------------------------------------
    # Step 5: Center-crop to (160, 192, 224)
    # ------------------------------------------------------------------
    seg_cropped = center_crop(seg_merged, TARGET_SHAPE)
    verify(seg_cropped, f"after crop {TARGET_SHAPE}")

    # Compute cropped affine (shift origin by crop offset)
    starts = tuple((s - t) // 2 for s, t in zip(seg_merged.shape, TARGET_SHAPE))
    crop_affine = affine.copy()
    crop_affine[:3, 3] += affine[:3, :3] @ np.array(starts, dtype=np.float64)

    os.makedirs(os.path.dirname(args.output_seg), exist_ok=True)
    nib.save(
        nib.Nifti1Image(seg_cropped.astype(np.int32), crop_affine),
        args.output_seg,
    )
    print(f"Saved cropped seg: {args.output_seg}")

    # ------------------------------------------------------------------
    # Usage instructions
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("DONE. To use the new atlas segmentation:")
    print(f"{'='*60}")
    print(f"  --atlas-seg {args.output_seg}")
    print()
    print("Example:")
    print(f"  python run_transmorph.py --train --amp \\")
    print(f"      --data-dir ../Voxelmorph/data/yale_phase2_mni_160 \\")
    print(f"      --atlas ../Atlas/mni_icbm152_t1_padded_160x192x224.nii.gz \\")
    print(f"      --atlas-seg ../{os.path.relpath(args.output_seg, PROJECT_ROOT)} \\")
    print(f"      --config TransMorph --model transmorph \\")
    print(f"      --masked-ncc --vol-pres-weight 0.1 \\")
    print(f"      --save-dir checkpoints_improved")


if __name__ == "__main__":
    main()
