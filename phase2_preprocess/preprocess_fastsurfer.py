#!/usr/bin/env python3
"""Affine preprocessing for FastSurfer data → VoxelMorph-ready .npz files.

Affine-registers all FastSurfer volumes to a template (IXI atlas) using ANTs,
then saves ready-to-train .npz files with 'vol' and 'seg' keys.

This fixes two critical preprocessing issues:
  1. No affine pre-registration — brains sit in arbitrary positions in 256³ space
  2. Zoom instead of crop — scipy.ndimage.zoom distorts brain geometry

Usage:
    conda activate transmorph
    python preprocessing/preprocess_fastsurfer.py \\
        --data-dir fastsurfer_output/phase2 \\
        --template Atlas/mni_icbm152_t1_padded.nii.gz \\
        --output-dir Voxelmorph/data/fastsurfer_preprocessed_mni \\
        --workers 4
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback

import ants
import nibabel as nib
import numpy as np

SEG_FILENAME = 'aparc.DKTatlas+aseg.deep.mgz'


def collect_subjects(data_dir, required_shape=(256, 256, 256)):
    """Find orig_nu.mgz + mask.mgz + seg in fastsurfer_output, filter by shape.

    Returns list of (orig_path, mask_path, seg_path) tuples.
    Only includes subjects where all three files exist.
    """
    # Support both flat (phase2/{SID}/mri/) and nested (cohort/{SID}/mri/) layouts
    all_files = sorted(
        glob.glob(os.path.join(data_dir, '*/mri/orig_nu.mgz'))
        + glob.glob(os.path.join(data_dir, '*/*/mri/orig_nu.mgz'))
    )
    print(f"Found {len(all_files)} total orig_nu.mgz files in {data_dir}")

    valid = []
    skipped_shape = 0
    skipped_missing = 0
    for f in all_files:
        try:
            hdr = nib.load(f).header
            shape = tuple(hdr.get_data_shape()[:3])
            if shape != required_shape:
                skipped_shape += 1
                continue

            mri_dir = os.path.dirname(f)
            mask_path = os.path.join(mri_dir, 'mask.mgz')
            seg_path = os.path.join(mri_dir, SEG_FILENAME)

            if not os.path.exists(mask_path) or not os.path.exists(seg_path):
                skipped_missing += 1
                continue

            valid.append((f, mask_path, seg_path))
        except Exception as e:
            print(f"  Skipping {f}: {e}")
            skipped_shape += 1

    print(f"Kept {len(valid)} subjects (skipped {skipped_shape} wrong shape, "
          f"{skipped_missing} missing mask/seg)")
    return valid


def subject_name_from_path(orig_path):
    """Extract subject session name from path.

    e.g. .../prevalent/YG_XXX_2015-09-29/mri/orig.mgz → YG_XXX_2015-09-29
    """
    return orig_path.split(os.sep)[-3]


def preprocess_one(args_tuple):
    """Process a single subject: affine register to template, save .npz.

    Args:
        args_tuple: (orig_path, mask_path, seg_path, template_path, output_dir)

    Returns:
        (orig_path, npz_path) on success, or (orig_path, None) on failure.
    """
    orig_path, mask_path, seg_path, template_path, output_dir = args_tuple

    subject = subject_name_from_path(orig_path)
    out_path = os.path.join(output_dir, f'{subject}.npz')

    # Skip if already processed
    if os.path.exists(out_path):
        return orig_path, out_path

    try:
        # Load template (each worker loads its own — ANTs images aren't picklable)
        template = ants.image_read(template_path)

        # Load orig and mask with ANTs
        orig = ants.image_read(orig_path)
        mask = ants.image_read(mask_path)

        # Skull-strip: multiply by binary mask
        brain = orig * ants.threshold_image(mask, 1, mask.max())

        # Affine register brain → template
        result = ants.registration(
            fixed=template,
            moving=brain,
            type_of_transform='Affine',
            aff_metric='mattes',
            aff_iterations=(2100, 1200, 1200, 10),
            aff_shrink_factors=(6, 4, 2, 1),
            aff_smoothing_sigmas=(3, 2, 1, 0),
        )

        # Registered brain volume
        vol = result['warpedmovout'].numpy()

        # Apply same transform to segmentation (nearest-neighbor)
        seg_ants = ants.image_read(seg_path)
        seg_warped = ants.apply_transforms(
            fixed=template,
            moving=seg_ants,
            transformlist=result['fwdtransforms'],
            interpolator='nearestNeighbor',
        )
        seg = seg_warped.numpy()

        # Merge cortical parcels → Left/Right-Cerebral-Cortex (TransMorph convention)
        seg[(seg >= 1002) & (seg <= 1035)] = 3
        seg[(seg >= 2002) & (seg <= 2035)] = 42

        # Normalize volume to [0, 1]
        vmin, vmax = vol.min(), vol.max()
        if vmax - vmin > 1e-8:
            vol = (vol - vmin) / (vmax - vmin)
        else:
            vol = np.zeros_like(vol)

        # Save
        vol = vol.astype(np.float32)
        seg = seg.astype(np.int32)
        np.savez(out_path, vol=vol, seg=seg)

        # Save ANTs transform files (needed for warping tumor masks later)
        transforms_dir = os.path.join(os.path.dirname(out_path), 'transforms')
        os.makedirs(transforms_dir, exist_ok=True)
        for tf in result.get('fwdtransforms', []):
            if os.path.exists(tf):
                dst = os.path.join(transforms_dir, f'{subject}_fwd.mat')
                shutil.copy2(tf, dst)
                os.remove(tf)
        for tf in result.get('invtransforms', []):
            if os.path.exists(tf):
                dst = os.path.join(transforms_dir, f'{subject}_inv.mat')
                shutil.copy2(tf, dst)
                os.remove(tf)

        return orig_path, out_path

    except Exception as e:
        print(f"  FAILED {subject}: {e}")
        traceback.print_exc()
        return orig_path, None


def remap_split(split_json_path, output_dir):
    """Map existing data_split.json (orig.mgz paths) → new .npz paths.

    Preserves the exact same train/val/test partition.
    """
    with open(split_json_path) as f:
        split = json.load(f)

    new_split = {}
    for key in ('train', 'val', 'test'):
        new_paths = []
        for orig_path in split[key]:
            subject = subject_name_from_path(orig_path)
            npz_path = os.path.join(output_dir, f'{subject}.npz')
            if os.path.exists(npz_path):
                new_paths.append(npz_path)
        new_split[key] = new_paths

    out_split_path = os.path.join(output_dir, 'data_split.json')
    with open(out_split_path, 'w') as f:
        json.dump(new_split, f, indent=2)

    total = sum(len(v) for v in new_split.values())
    print(f"\nData split saved to {out_split_path}")
    print(f"  train: {len(new_split['train'])}  |  "
          f"val: {len(new_split['val'])}  |  "
          f"test: {len(new_split['test'])}  |  total: {total}")
    return new_split


def main():
    parser = argparse.ArgumentParser(
        description='Affine preprocess FastSurfer data for VoxelMorph')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='FastSurfer output root (contains */*/mri/orig.mgz)')
    parser.add_argument('--template', type=str, required=True,
                        help='Path to template.nii.gz (e.g. IXI atlas)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Where to save .npz files')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel processes (default: 4)')
    parser.add_argument('--split-json', type=str, default=None,
                        help='Path to existing data_split.json to remap '
                             '(default: Voxelmorph/checkpoints/data_split.json)')
    args = parser.parse_args()

    # Resolve paths
    args.data_dir = os.path.expanduser(args.data_dir)
    args.template = os.path.expanduser(args.template)
    args.output_dir = os.path.expanduser(args.output_dir)

    # Validate template
    if not os.path.exists(args.template):
        print(f"ERROR: Template not found: {args.template}")
        sys.exit(1)

    # Show template info
    template = ants.image_read(args.template)
    print(f"Template: {args.template}")
    print(f"  shape: {template.shape}  spacing: {template.spacing}")

    # Collect subjects
    subjects = collect_subjects(args.data_dir)
    if not subjects:
        print("ERROR: No valid subjects found.")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check how many are already done
    already_done = sum(
        1 for s in subjects
        if os.path.exists(
            os.path.join(args.output_dir, f'{subject_name_from_path(s[0])}.npz'))
    )
    remaining = len(subjects) - already_done
    print(f"\nTotal subjects: {len(subjects)}")
    print(f"Already processed: {already_done}")
    print(f"Remaining: {remaining}")

    # Build work list
    work = [
        (orig, mask, seg, args.template, args.output_dir)
        for orig, mask, seg in subjects
    ]

    # Process
    t0 = time.time()
    successes = 0
    failures = 0

    if args.workers <= 1:
        # Sequential
        for i, w in enumerate(work):
            orig_path, npz_path = preprocess_one(w)
            if npz_path:
                successes += 1
            else:
                failures += 1
            if (i + 1) % 10 == 0 or (i + 1) == len(work):
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(work)}] "
                      f"{successes} ok, {failures} failed, "
                      f"{elapsed:.0f}s elapsed")
    else:
        # Parallel
        with mp.Pool(args.workers) as pool:
            for i, (orig_path, npz_path) in enumerate(
                    pool.imap_unordered(preprocess_one, work)):
                if npz_path:
                    successes += 1
                else:
                    failures += 1
                if (i + 1) % 10 == 0 or (i + 1) == len(work):
                    elapsed = time.time() - t0
                    print(f"  [{i+1}/{len(work)}] "
                          f"{successes} ok, {failures} failed, "
                          f"{elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Successes: {successes}")
    print(f"  Failures:  {failures}")

    # Verify one output
    sample_files = glob.glob(os.path.join(args.output_dir, '*.npz'))
    if sample_files:
        d = np.load(sample_files[0])
        print(f"\nSample output: {os.path.basename(sample_files[0])}")
        print(f"  vol: shape={d['vol'].shape}, "
              f"range=[{d['vol'].min():.3f}, {d['vol'].max():.3f}]")
        print(f"  seg: shape={d['seg'].shape}, "
              f"{len(np.unique(d['seg']))} unique labels")

    # Remap data split
    if args.split_json is None:
        # Default location
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.split_json = os.path.join(
            base_dir, 'Voxelmorph', 'checkpoints', 'data_split.json')

    if os.path.exists(args.split_json):
        print(f"\nRemapping split from: {args.split_json}")
        remap_split(args.split_json, args.output_dir)
    else:
        print(f"\nNo split file found at {args.split_json}, skipping remap")


if __name__ == '__main__':
    main()
