#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Center-crop .npz volumes and NIfTI atlas to a target shape.

Following the VoxelMorph (Balakrishnan et al., 2019) and TransMorph
(Chen et al., 2022) preprocessing protocol, volumes are CROPPED (not
resized) to the target shape.  This preserves the original 1 mm
isotropic voxel spacing and introduces zero interpolation artifacts.

Usage:
    python crop_to_transmorph.py \
        --src-dir ../Voxelmorph/data/fastsurfer_preprocessed_mni \
        --dst-dir ../Voxelmorph/data/fastsurfer_preprocessed_mni_160 \
        --target-shape 160 192 224 \
        --atlas ../Atlas/mni_icbm152_t1_padded.nii.gz \
        --atlas-seg ../Atlas/fastsurfer_seg_160x192x224.nii.gz \
        --workers 8
"""

import os
import sys
import glob
import shutil
import argparse
from multiprocessing import Pool
from functools import partial

import numpy as np
import nibabel as nib
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────────────────────────────

def center_crop(arr, target_shape):
    """Center-crop a 3-D array to *target_shape*.

    Returns the cropped array and the (start0, start1, start2) offsets.
    """
    starts = tuple((s - t) // 2 for s, t in zip(arr.shape, target_shape))
    slices = tuple(slice(s, s + t) for s, t in zip(starts, target_shape))
    return arr[slices], starts


def crop_npz(src_path, dst_path, target_shape):
    """Load an .npz, center-crop vol & seg, save to *dst_path*."""
    data = np.load(src_path)
    vol = data['vol']
    seg = data['seg']

    vol_c, _ = center_crop(vol, target_shape)
    seg_c, _ = center_crop(seg, target_shape)

    assert vol_c.shape == tuple(target_shape), \
        f"Bad vol shape {vol_c.shape} (expected {target_shape})"
    assert seg_c.shape == tuple(target_shape), \
        f"Bad seg shape {seg_c.shape} (expected {target_shape})"

    out = dict(vol=vol_c.astype(np.float32), seg=seg_c.astype(np.int32))

    # Propagate tumor_mask if present
    if 'tumor_mask' in data:
        tm_c, _ = center_crop(data['tumor_mask'], target_shape)
        out['tumor_mask'] = tm_c.astype(np.uint8)

    np.savez(dst_path, **out)


def crop_nifti(src_path, dst_path, target_shape):
    """Load a NIfTI, center-crop, adjust affine, save.

    Preserves the on-disk dtype to avoid nibabel float64->int32
    rounding errors that shift integer labels.
    """
    nii = nib.load(src_path)
    orig_dtype = nii.header.get_data_dtype()
    data = np.asarray(nii.dataobj, dtype=orig_dtype)
    affine = nii.affine.copy()

    cropped, starts = center_crop(data, target_shape)

    # Shift origin to account for removed voxels
    affine[:3, 3] += affine[:3, :3] @ np.array(starts, dtype=np.float64)

    out = nib.Nifti1Image(cropped, affine)
    nib.save(out, dst_path)

    # Verify roundtrip integrity
    reloaded = np.asarray(nib.load(dst_path).dataobj, dtype=orig_dtype)
    assert np.array_equal(cropped, reloaded), \
        f"NIfTI roundtrip failed for {dst_path}: " \
        f"{np.sum(cropped != reloaded)} voxels differ"

    return starts


# ──────────────────────────────────────────────────────────────────────
# Worker wrapper (for multiprocessing)
# ──────────────────────────────────────────────────────────────────────

def _worker(src_path, dst_dir, target_shape):
    """Process a single .npz file (called by Pool)."""
    basename = os.path.basename(src_path)
    dst_path = os.path.join(dst_dir, basename)
    try:
        crop_npz(src_path, dst_path, target_shape)
        return basename, True, None
    except Exception as e:
        return basename, False, str(e)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Center-crop .npz volumes (and atlas) to a target shape.')
    parser.add_argument('--src-dir', required=True,
                        help='Source directory with .npz files')
    parser.add_argument('--dst-dir', required=True,
                        help='Destination directory for cropped files')
    parser.add_argument('--target-shape', type=int, nargs=3,
                        default=[160, 192, 224],
                        help='Target shape (default: 160 192 224)')
    parser.add_argument('--atlas', default=None,
                        help='Atlas .nii.gz to crop')
    parser.add_argument('--atlas-seg', default=None,
                        help='Atlas segmentation .nii.gz to crop')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--dry-run', action='store_true',
                        help='Process 5 files, print diagnostics, do not save')
    args = parser.parse_args()

    target = tuple(args.target_shape)
    src_dir = os.path.abspath(args.src_dir)
    dst_dir = os.path.abspath(args.dst_dir)

    npz_files = sorted(glob.glob(os.path.join(src_dir, '*.npz')))
    if not npz_files:
        print(f"No .npz files found in {src_dir}")
        sys.exit(1)

    # Verify source shape
    sample = np.load(npz_files[0])
    src_shape = sample['vol'].shape
    print(f"Source shape: {src_shape}")
    print(f"Target shape: {target}")
    for i in range(3):
        if src_shape[i] < target[i]:
            print(f"ERROR: source dim {i} ({src_shape[i]}) < target ({target[i]})")
            sys.exit(1)
    crop_starts = tuple((s - t) // 2 for s, t in zip(src_shape, target))
    crop_ends = tuple(s + t for s, t in zip(crop_starts, target))
    print(f"Crop offsets: start={crop_starts}, end={crop_ends}")
    print(f"Found {len(npz_files)} .npz files\n")

    # ── Dry run ──
    if args.dry_run:
        print("=== DRY RUN (5 files) ===\n")
        for f in npz_files[:5]:
            d = np.load(f)
            vol, seg = d['vol'], d['seg']
            vc, _ = center_crop(vol, target)
            sc, _ = center_crop(seg, target)
            orig_labels = set(np.unique(seg)) - {0}
            crop_labels = set(np.unique(sc)) - {0}
            lost = orig_labels - crop_labels
            brain_pct = 100 * np.sum(vc > 0.01) / max(np.sum(vol > 0.01), 1)
            print(f"  {os.path.basename(f)}:")
            print(f"    shape: {vol.shape} -> {vc.shape}")
            print(f"    vol range: [{vc.min():.4f}, {vc.max():.4f}]")
            print(f"    labels: {len(orig_labels)} -> {len(crop_labels)}"
                  f"  lost: {lost if lost else 'none'}")
            print(f"    brain voxels retained: {brain_pct:.1f}%")
        print("\nDry run complete. Remove --dry-run to process all files.")
        return

    # ── Create output dir ──
    os.makedirs(dst_dir, exist_ok=True)

    # ── Crop atlas ──
    if args.atlas:
        atlas_path = os.path.abspath(args.atlas)
        name = os.path.splitext(os.path.splitext(os.path.basename(atlas_path))[0])[0]
        dst_atlas = os.path.join(os.path.dirname(atlas_path),
                                 f"{name}_{target[0]}x{target[1]}x{target[2]}.nii.gz")
        print(f"Cropping atlas: {atlas_path}")
        starts = crop_nifti(atlas_path, dst_atlas, target)
        print(f"  -> {dst_atlas}  (offset: {starts})")

    if args.atlas_seg:
        seg_path = os.path.abspath(args.atlas_seg)
        name = os.path.splitext(os.path.splitext(os.path.basename(seg_path))[0])[0]
        dst_seg = os.path.join(os.path.dirname(seg_path),
                               f"{name}_{target[0]}x{target[1]}x{target[2]}.nii.gz")
        print(f"Cropping atlas seg: {seg_path}")
        starts = crop_nifti(seg_path, dst_seg, target)
        print(f"  -> {dst_seg}  (offset: {starts})")

    # ── Crop .npz files ──
    print(f"\nCropping {len(npz_files)} volumes with {args.workers} workers...")
    worker_fn = partial(_worker, dst_dir=dst_dir, target_shape=target)

    failures = []
    with Pool(args.workers) as pool:
        results = pool.imap_unordered(worker_fn, npz_files)
        for name, ok, err in tqdm(results, total=len(npz_files)):
            if not ok:
                failures.append((name, err))

    # ── Copy data_split.json ──
    split_src = os.path.join(src_dir, 'data_split.json')
    if os.path.exists(split_src):
        shutil.copy2(split_src, os.path.join(dst_dir, 'data_split.json'))
        print("Copied data_split.json")

    # ── Summary ──
    print(f"\nDone: {len(npz_files) - len(failures)} / {len(npz_files)} succeeded")
    if failures:
        print(f"Failures ({len(failures)}):")
        for name, err in failures:
            print(f"  {name}: {err}")


if __name__ == '__main__':
    main()
