#!/usr/bin/env python3
"""Pad the MNI-ICBM152 2009c T1 template to TransMorph-compatible dimensions.

The raw MNI T1 is (193, 229, 193) at 1mm iso. Skull-strips (subjects are
skull-stripped, so the template must be too), normalizes to [0,1], and pads to
(224, 256, 224) — every dim divisible by 32 for TransMorph (4-level, full-res).

Output:
  - Atlas/mni_icbm152_t1_padded.nii.gz       — padded MNI T1 (224, 256, 224)

Usage:
    python phase2_preprocess/prepare_mni_template.py
"""

import os
import numpy as np
import nibabel as nib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# 1. Load MNI T1 template
# ---------------------------------------------------------------------------
mni_t1_path = os.path.join(
    PROJECT_ROOT,
    "Atlas/mni_icbm152_nlin_sym_09c_nifti/"
    "mni_icbm152_nlin_sym_09c/"
    "mni_icbm152_t1_tal_nlin_sym_09c.nii",
)
mni_mask_path = os.path.join(
    PROJECT_ROOT,
    "Atlas/mni_icbm152_nlin_sym_09c_nifti/"
    "mni_icbm152_nlin_sym_09c/"
    "mni_icbm152_t1_tal_nlin_sym_09c_mask.nii",
)
mni_nii = nib.load(mni_t1_path)
mni_vol = np.squeeze(mni_nii.get_fdata().astype(np.float32))
print(f"MNI T1 shape: {mni_vol.shape}")

# Skull-strip: subjects are skull-stripped, so the template must be too
mni_mask = np.squeeze(nib.load(mni_mask_path).get_fdata())
mni_vol = mni_vol * (mni_mask > 0).astype(np.float32)
print(f"Applied brain mask: {np.sum(mni_mask > 0)} brain voxels")

assert mni_vol.shape == (193, 229, 193), f"Unexpected MNI shape: {mni_vol.shape}"

# ---------------------------------------------------------------------------
# 2. Normalize MNI T1 to [0, 1]
# ---------------------------------------------------------------------------
vmin, vmax = mni_vol.min(), mni_vol.max()
if vmax - vmin > 0:
    mni_vol = (mni_vol - vmin) / (vmax - vmin)
print(f"MNI vol range: [{mni_vol.min():.4f}, {mni_vol.max():.4f}]")

# ---------------------------------------------------------------------------
# 3. Pad (193, 229, 193) -> (224, 256, 224)
#    Each dim = window_size * 32 for TransMorph (4-level, full-res) compatibility:
#      - dim divisible by 4 (PatchEmbed stride)
#      - dim/4, dim/8, dim/16 all even (3 PatchMerging operations)
#      - dim/4, dim/8, dim/16, dim/32 all divisible by window_size (zero-waste attention)
#    Axis 0: 193->224 = 7*32  (15 before + 16 after = 31)
#    Axis 1: 229->256 = 8*32  (13 before + 14 after = 27)
#    Axis 2: 193->224 = 7*32  (15 before + 16 after = 31)
#    window_size = (7, 8, 7)
# ---------------------------------------------------------------------------
PAD = ((15, 16), (13, 14), (15, 16))

mni_padded = np.pad(mni_vol, PAD, mode='constant', constant_values=0)

print(f"Padded MNI T1 shape: {mni_padded.shape}")
assert mni_padded.shape == (224, 256, 224), f"Bad shape: {mni_padded.shape}"

# ---------------------------------------------------------------------------
# 4. Update affine to account for padding shift
#    Shift origin by -N voxels on each axis (N = before-padding per axis)
# ---------------------------------------------------------------------------
affine = mni_nii.affine.copy()
pad_before = np.array([15, 13, 15], dtype=np.float64)
affine[:3, 3] -= affine[:3, :3] @ pad_before

# ---------------------------------------------------------------------------
# 5. Save output
# ---------------------------------------------------------------------------
out_vol_path = os.path.join(PROJECT_ROOT, "Atlas/mni_icbm152_t1_padded.nii.gz")

nib.save(nib.Nifti1Image(mni_padded, affine), out_vol_path)

print(f"\nSaved: {out_vol_path}")

# ---------------------------------------------------------------------------
# 6. Verification
# ---------------------------------------------------------------------------
print("\n=== Verification ===")
print(f"Padded T1 shape: {mni_padded.shape}")
print(f"Padded T1 range: [{mni_padded.min():.4f}, {mni_padded.max():.4f}]")

# Check brain not clipped: verify nonzero voxels don't touch the volume edges
for ax in range(3):
    slices_first = [slice(None)] * 3
    slices_last = [slice(None)] * 3
    slices_first[ax] = 0
    slices_last[ax] = -1
    first_sum = mni_padded[tuple(slices_first)].sum()
    last_sum = mni_padded[tuple(slices_last)].sum()
    print(f"Axis {ax}: first slice sum={first_sum:.2f}, last slice sum={last_sum:.2f}")

# TransMorph compatibility check (4-level, full-res, patch_size=4, 3 PatchMerging)
after_patch = tuple(s // 4 for s in mni_padded.shape)
ws = (7, 8, 7)
print(f"After PatchEmbed (÷4): {after_patch}")
dims = list(after_patch)
for level in range(4):
    divs = tuple(d % w for d, w in zip(dims, ws))
    even = tuple(d % 2 for d in dims)
    ok_ws = all(d == 0 for d in divs)
    ok_even = all(d == 0 for d in even) if level < 3 else True
    print(f"  Level {level}: {tuple(dims)} %ws={divs} {'✓' if ok_ws else '✗'}  %2={even} {'✓' if ok_even else '✗'}")
    if level < 3:
        dims = [d // 2 for d in dims]
print(f"Full-res divisible by 32: {all(s % 32 == 0 for s in mni_padded.shape)}")

print("\nDone.")
