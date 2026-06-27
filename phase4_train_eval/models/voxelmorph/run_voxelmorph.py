# -*- coding: utf-8 -*-
"""VoxelMorph training & inference script (PyTorch backend).

Adapted from Colab notebook:
    https://colab.research.google.com/drive/1PtOMdMCSEsJXWtOBDI2OhnWJlf_wIG34

Usage:
    conda activate transmorph

    # Train on FastSurfer data (skull-stripped via mask.mgz)
    python run_voxelmorph.py --train --data-dir fastsurfer_output/phase2

    # Resume training (auto-detects checkpoint.pt)
    python run_voxelmorph.py --train --data-dir fastsurfer_output/phase2

    # Evaluate on test set (Dice + Jacobian) — FastSurfer .mgz data
    python run_voxelmorph.py --eval --data-dir fastsurfer_output/phase2 --weights checkpoints/vxm_final.pt

    # Evaluate on .npz data (auto-detected)
    python run_voxelmorph.py --eval --data-dir Voxelmorph/data --weights checkpoints/vxm_final.pt

    # Inference with trained weights
    python run_voxelmorph.py --weights checkpoints/vxm_final.pt

    # Untrained demo with sample data
    python run_voxelmorph.py

Sections:
  Training  — scan-to-scan on FastSurfer orig.mgz volumes (train split)
  Eval      — Dice + % negative Jacobian on test split
  Tutorial  — sample data from VoxelMorph repo (inference only)
  Yale Data — registration on local Yale dataset (inference only)
"""

import os
import sys
import json
os.environ['VXM_BACKEND'] = 'pytorch'

import glob
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'common'))
from power_monitor import PowerMonitor
from labels import EVAL_LABELS_CEREBRA, EVAL_LABELS_30
from tqdm import tqdm, trange
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import zoom, correlate
import matplotlib.pyplot as plt
import torch
import voxelmorph as vxm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'common'))
from losses import (BendingEnergy, AntifoldLoss, MaskedNCC,
                     DiceLoss, VolumePreservationLoss, regularize_loss_3d,
                     eval_stsr, eval_tvcf, tvcf_pair_passes_filter,
                     smooth_tumor_mask,
                     distribution_summary, format_distribution)
from pairs import extract_patient_id, generate_pairs
from argparse import ArgumentParser

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATASET_DIR = os.path.join(BASE_DIR, 'Dataset (Teste)')
DATA_DIR = os.path.join(BASE_DIR, 'Voxelmorph/data')

# 95 non-background FreeSurfer DKTatlas+aseg labels
DKTATLAS_LABELS = [
    2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 31,
    41, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63, 77,
    1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013,
    1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024,
    1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035,
    2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
    2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024,
    2025, 2026, 2027, 2028, 2029, 2030, 2031, 2034, 2035,
]

# 33 subcortical FreeSurfer labels (DKTatlas labels < 1000)
SUBCORTICAL_LABELS = [l for l in DKTATLAS_LABELS if l < 1000]

SEG_FILENAME = 'aparc.DKTatlas+aseg.deep.mgz'

# FreeSurfer aseg labels present in tutorial .npz files (non-background)
ASEG_LABELS = [
    2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28,
    30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
    63, 77, 85, 251, 252, 253, 254, 255,
]

# ============================================================
# Helper functions
# ============================================================

def _seg_to_onehot(seg_np, labels):
    """Convert integer label map to one-hot [1, K, D, H, W] tensor.

    Used for differentiable Dice loss per TransMorph Sec. 3.2 / VoxelMorph Sec. 3.
    """
    K = len(labels)
    D, H, W = seg_np.shape
    onehot = np.zeros((1, K, D, H, W), dtype=np.float32)
    for i, lbl in enumerate(labels):
        onehot[0, i] = (seg_np == lbl).astype(np.float32)
    return torch.from_numpy(onehot)


def adjust_learning_rate(optimizer, epoch, max_epochs, init_lr, power=0.9):
    """Polynomial LR decay (from original VoxelMorph/TransMorph training)."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - epoch / max_epochs, power), 8)


def is_isotropic(spacing, tolerance=1e-3):
    """Verifica se o spacing e isotropico"""
    max_diff = max(
        abs(spacing[0] - spacing[1]),
        abs(spacing[1] - spacing[2]),
        abs(spacing[0] - spacing[2])
    )
    return max_diff < tolerance


def resample_to_isotropic(image_sitk, target_spacing=(1.0, 1.0, 1.0)):
    """Resampling usando B-spline cubico"""
    original_spacing = image_sitk.GetSpacing()
    original_size = image_sitk.GetSize()

    new_size = [
        int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]

    print(f"  Original: spacing={original_spacing}, size={original_size}")
    print(f"  Target:   spacing={target_spacing}, size={new_size}")

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image_sitk.GetDirection())
    resampler.SetOutputOrigin(image_sitk.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkBSpline)

    return resampler.Execute(image_sitk)


def normalize_intensity(volume):
    """Normalizacao [0, 1]"""
    return (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)


def resize_to_fixed_shape(volume, target_shape=(160, 192, 144)):
    """Resize para shape fixo usando scipy zoom"""
    if volume.shape == target_shape:
        return volume

    zoom_factors = [target_shape[i] / volume.shape[i] for i in range(3)]
    resized = zoom(volume, zoom_factors, order=3)
    return resized


def load_and_preprocess_yale(filepath,
                              target_spacing=(1.0, 1.0, 1.0),
                              target_shape=None):
    """Carrega e preprocessa volume Yale"""
    print(f"\nCarregando: {filepath}")

    image_sitk = sitk.ReadImage(filepath)
    original_spacing = image_sitk.GetSpacing()
    print(f"Original spacing: {original_spacing}")

    if is_isotropic(original_spacing):
        print("Ja e isotropico, pulando resampling")
        volume = sitk.GetArrayFromImage(image_sitk)
    else:
        print("Aplicando resampling isotropico (B-spline)...")
        resampled = resample_to_isotropic(image_sitk, target_spacing)
        volume = sitk.GetArrayFromImage(resampled)
        image_sitk = resampled

    if target_shape is not None:
        print(f"Resizing para shape fixo {target_shape}...")
        volume = resize_to_fixed_shape(volume, target_shape)
    else:
        volume = sitk.GetArrayFromImage(image_sitk)

    print("Normalizando intensidades [0, 1]...")
    volume = normalize_intensity(volume)

    nib_img = nib.load(filepath)
    affine = nib_img.affine
    header = nib_img.header

    print(f"Shape final: {volume.shape}")
    print(f"Value range: [{volume.min():.3f}, {volume.max():.3f}]")

    return volume, affine, header


def run_inference(model, moving_vol, fixed_vol, device):
    """Run VxmDense forward pass and return moved volume + warp field."""
    moving_t = torch.from_numpy(moving_vol).float().unsqueeze(0).unsqueeze(0).to(device)
    fixed_t = torch.from_numpy(fixed_vol).float().unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        moved_t, warp_t = model(moving_t, fixed_t)

    moved = moved_t.squeeze().cpu().numpy()
    warp = warp_t.squeeze().cpu().numpy()
    return moved, warp


def plot_3x3(fixed, moving, moved, vol_shape, offset=0):
    """3x3 grid: Sag/Cor/Ax for Fixed, Moving, Moved"""
    slices = []
    for vol in [fixed, moving, moved]:
        mid = [np.take(vol, vol_shape[d]//2 - offset, axis=d) for d in range(3)]
        mid[1] = np.rot90(mid[1], 1)
        mid[2] = np.rot90(mid[2], -1)
        slices.extend(mid)

    titles = ['Fixed Sag', 'Fixed Cor', 'Fixed Ax',
              'Moving Sag', 'Moving Cor', 'Moving Ax',
              'Moved Sag', 'Moved Cor', 'Moved Ax']

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, (sl, title) in enumerate(zip(slices, titles)):
        ax = axes[i // 3, i % 3]
        ax.imshow(sl, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_diff(fixed, moving, moved, vol_shape):
    """Axial difference maps"""
    idx = vol_shape[2] // 2
    data = [moving[:, :, idx], fixed[:, :, idx], moved[:, :, idx],
            np.abs(fixed[:, :, idx] - moving[:, :, idx]),
            np.abs(fixed[:, :, idx] - moved[:, :, idx])]
    cmaps = ['gray', 'gray', 'gray', 'hot', 'hot']
    titles = ['Moving', 'Fixed', 'Moved', 'Diff Before', 'Diff After']

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, d, cmap, title in zip(axes, data, cmaps, titles):
        im = ax.imshow(d, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


# ============================================================
# Data collection & splitting
# ============================================================

def collect_mgz_files(data_dir, required_shape=(256, 256, 256)):
    """Find orig.mgz + mask.mgz + seg in fastsurfer_output, filter by shape.

    Returns list of (orig_path, mask_path, seg_path) tuples.
    mask_path and seg_path can be None if files are missing.
    """
    pattern = os.path.join(data_dir, '*/*/mri/orig.mgz')
    all_files = sorted(glob.glob(pattern))
    print(f"Found {len(all_files)} total orig.mgz files in {data_dir}")

    valid = []
    skipped = 0
    for f in all_files:
        try:
            hdr = nib.load(f).header
            shape = tuple(hdr.get_data_shape()[:3])
            if shape == required_shape:
                mri_dir = os.path.dirname(f)
                mask_path = os.path.join(mri_dir, 'mask.mgz')
                seg_path = os.path.join(mri_dir, SEG_FILENAME)
                valid.append((
                    f,
                    mask_path if os.path.exists(mask_path) else None,
                    seg_path if os.path.exists(seg_path) else None,
                ))
            else:
                skipped += 1
        except Exception as e:
            print(f"  Skipping {f}: {e}")
            skipped += 1

    n_masked = sum(1 for _, m, _ in valid if m is not None)
    n_seg = sum(1 for _, _, s in valid if s is not None)
    print(f"Kept {len(valid)} volumes with shape {required_shape} "
          f"({n_masked} with mask, {n_seg} with seg, skipped {skipped})")
    return valid


def _subject_id(orig_path):
    """Extract subject ID from path. e.g. .../YG_XXX_2015-09-29/mri/orig.mgz -> YG_XXX"""
    session_dir = orig_path.split(os.sep)[-3]
    return session_dir.rsplit('_', 1)[0]


def get_or_create_split(vol_files, split_path, ratios=(0.8, 0.1, 0.1), seed=42):
    """Split by subject into train/val/test. Persists to JSON.

    Splitting by subject (not session) prevents data leakage when
    subjects have multiple sessions.
    """
    if os.path.exists(split_path):
        with open(split_path) as f:
            saved = json.load(f)
        orig_to_tuple = {t[0]: t for t in vol_files}
        split = {}
        for key in ('train', 'val', 'test'):
            split[key] = [orig_to_tuple[p] for p in saved[key]
                          if p in orig_to_tuple]
        print(f"Loaded split: {len(split['train'])} train / "
              f"{len(split['val'])} val / {len(split['test'])} test")
        return split

    # Group sessions by subject
    subj_files = defaultdict(list)
    for t in vol_files:
        subj_files[_subject_id(t[0])].append(t)

    # Shuffle subjects, then split
    subjects = sorted(subj_files.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train_subjs = subjects[:n_train]
    val_subjs = subjects[n_train:n_train + n_val]
    test_subjs = subjects[n_train + n_val:]

    split = {
        'train': [t for s in train_subjs for t in subj_files[s]],
        'val':   [t for s in val_subjs   for t in subj_files[s]],
        'test':  [t for s in test_subjs  for t in subj_files[s]],
    }

    # Persist orig paths
    saved = {k: [t[0] for t in v] for k, v in split.items()}
    with open(split_path, 'w') as f:
        json.dump(saved, f, indent=2)
    print(f"Created split: {len(split['train'])} train / "
          f"{len(split['val'])} val / {len(split['test'])} test "
          f"({len(train_subjs)}/{len(val_subjs)}/{len(test_subjs)} subjects)")
    print(f"  Saved to {split_path}")
    return split


# ============================================================
# Training helpers
# ============================================================

def load_and_preprocess_mgz(orig_path, mask_path, vol_shape):
    """Load orig.mgz, apply brain mask, resize, normalize to [0, 1]."""
    vol = nib.load(orig_path).get_fdata().astype(np.float32)
    if mask_path is not None:
        mask = nib.load(mask_path).get_fdata()
        vol = vol * (mask > 0)
    vol = resize_to_fixed_shape(vol, vol_shape)
    vol = normalize_intensity(vol)
    return vol


def collect_npz_files(data_dir):
    """Find .npz files containing 'vol' (and optionally 'seg') arrays.

    Returns list of npz paths. Each file is expected to have at least
    a 'vol' key.  Files with a 'seg' key are usable for evaluation.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    valid = []
    for p in paths:
        try:
            with np.load(p) as d:
                if 'vol' in d:
                    valid.append(p)
        except Exception as e:
            print(f"  Skipping {p}: {e}")
    n_seg = sum(1 for p in valid if 'seg' in np.load(p))
    print(f"Found {len(valid)} .npz volumes in {data_dir} ({n_seg} with seg)")
    return valid


def load_npz_vol(npz_path, vol_shape):
    """Load volume from .npz, resize and normalize to [0, 1]."""
    vol = np.load(npz_path)['vol'].astype(np.float32)
    vol = resize_to_fixed_shape(vol, vol_shape)
    vol = normalize_intensity(vol)
    return vol


def load_npz_seg(npz_path, vol_shape):
    """Load segmentation from .npz, resize with nearest-neighbor."""
    seg = np.load(npz_path)['seg'].astype(np.float32)
    if seg.shape != vol_shape:
        zoom_factors = [vol_shape[i] / seg.shape[i] for i in range(3)]
        seg = zoom(seg, zoom_factors, order=0)
    return seg


def detect_labels(npz_files, max_scan=10):
    """Auto-detect non-background segmentation labels from .npz files."""
    all_labels = set()
    for p in npz_files[:max_scan]:
        with np.load(p) as d:
            if 'seg' in d:
                all_labels.update(np.unique(d['seg']).tolist())
    all_labels.discard(0)
    return sorted(all_labels)


class PairDataset(torch.utils.data.IterableDataset):
    """Infinite random-pair dataset for DataLoader with multiple workers."""

    def __init__(self, vol_files, vol_shape):
        self.vol_files = vol_files
        self.vol_shape = vol_shape

    def __iter__(self):
        n = len(self.vol_files)
        while True:
            i, j = np.random.choice(n, size=2, replace=False)
            orig_i, mask_i, _ = self.vol_files[i]
            orig_j, mask_j, _ = self.vol_files[j]
            moving = load_and_preprocess_mgz(orig_i, mask_i, self.vol_shape)
            fixed = load_and_preprocess_mgz(orig_j, mask_j, self.vol_shape)
            # shape: [1, 1, D, H, W] (batch=1, channel=1)
            yield (torch.from_numpy(moving)[None, None],
                   torch.from_numpy(fixed)[None, None])


class NpzPairDataset(torch.utils.data.IterableDataset):
    """Infinite random-pair dataset loading from preprocessed .npz files."""

    def __init__(self, npz_paths, vol_shape):
        self.npz_paths = npz_paths
        self.vol_shape = vol_shape

    def __iter__(self):
        n = len(self.npz_paths)
        while True:
            i, j = np.random.choice(n, size=2, replace=False)
            moving = load_npz_vol(self.npz_paths[i], self.vol_shape)
            fixed = load_npz_vol(self.npz_paths[j], self.vol_shape)
            yield (torch.from_numpy(moving)[None, None],
                   torch.from_numpy(fixed)[None, None])


class NpzAtlasDataset(torch.utils.data.IterableDataset):
    """Infinite atlas-based dataset for training.

    When reverse=False (atlas-to-scan): yields (atlas, subject, ...) as (moving, fixed).
    When reverse=True (scan-to-atlas): yields (subject, atlas, ...) as (moving, fixed).
    Tumor mask always comes from the subject.
    """

    def __init__(self, npz_paths, atlas_vol, vol_shape,
                 load_tumor_mask=False, load_seg=False, atlas_seg=None,
                 reverse=False):
        self.npz_paths = npz_paths
        self.atlas = atlas_vol
        self.vol_shape = vol_shape
        self.load_tumor_mask = load_tumor_mask
        self.load_seg = load_seg
        self.atlas_seg = atlas_seg
        self.reverse = reverse

    def __iter__(self):
        n = len(self.npz_paths)
        while True:
            i = np.random.randint(n)
            d = np.load(self.npz_paths[i])
            subject = d['vol'].astype(np.float32)
            if subject.max() > 1:
                subject = subject / subject.max()
            atlas = self.atlas.copy()

            if self.reverse:
                result = [torch.from_numpy(subject)[None, None],
                          torch.from_numpy(atlas)[None, None]]
            else:
                result = [torch.from_numpy(atlas)[None, None],
                          torch.from_numpy(subject)[None, None]]

            if self.load_tumor_mask:
                if 'tumor_mask' in d:
                    tm = d['tumor_mask'].astype(np.float32)
                else:
                    tm = np.zeros_like(subject)
                result.append(torch.from_numpy(tm)[None, None])

            if self.load_seg:
                seg = d['seg'].astype(np.float32)
                result.append(torch.from_numpy(seg)[None, None])
                if self.atlas_seg is not None:
                    a_seg = self.atlas_seg.astype(np.float32)
                    result.append(torch.from_numpy(a_seg)[None, None])

            yield tuple(result)


class NpzScanPairDataset(torch.utils.data.Dataset):
    """Map-style scan-to-scan training dataset.

    Returns (mov, fix, [tm_mov, tm_fix], [seg_mov, seg_fix]).
    Tumor masks: BOTH moving and fixed (symmetric Brett 2001 cost-function
    masking, required for longitudinal mets where fixed-side tumor is in a
    different location than moving-side).
    Organ mask comes from the moving image (Dong et al. ICCV 2023).

    When mode='scan-to-scan-intra', the fixed scan is sampled from other scans
    of the SAME patient (extracted via patient_id_fn from the npz basename).
    Patients with < 2 scans are dropped from the index entirely (no silent
    inter-fallback). When mode='scan-to-scan-inter', the fixed scan is sampled
    from scans of DIFFERENT patients.

    Yield shape per element is [1, D, H, W]; DataLoader(batch_size=1) adds
    the batch dim to produce [1, 1, D, H, W] (matches today's IterableDataset
    runtime shape exactly).
    """

    def __init__(self, npz_paths, vol_shape,
                 load_tumor_mask=False, load_seg=False,
                 mode='scan-to-scan-intra', patient_id_fn=None):
        self.vol_shape = vol_shape
        self.load_tumor_mask = load_tumor_mask
        self.load_seg = load_seg
        self.mode = mode
        self.patient_id_fn = patient_id_fn

        self._patient_of = {}
        self._by_patient = defaultdict(list)
        self.npz_paths = list(npz_paths)
        if patient_id_fn is not None:
            for i, p in enumerate(self.npz_paths):
                pid = patient_id_fn(os.path.basename(p))
                self._patient_of[i] = pid
                self._by_patient[pid].append(i)

        if (mode == 'scan-to-scan-intra'
                and patient_id_fn is not None):
            keep_pids = {pid for pid, idxs in self._by_patient.items()
                         if len(idxs) >= 2}
            n_dropped = sum(1 for pid in self._by_patient
                            if pid not in keep_pids)
            if n_dropped > 0:
                print(f"NpzScanPairDataset: dropped {n_dropped} "
                      f"single-scan patients from intra index")
            kept_paths = [self.npz_paths[i] for i, p in enumerate(self.npz_paths)
                          if self._patient_of[i] in keep_pids]
            self.npz_paths = kept_paths
            self._patient_of = {}
            self._by_patient = defaultdict(list)
            for i, p in enumerate(self.npz_paths):
                pid = patient_id_fn(os.path.basename(p))
                self._patient_of[i] = pid
                self._by_patient[pid].append(i)
            if len(self.npz_paths) == 0:
                raise RuntimeError(
                    'scan-to-scan-intra requires at least one patient '
                    'with >= 2 scans in the training split.')

    def _pick_fixed_index(self, i):
        n = len(self.npz_paths)
        if self.mode == 'scan-to-scan-intra' and self.patient_id_fn is not None:
            pid = self._patient_of[i]
            candidates = [j for j in self._by_patient[pid] if j != i]
            return int(np.random.choice(candidates))
        if self.mode == 'scan-to-scan-inter' and self.patient_id_fn is not None:
            pid = self._patient_of[i]
            candidates = [other for other in self._by_patient.keys()
                          if other != pid]
            chosen_pid = candidates[int(np.random.randint(len(candidates)))]
            return int(np.random.choice(self._by_patient[chosen_pid]))
        j = i
        while j == i:
            j = np.random.randint(n)
        return j

    def __getitem__(self, index):
        d_mov = np.load(self.npz_paths[index])
        j = self._pick_fixed_index(index)
        d_fix = np.load(self.npz_paths[j])

        mov = d_mov['vol'].astype(np.float32)
        fix = d_fix['vol'].astype(np.float32)
        if mov.max() > 1:
            mov = mov / mov.max()
        if fix.max() > 1:
            fix = fix / fix.max()

        # Yield each element with shape [1, D, H, W]; DataLoader adds batch dim.
        result = [torch.from_numpy(mov)[None],
                  torch.from_numpy(fix)[None]]

        if self.load_tumor_mask:
            tm_mov = d_mov['tumor_mask'].astype(np.float32)
            tm_fix = d_fix['tumor_mask'].astype(np.float32)
            result.append(torch.from_numpy(tm_mov)[None])
            result.append(torch.from_numpy(tm_fix)[None])

        if self.load_seg:
            mov_seg = d_mov['seg'].astype(np.float32)
            fix_seg = d_fix['seg'].astype(np.float32)
            result.append(torch.from_numpy(mov_seg)[None])
            result.append(torch.from_numpy(fix_seg)[None])

        return tuple(result)

    def __len__(self):
        return len(self.npz_paths)


class NpzValDataset(torch.utils.data.Dataset):
    """Validation dataset returning (vol, seg) for multi-worker DataLoader."""

    def __init__(self, npz_paths, vol_shape):
        self.vol_shape = vol_shape
        self.paths = []
        for p in npz_paths:
            try:
                with np.load(p) as d:
                    if 'seg' in d:
                        self.paths.append(p)
            except Exception:
                pass

    def __getitem__(self, index):
        d = np.load(self.paths[index])
        vol = d['vol'].astype(np.float32)
        seg = d['seg'].astype(np.float32)
        tm = d['tumor_mask'].astype(np.float32) if 'tumor_mask' in d else np.zeros_like(vol)
        vol = resize_to_fixed_shape(vol, self.vol_shape)
        vol = normalize_intensity(vol)
        return (torch.from_numpy(vol).float().unsqueeze(0),
                torch.from_numpy(seg).float().unsqueeze(0),
                torch.from_numpy(tm).float().unsqueeze(0))

    def __len__(self):
        return len(self.paths)


# ============================================================
# Evaluation helpers
# ============================================================

def load_seg(seg_path, vol_shape):
    """Load segmentation and resize with nearest-neighbor (preserves labels)."""
    seg = nib.load(seg_path).get_fdata().astype(np.float32)
    if seg.shape != vol_shape:
        zoom_factors = [vol_shape[i] / seg.shape[i] for i in range(3)]
        seg = zoom(seg, zoom_factors, order=0)  # nearest-neighbor
    return seg


def compute_jacobian_det(disp):
    """Jacobian determinant of a 3D displacement field.

    Args:
        disp: numpy array of shape (3, D, H, W).
    Returns:
        jacdet: numpy array (D-4, H-4, W-4) — border-trimmed.
    """
    gradx = np.array([-0.5, 0, 0.5]).reshape(3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1)
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3)

    gradx_disp = np.stack([correlate(disp[i], gradx, mode='constant', cval=0.0)
                           for i in range(3)], axis=0)
    grady_disp = np.stack([correlate(disp[i], grady, mode='constant', cval=0.0)
                           for i in range(3)], axis=0)
    gradz_disp = np.stack([correlate(disp[i], gradz, mode='constant', cval=0.0)
                           for i in range(3)], axis=0)

    grad_disp = np.stack([gradx_disp, grady_disp, gradz_disp], axis=0)
    jacobian = grad_disp + np.eye(3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]

    jacdet = (jacobian[0, 0] * (jacobian[1, 1] * jacobian[2, 2] -
                                 jacobian[1, 2] * jacobian[2, 1]) -
              jacobian[1, 0] * (jacobian[0, 1] * jacobian[2, 2] -
                                 jacobian[0, 2] * jacobian[2, 1]) +
              jacobian[2, 0] * (jacobian[0, 1] * jacobian[1, 2] -
                                 jacobian[0, 2] * jacobian[1, 1]))
    return jacdet


def jacobian_neg_pct(disp):
    """Percent of voxels with negative Jacobian determinant (folding)."""
    jac_det = compute_jacobian_det(disp)
    return 100.0 * np.sum(jac_det < 0) / jac_det.size


def evaluate(model, test_files, vol_shape, device, labels, num_pairs=100):
    """Post-training evaluation: Dice + % negative Jacobian on test pairs.

    For each pair: forward pass with registration=True (integrated displacement),
    warp moving seg with nearest-neighbor, compute Dice via vxm.py.utils.dice().

    Also computes baseline (unregistered) Dice and optional subcortical-only Dice.
    """
    files_with_seg = [(o, m, s) for o, m, s in test_files if s is not None]
    if len(files_with_seg) < 2:
        raise RuntimeError(
            f"Need >= 2 test subjects with segmentation, got {len(files_with_seg)}")

    model.eval()
    seg_warper = vxm.layers.SpatialTransformer(vol_shape, mode='nearest').to(device)

    all_dice = []
    all_baseline_dice = []
    all_neg_jac = []
    rng = np.random.RandomState(0)

    for _ in trange(num_pairs, desc='Evaluating'):
        i, j = rng.choice(len(files_with_seg), size=2, replace=False)
        orig_m, mask_m, seg_m_path = files_with_seg[i]
        orig_f, mask_f, seg_f_path = files_with_seg[j]

        moving_vol = load_and_preprocess_mgz(orig_m, mask_m, vol_shape)
        fixed_vol = load_and_preprocess_mgz(orig_f, mask_f, vol_shape)
        seg_moving = load_seg(seg_m_path, vol_shape)
        seg_fixed = load_seg(seg_f_path, vol_shape)

        # Baseline Dice (before registration)
        baseline_dice = vxm.py.utils.dice(
            seg_moving.astype(int), seg_fixed.astype(int), labels)
        all_baseline_dice.append(baseline_dice)

        # Model forward pass
        moving_t = torch.from_numpy(moving_vol).float()[None, None].to(device)
        fixed_t = torch.from_numpy(fixed_vol).float()[None, None].to(device)

        with torch.no_grad():
            _, pos_flow = model(moving_t, fixed_t, registration=True)

        seg_m_t = torch.from_numpy(seg_moving).float()[None, None].to(device)
        with torch.no_grad():
            warped_seg_t = seg_warper(seg_m_t, pos_flow)
        warped_seg = warped_seg_t.squeeze().cpu().numpy()

        # Dice via vxm built-in (all labels)
        dice_scores = vxm.py.utils.dice(
            warped_seg.astype(int), seg_fixed.astype(int), labels)
        all_dice.append(dice_scores)

        # Jacobian
        flow_np = pos_flow.squeeze().cpu().numpy()
        all_neg_jac.append(jacobian_neg_pct(flow_np))

    all_dice = np.array(all_dice)  # (num_pairs, num_labels)
    all_baseline_dice = np.array(all_baseline_dice)
    mean_dice_per_pair = np.nanmean(all_dice, axis=1)
    mean_baseline_per_pair = np.nanmean(all_baseline_dice, axis=1)

    result = {
        'dice_mean': float(np.nanmean(all_dice)),
        'dice_std': float(np.nanstd(mean_dice_per_pair)),
        'baseline_dice_mean': float(np.nanmean(all_baseline_dice)),
        'baseline_dice_std': float(np.nanstd(mean_baseline_per_pair)),
        'dice_per_label': {int(l): float(d)
                           for l, d in zip(labels, np.nanmean(all_dice, axis=0))},
        'neg_jac_pct_mean': float(np.mean(all_neg_jac)),
        'neg_jac_pct_std': float(np.std(all_neg_jac)),
        'num_pairs': num_pairs,
    }

    sub_labels = [l for l in SUBCORTICAL_LABELS if l in labels]
    if sub_labels:
        sub_idx = [list(labels).index(l) for l in sub_labels]
        sub_dice = all_dice[:, sub_idx]
        sub_baseline = all_baseline_dice[:, sub_idx]
        result['subcortical_dice_mean'] = float(np.nanmean(sub_dice))
        result['subcortical_dice_std'] = float(
            np.nanstd(np.nanmean(sub_dice, axis=1)))
        result['subcortical_baseline_dice_mean'] = float(
            np.nanmean(sub_baseline))
        result['subcortical_baseline_dice_std'] = float(
            np.nanstd(np.nanmean(sub_baseline, axis=1)))
        result['num_subcortical_labels'] = len(sub_labels)

    return result


def evaluate_npz(model, npz_files, vol_shape, device, labels, num_pairs=100,
                 atlas_vol=None, atlas_seg=None, mode=None):
    """Evaluation on .npz files.

    If atlas_vol and atlas_seg are provided, uses atlas-based evaluation
    with STSR (Dong et al. ICCV 2023). Otherwise scan-to-scan pairs.

    When mode is 'scan-to-scan-intra' or 'scan-to-scan-inter', s2s pairs
    are sampled patient-grouped via common.pairs.generate_pairs (matches
    training-time sampling regime). When mode is None, falls back to
    fully random pair selection (legacy behavior).
    """
    files_with_seg = [p for p in npz_files if 'seg' in np.load(p)]
    if len(files_with_seg) < (1 if atlas_vol is not None else 2):
        raise RuntimeError(
            f"Need .npz files with 'seg', got {len(files_with_seg)}")

    model.eval()
    seg_warper = vxm.layers.SpatialTransformer(vol_shape, mode='nearest').to(device)
    bi_warper = vxm.layers.SpatialTransformer(vol_shape).to(device)

    all_dice = []
    all_baseline_dice = []
    all_neg_jac = []
    all_stsr = []
    all_tvcf = []
    all_lvcr = []
    tvcf_eligible_pairs = 0  # pairs where both masks are non-empty

    if (atlas_vol is not None and atlas_seg is not None
            and mode not in ('scan-to-scan-intra', 'scan-to-scan-inter')):
        # --- Atlas-based evaluation with STSR ---
        atlas_t = torch.from_numpy(atlas_vol).float()[None, None].to(device)
        atlas_seg_t = torch.from_numpy(atlas_seg).float()[None, None].to(device)
        organ_t = (atlas_seg_t > 0).float()

        eval_count = min(num_pairs, len(files_with_seg))
        for idx in trange(eval_count, desc='Evaluating (atlas)'):
            d = np.load(files_with_seg[idx])
            subject = d['vol'].astype(np.float32)
            subject_seg = d['seg'].astype(np.float32)
            if subject.max() > 1:
                subject = subject / subject.max()

            baseline_dice = vxm.py.utils.dice(
                atlas_seg.astype(int), subject_seg.astype(int), labels)
            all_baseline_dice.append(baseline_dice)

            sub_t = torch.from_numpy(subject).float()[None, None].to(device)

            with torch.no_grad():
                _, pos_flow = model(atlas_t, sub_t, registration=True)
                warped_seg = seg_warper(atlas_seg_t, pos_flow)

            warped_np = warped_seg.squeeze().cpu().numpy()
            dice_scores = vxm.py.utils.dice(
                warped_np.astype(int), subject_seg.astype(int), labels)
            all_dice.append(dice_scores)

            flow_np = pos_flow.squeeze().cpu().numpy()
            all_neg_jac.append(jacobian_neg_pct(flow_np))

            # STSR (Dong et al. ICCV 2023)
            tumor_mask_np = d['tumor_mask'] if 'tumor_mask' in d else None
            if tumor_mask_np is not None and tumor_mask_np.sum() > 0:
                with torch.no_grad():
                    tm_t = (torch.from_numpy(tumor_mask_np.astype(np.float32))[None, None].to(device) > 0).float()
                    warped_tm = bi_warper(tm_t, pos_flow)
                    warped_org = bi_warper(organ_t, pos_flow)
                    stsr_val = eval_stsr(warped_tm, tm_t, warped_org, organ_t)
                    all_stsr.append(stsr_val)
    else:
        # --- Scan-to-scan evaluation ---
        bn_to_path = {os.path.basename(p): p for p in files_with_seg}
        if mode in ('scan-to-scan-intra', 'scan-to-scan-inter'):
            # Patient-grouped pairs (matches training sampling regime)
            pair_basenames = generate_pairs(
                mode, [os.path.basename(p) for p in files_with_seg],
                max_pairs=num_pairs)[:num_pairs]
            print(f"  s2s eval mode={mode}: {len(pair_basenames)} pairs "
                  f"from {len(files_with_seg)} test files")
            pair_indices = [(files_with_seg.index(bn_to_path[a]),
                             files_with_seg.index(bn_to_path[b]))
                            for a, b in pair_basenames]
        else:
            # Legacy fallback: random pairs
            rng = np.random.RandomState(0)
            pair_indices = []
            for _ in range(num_pairs):
                i, j = rng.choice(len(files_with_seg), size=2, replace=False)
                pair_indices.append((int(i), int(j)))

        for i, j in tqdm(pair_indices, desc='Evaluating'):
            moving_vol = load_npz_vol(files_with_seg[i], vol_shape)
            fixed_vol = load_npz_vol(files_with_seg[j], vol_shape)
            seg_moving = load_npz_seg(files_with_seg[i], vol_shape)
            seg_fixed = load_npz_seg(files_with_seg[j], vol_shape)

            baseline_dice = vxm.py.utils.dice(
                seg_moving.astype(int), seg_fixed.astype(int), labels)
            all_baseline_dice.append(baseline_dice)

            moving_t = torch.from_numpy(moving_vol).float()[None, None].to(device)
            fixed_t = torch.from_numpy(fixed_vol).float()[None, None].to(device)

            with torch.no_grad():
                _, pos_flow = model(moving_t, fixed_t, registration=True)

            seg_m_t = torch.from_numpy(seg_moving).float()[None, None].to(device)
            with torch.no_grad():
                warped_seg_t = seg_warper(seg_m_t, pos_flow)
            warped_seg = warped_seg_t.squeeze().cpu().numpy()

            dice_scores = vxm.py.utils.dice(
                warped_seg.astype(int), seg_fixed.astype(int), labels)
            all_dice.append(dice_scores)

            flow_np = pos_flow.squeeze().cpu().numpy()
            all_neg_jac.append(jacobian_neg_pct(flow_np))

            # STSR Dong: tumor + organ from moving (Dong et al. ICCV 2023)
            d_mov = np.load(files_with_seg[i])
            d_fix = np.load(files_with_seg[j])
            mov_tm_np = d_mov['tumor_mask'] if 'tumor_mask' in d_mov else None
            fix_tm_np = d_fix['tumor_mask'] if 'tumor_mask' in d_fix else None
            if mov_tm_np is not None and mov_tm_np.sum() > 0:
                with torch.no_grad():
                    tm_t = (torch.from_numpy(mov_tm_np.astype(np.float32))[None, None].to(device) > 0).float()
                    organ_t_mov = (torch.from_numpy(seg_moving)[None, None].to(device) > 0).float()
                    warped_tm = bi_warper(tm_t, pos_flow)
                    warped_org = bi_warper(organ_t_mov, pos_flow)
                    all_stsr.append(eval_stsr(warped_tm, tm_t, warped_org, organ_t_mov))

                    # TVCF: predicted-vs-GT longitudinal volume change
                    # (s2s-intra only, after topology-preservation filter)
                    if (fix_tm_np is not None and fix_tm_np.sum() > 0):
                        tvcf_eligible_pairs += 1
                        passes, _reason = tvcf_pair_passes_filter(
                            mov_tm_np, fix_tm_np)
                        if passes:
                            fix_tm_t = (torch.from_numpy(fix_tm_np.astype(np.float32))[None, None].to(device) > 0).float()
                            tvcf_v, lvcr_v = eval_tvcf(warped_tm, tm_t, fix_tm_t)
                            if not np.isnan(tvcf_v):
                                all_tvcf.append(tvcf_v)
                                all_lvcr.append(lvcr_v)

    all_dice = np.array(all_dice)
    all_baseline_dice = np.array(all_baseline_dice)
    mean_dice_per_pair = np.nanmean(all_dice, axis=1)
    mean_baseline_per_pair = np.nanmean(all_baseline_dice, axis=1)

    result = {
        'eval_mode': mode if mode is not None else ('atlas-to-scan' if atlas_vol is not None else 'scan-to-scan-random'),
        'dice_mean': float(np.nanmean(all_dice)),
        'dice_std': float(np.nanstd(mean_dice_per_pair)),
        'baseline_dice_mean': float(np.nanmean(all_baseline_dice)),
        'baseline_dice_std': float(np.nanstd(mean_baseline_per_pair)),
        'dice_per_label': {int(l): float(d)
                           for l, d in zip(labels, np.nanmean(all_dice, axis=0))},
        'neg_jac_pct_mean': float(np.mean(all_neg_jac)),
        'neg_jac_pct_std': float(np.std(all_neg_jac)),
        'num_pairs': len(all_dice),
        'stsr_mean': float(np.nanmean(all_stsr)) if all_stsr else None,
        'stsr_std': float(np.nanstd(all_stsr)) if all_stsr else None,
        'stsr_n': len(all_stsr),
        'tvcf_mean': float(np.nanmean(all_tvcf)) if all_tvcf else None,
        'tvcf_std': float(np.nanstd(all_tvcf)) if all_tvcf else None,
        'tvcf_n': len(all_tvcf),
        'tvcf_eligible': tvcf_eligible_pairs,
        'tvcf_retention': (
            len(all_tvcf) / tvcf_eligible_pairs
            if tvcf_eligible_pairs > 0 else None),
        'lvcr_mean': float(np.nanmean(all_lvcr)) if all_lvcr else None,
        'lvcr_std': float(np.nanstd(all_lvcr)) if all_lvcr else None,
        'dice_distribution':     distribution_summary(mean_dice_per_pair),
        'baseline_dice_distribution': distribution_summary(
            mean_baseline_per_pair),
        'stsr_distribution':     distribution_summary(all_stsr),
        'tvcf_distribution':     distribution_summary(all_tvcf),
        'lvcr_distribution':     distribution_summary(all_lvcr),
    }

    sub_labels = [l for l in SUBCORTICAL_LABELS if l in labels]
    if sub_labels:
        sub_idx = [list(labels).index(l) for l in sub_labels]
        sub_dice = all_dice[:, sub_idx]
        sub_baseline = all_baseline_dice[:, sub_idx]
        result['subcortical_dice_mean'] = float(np.nanmean(sub_dice))
        result['subcortical_dice_std'] = float(
            np.nanstd(np.nanmean(sub_dice, axis=1)))
        result['subcortical_baseline_dice_mean'] = float(
            np.nanmean(sub_baseline))
        result['subcortical_baseline_dice_std'] = float(
            np.nanstd(np.nanmean(sub_baseline, axis=1)))
        result['num_subcortical_labels'] = len(sub_labels)

    return result


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to trained .pt weights (VxmDense)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')

    # Training arguments
    parser.add_argument('--train', action='store_true',
                        help='Run training instead of inference')
    parser.add_argument('--eval', action='store_true',
                        help='Post-training evaluation (Dice + Jacobian) on test set')
    parser.add_argument('--data-dir', type=str,
                        default=os.path.expanduser('fastsurfer_output/phase2'),
                        help='Root dir: fastsurfer_output tree or folder of .npz files')
    parser.add_argument('--vol-shape', type=int, nargs=3,
                        default=[160, 192, 224],
                        help='Volume shape for model (D H W)')
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--steps-per-epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--reg-param', type=float, default=None,
                        help='Regularization weight (default: 0.02 for mse, 1.0 for ncc)')
    parser.add_argument('--loss', type=str, default='mse',
                        choices=['ncc', 'mse'],
                        help='Similarity loss function (default: mse)')
    parser.add_argument('--int-steps', type=int, default=7,
                        help='Integration steps (0=non-diffeomorphic, 7=default diffeo)')
    parser.add_argument('--int-downsize', type=int, default=2,
                        help='Flow field downsample factor for integration')
    parser.add_argument('--reg-type', type=str, default='diffusion',
                        choices=['diffusion', 'bending'],
                        help='Regularizer type (default: diffusion)')
    parser.add_argument('--antifold-weight', type=float, default=0.0,
                        help='Anti-folding loss weight (0=disabled, paper: 100.0, Mok & Chung 2020)')
    parser.add_argument('--vp', action='store_true',
                        help='Enable volume preservation pipeline (Dong et al. ICCV 2023): '
                             'soft-weighted Pearson similarity + VP loss + Dong regularization')
    parser.add_argument('--dice-weight', type=float, default=0.0,
                        help='Dice auxiliary loss weight (0=disabled, paper: 1.0, TransMorph Eq.16)')
    parser.add_argument('--vol-pres-weight', type=float, default=0.0,
                        help='Volume preservation loss weight (0=disabled, paper: 0.1, Dong et al. 2023)')
    parser.add_argument('--mask-smooth', type=str, default='gaussian',
                        choices=['none', 'gaussian'],
                        help='Tumor mask smoothing: none (binary), '
                             'gaussian (Brett 2001 principle, mets-tuned 4mm FWHM). Default: gaussian')
    parser.add_argument('--atlas-seg', type=str, default=None,
                        help='Atlas segmentation path for Dice loss')
    parser.add_argument('--save-dir', type=str,
                        default=os.path.join(BASE_DIR, 'Voxelmorph/checkpoints'),
                        help='Directory for saved checkpoints')
    parser.add_argument('--save-every', type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--eval-pairs', type=int, default=100,
                        help='Number of random pairs for evaluation')
    parser.add_argument('--atlas', type=str, default=None,
                        help='Atlas .nii.gz for atlas-to-scan training '
                             '(if omitted, trains scan-to-scan)')
    parser.add_argument('--mode', type=str, default='atlas-to-scan',
                        choices=['atlas-to-scan', 'scan-to-atlas',
                                 'scan-to-scan-intra', 'scan-to-scan-inter'],
                        help='Registration mode')
    parser.add_argument('--max-pairs', type=int, default=100,
                        help='Max pairs for scan-to-scan-inter eval (default: 100)')
    parser.add_argument('--pair-seed', type=int, default=42,
                        help='Random seed for scan-to-scan-inter pair selection')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    vol_shape = tuple(args.vol_shape)
    nb_features = [
        [16, 32, 32, 32],              # encoder
        [32, 32, 32, 32, 32, 16, 16]   # decoder
    ]

    if args.train:
        # ==========================================================
        # Training
        # ==========================================================
        print("\n" + "=" * 60)
        print("  VoxelMorph — Training")
        print("=" * 60)

        # --- Collect & split data ---
        os.makedirs(args.save_dir, exist_ok=True)

        # Detect data format: .npz files or FastSurfer .mgz tree
        npz_files = collect_npz_files(args.data_dir)
        use_npz = len(npz_files) >= 2

        if use_npz:
            # .npz mode — look for data_split.json in data dir
            split_path = os.path.join(args.data_dir, 'data_split.json')
            if os.path.exists(split_path):
                with open(split_path) as f:
                    saved = json.load(f)
                basename_to_path = {os.path.basename(p): p for p in npz_files}
                train_npz = [basename_to_path[os.path.basename(p)]
                             for p in saved['train']
                             if os.path.basename(p) in basename_to_path]
                val_npz = [basename_to_path[os.path.basename(p)]
                           for p in saved['val']
                           if os.path.basename(p) in basename_to_path]
                test_count = sum(1 for p in saved['test']
                                 if os.path.basename(p) in basename_to_path)
                print(f"Loaded .npz split: {len(train_npz)} train / "
                      f"{len(val_npz)} val / {test_count} test")
            else:
                train_npz = npz_files
                val_npz = []
                print(f"No split file found, using all {len(train_npz)} .npz for training")
            if len(train_npz) < 2:
                raise RuntimeError(f"Need at least 2 train volumes, found {len(train_npz)}")
        else:
            # .mgz mode
            vol_files = collect_mgz_files(args.data_dir)
            if len(vol_files) < 2:
                raise RuntimeError(f"Need at least 2 volumes, found {len(vol_files)}")
            split_path = os.path.join(args.save_dir, 'data_split.json')
            split = get_or_create_split(vol_files, split_path)
            train_files = split['train']

        # --- Build model ---
        model = vxm.networks.VxmDense(
            vol_shape, nb_features,
            int_steps=args.int_steps,
            int_downsize=args.int_downsize,
        )
        model.to(device)

        # --- Losses ---
        if args.reg_param is None:
            args.reg_param = 0.02 if args.loss == 'mse' else 1.0

        if args.vp:
            sim_loss_obj = MaskedNCC().to(device)
            sim_loss_fn = None  # use sim_loss_obj directly
        elif args.loss == 'ncc':
            sim_loss_fn = vxm.losses.NCC().loss
            sim_loss_obj = None
        else:
            sim_loss_fn = vxm.losses.MSE().loss
            sim_loss_obj = None

        if args.reg_type == 'bending':
            reg_loss_obj = BendingEnergy().to(device)
            use_dong_reg = False
        else:
            # Dong et al. ICCV 2023 regularize_loss_3d (diffusion, divisor /2)
            reg_loss_obj = None
            use_dong_reg = True

        criterion_antifold = AntifoldLoss().to(device) if args.antifold_weight > 0 else None
        criterion_dice = DiceLoss().to(device) if args.dice_weight > 0 else None
        criterion_vol_pres = VolumePreservationLoss().to(device) if args.vol_pres_weight > 0 else None

        use_tumor_mask = args.vp or args.vol_pres_weight > 0
        use_seg = args.dice_weight > 0 or args.vol_pres_weight > 0
        dice_seg_warper = vxm.layers.SpatialTransformer(vol_shape, mode='nearest').to(device) if args.dice_weight > 0 else None
        vp_brain_warper = vxm.layers.SpatialTransformer(vol_shape).to(device) if args.vol_pres_weight > 0 else None

        # --- Load atlas segmentation (for Dice / VP loss) ---
        atlas_seg_vol = None
        if args.atlas_seg:
            atlas_seg_vol = nib.load(args.atlas_seg).get_fdata().astype(np.float32)
            print(f"  Atlas seg: {args.atlas_seg} (shape {atlas_seg_vol.shape})")

        needs_atlas_seg = (args.vol_pres_weight > 0
                           and args.mode in ('atlas-to-scan', 'scan-to-atlas'))
        if needs_atlas_seg and not args.atlas_seg:
            parser.error('--atlas-seg is required when --vol-pres-weight > 0 '
                         'in atlas-based modes (organ mask = atlas brain mask). '
                         'In scan-to-scan modes the organ mask is the moving '
                         "subject's seg and --atlas-seg is unused.")

        # Pre-compute atlas brain mask for VP loss (Dong et al. ICCV 2023)
        atlas_brain_mask_t = None
        if atlas_seg_vol is not None and args.vol_pres_weight > 0:
            atlas_brain_mask_t = torch.from_numpy(
                (atlas_seg_vol > 0).astype(np.float32)
            )[None, None].to(device)  # [1, 1, D, H, W]

        # --- Optimizer ---
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=0, amsgrad=True)

        # --- Resume from checkpoint ---
        start_epoch = 1
        ckpt_path = os.path.join(args.save_dir, 'checkpoint.pt')
        if args.weights:
            ckpt_data = torch.load(args.weights, weights_only=False,
                                   map_location=device)
            model.load_state_dict(ckpt_data['model_state_dict'])
            optimizer.load_state_dict(ckpt_data['optimizer_state_dict'])
            start_epoch = ckpt_data['epoch'] + 1
            print(f"Resumed from epoch {ckpt_data['epoch']} ({args.weights})")
        elif os.path.exists(ckpt_path):
            ckpt_data = torch.load(ckpt_path, weights_only=False,
                                   map_location=device)
            model.load_state_dict(ckpt_data['model_state_dict'])
            optimizer.load_state_dict(ckpt_data['optimizer_state_dict'])
            start_epoch = ckpt_data['epoch'] + 1
            print(f"Auto-resumed from epoch {ckpt_data['epoch']} ({ckpt_path})")

        # --- Load atlas (if atlas-to-scan mode) ---
        atlas_vol = None
        if args.atlas:
            atlas_vol = nib.load(args.atlas).get_fdata().astype(np.float32)
            assert atlas_vol.max() <= 1.0, \
                f"Atlas not normalized to [0,1]: max={atlas_vol.max():.4f}. " \
                f"Normalize before passing to training."
            assert atlas_vol.shape == vol_shape, \
                f"Atlas shape {atlas_vol.shape} != vol_shape {vol_shape}. " \
                f"Crop/resample the atlas to match."
            print(f"  Atlas: {args.atlas} (shape {atlas_vol.shape})")

        # --- Data loader (parallel prefetch, train split only) ---
        mask_in_fixed_space = (args.mode == 'atlas-to-scan')
        if args.mode == 'scan-to-scan-inter':
            # Oncologist CONCERN #4: inter-patient s2s on a pathological
            # cohort is not a clinically meaningful evaluation target.
            print('[WARNING] inter-patient s2s on pathological cohort: '
                  'training-augmentation use only; not a clinically '
                  'meaningful evaluation target. (See FINDINGS.md §2.2.)')
        if use_npz:
            if args.mode in ('atlas-to-scan', 'scan-to-atlas') and atlas_vol is not None:
                dataset = NpzAtlasDataset(
                    train_npz, atlas_vol, vol_shape,
                    load_tumor_mask=use_tumor_mask,
                    load_seg=use_seg,
                    atlas_seg=atlas_seg_vol if use_seg else None,
                    reverse=(args.mode == 'scan-to-atlas'))
            elif args.mode.startswith('scan-to-scan'):
                dataset = NpzScanPairDataset(
                    train_npz, vol_shape,
                    load_tumor_mask=use_tumor_mask,
                    load_seg=use_seg,
                    mode=args.mode,
                    patient_id_fn=extract_patient_id)
            elif atlas_vol is not None:
                dataset = NpzAtlasDataset(
                    train_npz, atlas_vol, vol_shape,
                    load_tumor_mask=use_tumor_mask,
                    load_seg=use_seg,
                    atlas_seg=atlas_seg_vol if use_seg else None)
            else:
                dataset = NpzPairDataset(train_npz, vol_shape)
        else:
            dataset = PairDataset(train_files, vol_shape)
        if isinstance(dataset, torch.utils.data.IterableDataset):
            # IterableDataset path (a2s, s2a, NpzPairDataset, mgz PairDataset):
            # batch dim is already added in __iter__, batch_size=None passes through.
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=None, num_workers=4, prefetch_factor=2,
                pin_memory=True,
            )
            loader_iter = iter(loader)
        else:
            # Map-style NpzScanPairDataset: DataLoader adds batch dim.
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=True,
                num_workers=4, prefetch_factor=2, pin_memory=True,
            )
            # Inline 2-LOC infinite iterator (TM has its own at run_transmorph.py:218).
            def _infinite_loader(loader):
                while True:
                    yield from loader
            loader_iter = _infinite_loader(loader)

        if use_npz:
            n_train = len(train_npz)
        else:
            n_train = len(train_files)
        print(f"\n  Data format:     {'npz' if use_npz else 'mgz'}")
        print(f"  Train volumes:   {n_train}")
        print(f"  vol_shape:       {vol_shape}")
        print(f"  Epochs:          {args.epochs}")
        print(f"  Steps/epoch:     {args.steps_per_epoch}")
        print(f"  Total iters:     {args.epochs * args.steps_per_epoch}")
        print(f"  Learning rate:   {args.lr}")
        print(f"  Sim loss:        {'masked_ncc (Dong VP)' if args.vp else args.loss}")
        print(f"  Reg weight:      {args.reg_param}")
        print(f"  Int steps:       {args.int_steps} ({'diffeomorphic' if args.int_steps > 0 else 'non-diffeo'})")
        print(f"  Int downsize:    {args.int_downsize}")
        print(f"  Save dir:        {args.save_dir}")
        training_mode = args.mode
        print(f"  Training mode:   {training_mode}")
        print()

        # --- Power monitor ---
        power_log = os.path.join(args.save_dir, 'power_log.csv')
        monitor = PowerMonitor(filepath=power_log, gpu_index=args.gpu)

        # --- Eval labels: use all 30 TransMorph labels present in atlas_seg ---
        if atlas_seg_vol is not None:
            atlas_labels_present = set(np.unique(atlas_seg_vol).astype(int)) - {0}
            val_labels = [l for l in EVAL_LABELS_30 if l in atlas_labels_present]
        else:
            val_labels = EVAL_LABELS_CEREBRA
        seg_warper = vxm.layers.SpatialTransformer(vol_shape, mode='nearest').to(device)

        # Pre-compute one-hot atlas seg for Dice loss (TransMorph Sec. 3.2,
        # VoxelMorph Sec. 3). In s2s the precompute is unused (mov + fix
        # one-hots are built per-batch from the subject segs), so skip it
        # to save GPU memory.
        atlas_seg_onehot_t = None
        if (args.dice_weight > 0
                and atlas_seg_vol is not None
                and args.mode in ('atlas-to-scan', 'scan-to-atlas')):
            atlas_seg_onehot_t = _seg_to_onehot(
                atlas_seg_vol, val_labels
            ).to(device)
            print(f"  Atlas seg one-hot: {atlas_seg_onehot_t.shape} ({len(val_labels)} labels)")
        # Bilinear warper for one-hot channels (Dice training)
        dice_onehot_warper = vxm.layers.SpatialTransformer(vol_shape).to(device) if args.dice_weight > 0 else None

        # --- Validation DataLoader (multi-worker for pipelined I/O) ---
        val_loader = None
        if use_npz and val_npz:
            val_dataset = NpzValDataset(val_npz, vol_shape)
            if len(val_dataset) > 0:
                val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=1, shuffle=False,
                    num_workers=4, pin_memory=True)

        # --- Training log CSV ---
        log_path = os.path.join(args.save_dir, 'train_log.csv')
        log_exists = os.path.exists(log_path)
        log_file = open(log_path, 'a')
        if not log_exists:
            log_file.write('epoch,loss,sim,reg,lr,val_dsc,val_stsr\n')
            log_file.flush()

        best_dsc = 0
        try:
            epoch_bar = trange(start_epoch, args.epochs + 1, desc='Training')
            for epoch in epoch_bar:
                model.train()
                adjust_learning_rate(optimizer, epoch, args.epochs, args.lr)

                epoch_loss = 0.0
                epoch_sim = 0.0
                epoch_reg = 0.0

                monitor.start(epoch=epoch, step=0)

                step_bar = trange(args.steps_per_epoch, desc=f'Epoch {epoch}',
                                  leave=False)
                for step in step_bar:
                    batch = next(loader_iter)
                    moving = batch[0].to(device)
                    fixed = batch[1].to(device)

                    # Batch layout depends on mode:
                    #   a2s/s2a (NpzAtlasDataset IterableDataset): (mov, fix, [tm_mov], [seg_mov, atlas_seg])
                    #   s2s (NpzScanPairDataset Map-style): (mov, fix, [tm_mov, tm_fix], [seg_mov, seg_fix])
                    if args.mode in ('atlas-to-scan', 'scan-to-atlas'):
                        tumor_mask = batch[2].to(device) if use_tumor_mask and len(batch) > 2 else None
                        tumor_mask_fix = None
                        subject_seg = batch[3].to(device) if use_seg and len(batch) > 3 else None
                        # batch[4] = atlas_seg from NpzAtlasDataset is unused in
                        # the training loop (Dice uses atlas_seg_onehot_t precomputed).
                        fixed_seg_t = None
                    else:
                        tumor_mask = batch[2].to(device) if use_tumor_mask and len(batch) > 2 else None
                        tumor_mask_fix = batch[3].to(device) if use_tumor_mask and len(batch) > 3 else None
                        seg_off = 4 if use_tumor_mask else 2
                        subject_seg = batch[seg_off].to(device) if use_seg and len(batch) > seg_off else None
                        fixed_seg_t = batch[seg_off + 1].to(device) if use_seg and len(batch) > seg_off + 1 else None
                    if tumor_mask is not None and args.vp:
                        tumor_mask = smooth_tumor_mask(tumor_mask, method=args.mask_smooth)
                    if tumor_mask_fix is not None and args.vp:
                        tumor_mask_fix = smooth_tumor_mask(tumor_mask_fix, method=args.mask_smooth)

                    moved, flow = model(moving, fixed)

                    # --- Resolve masks in fixed space (Dong et al. L365) ---
                    if args.vp and tumor_mask is not None:
                        if mask_in_fixed_space:
                            sim_mask = tumor_mask
                            vp_mask = tumor_mask
                        else:
                            with torch.no_grad():
                                warped_mask = vp_brain_warper(tumor_mask, flow) if vp_brain_warper is not None else tumor_mask
                            sim_mask = warped_mask
                            vp_mask = warped_mask
                    else:
                        sim_mask = None
                        vp_mask = None

                    # Similarity loss
                    if sim_loss_obj is not None:
                        loss_sim = sim_loss_obj(fixed, moved, mask=sim_mask) if sim_mask is not None else sim_loss_obj(fixed, moved)
                    else:
                        loss_sim = sim_loss_fn(fixed, moved)

                    # Regularization loss
                    if use_dong_reg:
                        loss_reg = regularize_loss_3d(flow.float())
                    else:
                        loss_reg = reg_loss_obj(flow, None)

                    loss = loss_sim + args.reg_param * loss_reg

                    if criterion_antifold is not None:
                        loss = loss + args.antifold_weight * criterion_antifold(flow)

                    # VP Loss — organ mask depends on mode
                    if mask_in_fixed_space:
                        organ_mask_t = atlas_brain_mask_t
                    elif subject_seg is not None:
                        organ_mask_t = (subject_seg > 0).float()
                    else:
                        organ_mask_t = None

                    if criterion_vol_pres is not None and vp_mask is not None and organ_mask_t is not None:
                        with torch.no_grad():
                            warped_organ = vp_brain_warper(
                                organ_mask_t.expand(flow.shape[0], -1, -1, -1, -1), flow)
                        loss = loss + args.vol_pres_weight * criterion_vol_pres(
                            flow, vp_mask,
                            organ_mask_moving=organ_mask_t,
                            organ_mask_warped=warped_organ)

                    # Dice loss: TransMorph Eq. 17 — warp moving seg one-hot
                    # through the flow, compare against fixed seg one-hot.
                    # In a2s: moving == atlas, so we reuse the precomputed
                    # atlas one-hot. In s2s: build mov + fix one-hots per batch.
                    if criterion_dice is not None and subject_seg is not None and (
                            atlas_seg_onehot_t is not None or args.mode.startswith('scan-to-scan')):
                        if args.mode in ('atlas-to-scan', 'scan-to-atlas'):
                            mov_seg_onehot = atlas_seg_onehot_t
                            with torch.no_grad():
                                fix_seg_onehot = _seg_to_onehot(
                                    subject_seg.squeeze().cpu().numpy(),
                                    val_labels).to(device)
                        else:
                            with torch.no_grad():
                                mov_seg_onehot = _seg_to_onehot(
                                    subject_seg.squeeze().cpu().numpy(),
                                    val_labels).to(device)
                                fix_seg_onehot = _seg_to_onehot(
                                    fixed_seg_t.squeeze().cpu().numpy(),
                                    val_labels).to(device)

                        warped_seg_onehot = dice_onehot_warper(mov_seg_onehot, flow)

                        if tumor_mask is not None:
                            if args.mode == 'atlas-to-scan' or tumor_mask_fix is None:
                                # a2s: mov-only mask (today's behavior)
                                healthy_mask = (1.0 - tumor_mask).expand_as(warped_seg_onehot)
                            else:
                                # s2s: symmetric Brett 2001 mask — intersect
                                # mov + fix healthy regions so tumor voxels
                                # on EITHER side are excluded from Dice.
                                healthy_mask = ((1.0 - tumor_mask) * (1.0 - tumor_mask_fix)).expand_as(warped_seg_onehot)
                            warped_seg_onehot = warped_seg_onehot * healthy_mask
                            fix_seg_onehot = fix_seg_onehot * healthy_mask

                        loss_dice_term = args.dice_weight * criterion_dice(
                            warped_seg_onehot, fix_seg_onehot)
                        loss = loss + loss_dice_term

                        # Machine-checkable acceptance for the s2s 1-step
                        # dry-run: ensure the Dice term is non-zero before
                        # backward. Fires only on the dedicated 1-step run.
                        if (step == 0
                                and args.steps_per_epoch == 1
                                and args.mode.startswith('scan-to-scan')):
                            print(f"[s2s-smoke] dice_term={loss_dice_term.item():.6f} "
                                  f"requires_grad={loss_dice_term.requires_grad}")

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_sim += loss_sim.item()
                    epoch_reg += loss_reg.item()

                    monitor.update_step(step)
                    step_bar.set_postfix(loss=loss.item())

                monitor.stop()

                n = args.steps_per_epoch
                avg_loss = epoch_loss / n
                avg_sim = epoch_sim / n
                avg_reg = epoch_reg / n
                current_lr = optimizer.param_groups[0]['lr']

                # --- Validation DSC + STSR ---
                eval_dsc = 0.0
                eval_stsr_val = float('nan')
                val_stsr_values = []
                if val_loader is not None:
                    model.eval()
                    dscs = []
                    bi_warper_val = vxm.layers.SpatialTransformer(vol_shape).to(device)
                    with torch.no_grad():
                        if atlas_vol is not None and atlas_seg_vol is not None:
                            # Atlas-based evaluation
                            atlas_t = torch.from_numpy(atlas_vol).float()[None, None].to(device)
                            atlas_seg_eval_t = torch.from_numpy(atlas_seg_vol).float()[None, None].to(device)
                            organ_t = (atlas_seg_eval_t > 0).float()
                            for sub_vol_t, sub_seg_t, sub_tm_t in val_loader:
                                sub_t = sub_vol_t.to(device, non_blocking=True)
                                _, pos_flow = model(atlas_t, sub_t, registration=True)

                                warped_seg = seg_warper(atlas_seg_eval_t, pos_flow)
                                warped_np = warped_seg.squeeze().cpu().numpy()
                                fix_seg = sub_seg_t.squeeze().numpy()

                                dsc = vxm.py.utils.dice(
                                    warped_np.astype(int), fix_seg.astype(int),
                                    val_labels)
                                dscs.append(np.mean(dsc))

                                # STSR (Dong et al. ICCV 2023)
                                tm = sub_tm_t.to(device, non_blocking=True)
                                if (tm > 0).float().sum() > 0:
                                    tm_bin = (tm > 0).float()
                                    warped_tm = bi_warper_val(tm_bin, pos_flow)
                                    warped_org = bi_warper_val(organ_t, pos_flow)
                                    val_stsr_values.append(
                                        eval_stsr(warped_tm, tm_bin, warped_org, organ_t))
                        else:
                            # Scan-to-scan: pair consecutive items
                            prev = None
                            for sub_vol_t, sub_seg_t, sub_tm_t in val_loader:
                                if prev is None:
                                    prev = (sub_vol_t, sub_seg_t, sub_tm_t)
                                    continue
                                mov_t = prev[0].to(device, non_blocking=True)
                                fix_t = sub_vol_t.to(device, non_blocking=True)
                                mov_seg = prev[1]
                                fix_seg = sub_seg_t.squeeze().numpy()

                                _, pos_flow = model(mov_t, fix_t, registration=True)

                                seg_t = mov_seg.to(device, non_blocking=True)
                                warped_seg = seg_warper(seg_t, pos_flow)
                                warped_np = warped_seg.squeeze().cpu().numpy()

                                dsc = vxm.py.utils.dice(
                                    warped_np.astype(int), fix_seg.astype(int),
                                    val_labels)
                                dscs.append(np.mean(dsc))
                                prev = (sub_vol_t, sub_seg_t, sub_tm_t)

                    eval_dsc = float(np.mean(dscs)) if dscs else 0.0
                    eval_stsr_val = float(np.nanmean(val_stsr_values)) if val_stsr_values else float('nan')

                stsr_str = f'{eval_stsr_val:.4f}' if not np.isnan(eval_stsr_val) else 'N/A'
                epoch_bar.set_postfix(
                    loss=f'{avg_loss:.6f}',
                    sim=f'{avg_sim:.6f}',
                    reg=f'{avg_reg:.6f}',
                    dsc=f'{eval_dsc:.4f}',
                    stsr=stsr_str,
                )

                # Log to CSV
                log_file.write(
                    f'{epoch},{avg_loss:.6f},{avg_sim:.6f},{avg_reg:.6f},'
                    f'{current_lr},{eval_dsc:.4f},{stsr_str}\n')
                log_file.flush()

                # Save checkpoint
                ckpt_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training_mode': training_mode,
                    'best_dsc': best_dsc,
                    'best_stsr': eval_stsr_val,
                }
                torch.save(ckpt_state, ckpt_path)

                if eval_dsc > best_dsc:
                    best_dsc = eval_dsc
                    ckpt_state['best_dsc'] = best_dsc
                    best_path = os.path.join(args.save_dir,
                                             f'vxm_dsc{eval_dsc:.4f}.pt')
                    torch.save(ckpt_state, best_path)
                    tqdm.write(f"  -> New best DSC: {eval_dsc:.4f}")

                if epoch % args.save_every == 0:
                    numbered = os.path.join(args.save_dir, f'vxm_{epoch:04d}.pt')
                    torch.save(ckpt_state, numbered)
                    tqdm.write(f"  -> Saved {numbered}")

            final_path = os.path.join(args.save_dir, 'vxm_final.pt')
            torch.save(ckpt_state, final_path)
            print(f"\nTraining complete. Final model: {final_path}")
            print(f"Best DSC: {best_dsc:.4f}")

        except KeyboardInterrupt:
            print("\n\nTraining interrupted — checkpoint already saved.")

        finally:
            log_file.close()
            monitor.close()
            s = monitor.summary()
            if s:
                print(f"\nPower summary: GPU {s['gpu']['avg']}W avg / "
                      f"{s['gpu']['max']}W peak | "
                      f"CPU {s['cpu']['avg']}W avg | "
                      f"{s['total_energy_wh']} Wh total")
                print(f"Full power log: {power_log}")

    elif args.eval:
        # ==========================================================
        # Evaluation (test set)
        # ==========================================================
        print("\n" + "=" * 60)
        print("  VoxelMorph — Post-Training Evaluation")
        print("=" * 60)

        if not args.weights:
            fallback = os.path.join(args.save_dir, 'vxm_final.pt')
            if os.path.exists(fallback):
                args.weights = fallback
            else:
                raise RuntimeError("--eval requires --weights or vxm_final.pt in --save-dir")

        # Build & load model
        model = vxm.networks.VxmDense(
            vol_shape, nb_features,
            int_steps=args.int_steps,
            int_downsize=args.int_downsize,
        )
        ckpt_data = torch.load(args.weights, weights_only=False,
                               map_location=device)
        model.load_state_dict(ckpt_data['model_state_dict'])
        model.to(device)
        print(f"Loaded weights: {args.weights} (epoch {ckpt_data.get('epoch', '?')})")

        # --- Load atlas segmentation (mirror training branch) ---
        atlas_seg_vol = None
        if args.atlas_seg:
            atlas_seg_vol = nib.load(args.atlas_seg).get_fdata().astype(np.float32)
            print(f"  Atlas seg: {args.atlas_seg} (shape {atlas_seg_vol.shape})")

        # --- Load atlas (mirror training branch) ---
        atlas_vol = None
        if args.atlas:
            atlas_vol = nib.load(args.atlas).get_fdata().astype(np.float32)
            assert atlas_vol.max() <= 1.0, \
                f"Atlas not normalized to [0,1]: max={atlas_vol.max():.4f}. " \
                f"Normalize before passing to training."
            assert atlas_vol.shape == vol_shape, \
                f"Atlas shape {atlas_vol.shape} != vol_shape {vol_shape}. " \
                f"Crop/resample the atlas to match."
            print(f"  Atlas: {args.atlas} (shape {atlas_vol.shape})")

        # Detect data format: .npz files or FastSurfer .mgz tree
        npz_files = collect_npz_files(args.data_dir)
        if npz_files:
            # --- .npz mode ---
            # Use test split if available
            split_path = os.path.join(args.data_dir, 'data_split.json')
            if os.path.exists(split_path):
                with open(split_path) as f:
                    saved = json.load(f)
                basename_to_path = {os.path.basename(p): p for p in npz_files}
                test_npz = [basename_to_path[os.path.basename(p)]
                            for p in saved.get('test', [])
                            if os.path.basename(p) in basename_to_path]
                if len(test_npz) >= 2:
                    npz_files = test_npz
                    print(f"Using test split: {len(npz_files)} files")

            if atlas_seg_vol is not None:
                atlas_lp = set(np.unique(atlas_seg_vol).astype(int)) - {0}
                labels = [l for l in EVAL_LABELS_30 if l in atlas_lp]
            else:
                labels = EVAL_LABELS_CEREBRA
            sub_labels = labels
            print(f"\nUsing .npz mode — {len(npz_files)} files, "
                  f"{len(labels)} eval labels")
            print(f"Evaluating {args.eval_pairs} pairs...\n")
            results = evaluate_npz(model, npz_files, vol_shape, device,
                                   labels=labels, num_pairs=args.eval_pairs,
                                   atlas_vol=atlas_vol, atlas_seg=atlas_seg_vol,
                                   mode=args.mode)
        else:
            # --- FastSurfer .mgz mode ---
            vol_files = collect_mgz_files(args.data_dir)
            split_path = os.path.join(args.save_dir, 'data_split.json')
            split = get_or_create_split(vol_files, split_path)
            test_files = split['test']
            print(f"\nEvaluating {args.eval_pairs} pairs from "
                  f"{len(test_files)} test volumes...\n")
            results = evaluate(model, test_files, vol_shape, device,
                               labels=EVAL_LABELS_CEREBRA,
                               num_pairs=args.eval_pairs)

        print(f"\n{'=' * 60}")
        print(f"  Results ({results['num_pairs']} pairs)")
        print(f"{'=' * 60}")

        n_labels = len(results.get('dice_per_label', {}))
        if results.get('baseline_dice_distribution') is not None:
            print('\n' + format_distribution(
                results['baseline_dice_distribution'],
                'Dice before', note=f'{n_labels} labels, per-pair mean'))
        if results.get('dice_distribution') is not None:
            print(format_distribution(
                results['dice_distribution'],
                'Dice after', note=f'{n_labels} labels, per-pair mean'))

        if 'subcortical_dice_mean' in results:
            n_sub = results['num_subcortical_labels']
            print(f"\n  Subcortical only ({n_sub}):")
            print(f"    Before registration:  {results['subcortical_baseline_dice_mean']:.4f} "
                  f"+/- {results['subcortical_baseline_dice_std']:.4f}")
            print(f"    After registration:   {results['subcortical_dice_mean']:.4f} "
                  f"+/- {results['subcortical_dice_std']:.4f}")

        print(f"\n  Neg Jac (%):            {results['neg_jac_pct_mean']:.4f} "
              f"+/- {results['neg_jac_pct_std']:.4f}")

        if results.get('stsr_distribution') is not None:
            print('\n' + format_distribution(
                results['stsr_distribution'],
                'STSR',
                note='1.0 = perfect tumor-volume preservation'))

        if results.get('tvcf_distribution') is not None:
            n_elig = results.get('tvcf_eligible')
            ret = results.get('tvcf_retention')
            ret_str = f"{ret*100:.1f}%" if ret is not None else 'N/A'
            tvcf_label = (f'TVCF '
                          f'(eligible={n_elig}, '
                          f'topology-filter retention={ret_str})')
            print('\n' + format_distribution(
                results['tvcf_distribution'],
                tvcf_label,
                note='1.0 = predicted change matches truth'))

        if results.get('lvcr_distribution') is not None:
            lv = results['lvcr_distribution']['mean']
            sign = ('<0 under-deformed on average' if lv < 0
                    else '>0 over-deformed on average')
            print('\n' + format_distribution(
                results['lvcr_distribution'],
                'LVCR (signed log volume-change ratio)',
                note=sign))

        # Save results
        _mode_slug = (args.mode or 'atlas-to-scan').replace('-', '_')
        results_path = os.path.join(args.save_dir, f'eval_results_{_mode_slug}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    else:
        # ==========================================================
        # Inference
        # ==========================================================

        # Build / load model
        vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=7)
        if args.weights:
            ckpt_data = torch.load(args.weights, weights_only=False,
                                   map_location=device)
            vxm_model.load_state_dict(ckpt_data['model_state_dict'])
            print(f"Loaded weights from: {args.weights} "
                  f"(epoch {ckpt_data['epoch']})")
        else:
            print("WARNING: No weights provided — model is untrained, "
                  "results will be random.")
        vxm_model.to(device)

        # ==============================================================
        # 1. Tutorial — sample data
        # ==============================================================
        print("\n" + "=" * 60)
        print("  VoxelMorph Tutorial — Sample Data")
        print("=" * 60)

        os.makedirs(DATA_DIR, exist_ok=True)
        subj1_path = os.path.join(DATA_DIR, 'subj1.npz')
        subj2_path = os.path.join(DATA_DIR, 'subj2.npz')

        if not os.path.exists(subj1_path) or not os.path.exists(subj2_path):
            import urllib.request, tarfile
            tar_path = os.path.join(DATA_DIR, 'data.tar.gz')
            print("Downloading tutorial data...")
            urllib.request.urlretrieve(
                'https://surfer.nmr.mgh.harvard.edu/pub/data/voxelmorph/tutorial_data.tar.gz',
                tar_path
            )
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=DATA_DIR)
            os.remove(tar_path)
            print("Tutorial data downloaded.")
        else:
            print("Tutorial data already present.")

        val_volume_1 = np.load(subj1_path)['vol']
        val_volume_2 = np.load(subj2_path)['vol']

        moved_pred, pred_warp = run_inference(
            vxm_model, val_volume_1, val_volume_2, device)

        plot_3x3(val_volume_2, val_volume_1, moved_pred, vol_shape)

        # ==============================================================
        # 2. Yale Data
        # ==============================================================
        print("\n" + "=" * 60)
        print("  VoxelMorph — Yale Data Registration")
        print("=" * 60)

        fixed_file = os.path.join(
            DATASET_DIR,
            'YG_MMTRCML9MOT2/2018-08-23/'
            'YG_MMTRCML9MOT2_2018-08-23_10-36-39_POST.nii.gz')
        moving_file = os.path.join(
            DATASET_DIR,
            'YG_MMTRCML9MOT2/2020-03-25/'
            'YG_MMTRCML9MOT2_2020-03-25_16-34-14_POST.nii.gz')

        fixed_vol, fixed_affine, fixed_header = load_and_preprocess_yale(
            fixed_file, target_spacing=(1.0, 1.0, 1.0),
            target_shape=(160, 192, 224))

        moving_vol, moving_affine, moving_header = load_and_preprocess_yale(
            moving_file, target_spacing=(1.0, 1.0, 1.0),
            target_shape=(160, 192, 224))

        assert moving_vol.shape == fixed_vol.shape, \
            f"Shape mismatch: {moving_vol.shape} vs {fixed_vol.shape}"

        vol_shape = moving_vol.shape
        print(f"Volume shape: {vol_shape}")

        yale_moved, yale_warp = run_inference(
            vxm_model, moving_vol, fixed_vol, device)

        plot_3x3(fixed_vol, moving_vol, yale_moved, vol_shape, offset=5)
        plot_diff(fixed_vol, moving_vol, yale_moved, vol_shape)
