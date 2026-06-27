#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TransMorph training & evaluation script.

Based on the original IXI/TransMorph training (Chen et al.), adapted for:
  - .npz data loading (from fastsurfer preprocessing)
  - Half-resolution pipeline (avg_pool -> model -> upsample flow)
  - TransMorphTVF (multi-res + time steps)
  - PowerMonitor for energy tracking
  - Both CSV + TensorBoard logging
  - Checkpoint resume

Usage:
    conda activate transmorph

    # Train (atlas-based registration)
    python run_transmorph.py --train --data-dir ../Voxelmorph/data/fastsurfer_preprocessed

    # Resume training (auto-detects checkpoint.pt)
    python run_transmorph.py --train --data-dir ../Voxelmorph/data/fastsurfer_preprocessed

    # Evaluate (scan-to-scan Dice + Jacobian)
    python run_transmorph.py --eval --data-dir ../Voxelmorph/data/fastsurfer_preprocessed --weights checkpoints/tm_final.pt
"""

import os
import sys
import json
import glob
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("[Warning] tensorboard not installed — TensorBoard logging disabled. "
          "Install with: pip install tensorboard")
from torchvision import transforms
from argparse import ArgumentParser
from tqdm import tqdm, trange
from natsort import natsorted
from collections import defaultdict
import nibabel as nib

# Local imports
from TransMorph import CONFIGS as CONFIGS_TM
import TransMorph
# TransMorph-diff variant is out of scope (base models only); kept unusable.
TransMorphDiff = DiffBilinear = None
CONFIGS_DIFF = {}
import losses
import utils

import importlib
_common_losses_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'common', 'losses.py')
_spec = importlib.util.spec_from_file_location("common_losses", _common_losses_path)
_common_losses = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common_losses)
BendingEnergy = _common_losses.BendingEnergy
AntifoldLoss = _common_losses.AntifoldLoss
MaskedNCC = _common_losses.MaskedNCC
DiceLoss = _common_losses.DiceLoss
VolumePreservationLoss = _common_losses.VolumePreservationLoss
regularize_loss_3d = _common_losses.regularize_loss_3d
eval_stsr = _common_losses.eval_stsr
eval_tvcf = _common_losses.eval_tvcf
tvcf_pair_passes_filter = _common_losses.tvcf_pair_passes_filter
smooth_tumor_mask = _common_losses.smooth_tumor_mask
distribution_summary = _common_losses.distribution_summary
format_distribution = _common_losses.format_distribution
from data import datasets, trans

# PowerMonitor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'common'))
from power_monitor import PowerMonitor
from labels import EVAL_LABELS_30, EVAL_LABELS_CEREBRA
from pairs import extract_patient_id, generate_pairs

# ============================================================
# Labels for evaluation
# ============================================================

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
SUBCORTICAL_LABELS = [l for l in DKTATLAS_LABELS if l < 1000]


# ============================================================
# Helper functions
# ============================================================

def _seg_to_onehot(seg_np, labels):
    """Convert integer label map to one-hot [1, K, D, H, W] tensor.

    Used for differentiable Dice loss per TransMorph Sec. 3.2 / VoxelMorph Sec. 3:
    "We designed s_f and s_m as image volumes with K channels, each channel
    containing a binary mask defining the segmentation of a specific structure."

    Args:
        seg_np: numpy array [D, H, W] with integer labels
        labels: list of K label IDs to include

    Returns:
        tensor [1, K, D, H, W] float32
    """
    K = len(labels)
    D, H, W = seg_np.shape
    onehot = np.zeros((1, K, D, H, W), dtype=np.float32)
    for i, lbl in enumerate(labels):
        onehot[0, i] = (seg_np == lbl).astype(np.float32)
    return torch.from_numpy(onehot)


def adjust_learning_rate(optimizer, epoch, max_epochs, init_lr, power=0.9):
    """Polynomial LR decay (from original TransMorph training)."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - epoch / max_epochs, power), 8)


def load_atlas(atlas_path):
    """Load atlas volume from .nii.gz or .npz."""
    if atlas_path.endswith('.npz'):
        atlas = np.load(atlas_path)['vol'].astype(np.float32)
    else:
        atlas = nib.load(atlas_path).get_fdata().astype(np.float32)

    assert atlas.max() <= 1.0, \
        f"Atlas not normalized to [0,1]: max={atlas.max():.4f}. " \
        f"Normalize before passing to training."

    return atlas


def detect_labels(npz_files, max_scan=10):
    """Auto-detect non-background segmentation labels from .npz files."""
    all_labels = set()
    for p in npz_files[:max_scan]:
        try:
            with np.load(p) as d:
                if 'seg' in d:
                    all_labels.update(np.unique(d['seg']).tolist())
        except Exception:
            pass
    all_labels.discard(0)
    return sorted(all_labels)


def collect_npz_files(data_dir):
    """Find .npz files containing 'vol' arrays."""
    paths = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    valid = []
    for p in paths:
        try:
            with np.load(p) as d:
                if 'vol' in d:
                    valid.append(p)
        except Exception:
            pass
    n_seg = 0
    for p in valid:
        try:
            with np.load(p) as d:
                if 'seg' in d:
                    n_seg += 1
        except Exception:
            pass
    print(f"Found {len(valid)} .npz volumes in {data_dir} ({n_seg} with seg)")
    return valid


def get_or_create_split(npz_files, split_path, ratios=(0.8, 0.1, 0.1), seed=42):
    """Split files into train/val/test. Persists to JSON."""
    if os.path.exists(split_path):
        with open(split_path) as f:
            saved = json.load(f)
        basename_to_path = {os.path.basename(p): p for p in npz_files}
        split = {}
        for key in ('train', 'val', 'test'):
            split[key] = [basename_to_path[os.path.basename(p)]
                          for p in saved.get(key, [])
                          if os.path.basename(p) in basename_to_path]
        print(f"Loaded split: {len(split['train'])} train / "
              f"{len(split['val'])} val / {len(split['test'])} test")
        return split

    # Shuffle and split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(npz_files))
    n = len(npz_files)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    split = {
        'train': [npz_files[i] for i in indices[:n_train]],
        'val':   [npz_files[i] for i in indices[n_train:n_train + n_val]],
        'test':  [npz_files[i] for i in indices[n_train + n_val:]],
    }

    # Persist
    saved = {k: [os.path.basename(p) for p in v] for k, v in split.items()}
    with open(split_path, 'w') as f:
        json.dump(saved, f, indent=2)
    print(f"Created split: {len(split['train'])} train / "
          f"{len(split['val'])} val / {len(split['test'])} test")
    print(f"  Saved to {split_path}")
    return split


def infinite_loader(loader):
    """Wrap a DataLoader to yield infinitely (re-shuffles each pass)."""
    while True:
        for data in loader:
            yield data


def get_vol_shape(npz_files):
    """Detect volume shape from first .npz file."""
    with np.load(npz_files[0]) as d:
        return d['vol'].shape


# ============================================================
# Validation
# ============================================================

def validate_epoch(model, val_loader, vol_shape, device, svf=False,
                   labels=None, model_type='tvf',
                   atlas_vol=None, atlas_seg=None):
    """Quick validation: Dice + STSR on val set.

    Returns (mean_dsc, mean_stsr). STSR is NaN if no tumor subjects.
    """
    model.eval()
    spatial_trans_nn = TransMorph.SpatialTransformer(
        vol_shape, mode='nearest').to(device)
    spatial_trans_bi = TransMorph.SpatialTransformer(vol_shape).to(device)
    dscs = []
    stsr_vals = []
    eval_labels = labels if labels else EVAL_LABELS_CEREBRA

    if atlas_seg is not None and atlas_vol is not None:
        atlas_seg_t = torch.from_numpy(atlas_seg).float()[None, None].to(device)
        atlas_t = torch.from_numpy(atlas_vol).float()[None, None].to(device)
        organ_t = (atlas_seg_t > 0).float()

        with torch.no_grad():
            for sub_t, seg_t, tm_t in val_loader:
                sub_t = sub_t.to(device, non_blocking=True)
                subject_seg = seg_t.squeeze().numpy()

                if model_type == 'transmorph':
                    x_in = torch.cat((atlas_t, sub_t), dim=1)
                    _, flow = model(x_in)
                elif model_type == 'transmorph-diff':
                    _, deform_field, flow = model((atlas_t, sub_t))
                else:
                    atlas_half = F.avg_pool3d(atlas_t, 2)
                    sub_half = F.avg_pool3d(sub_t, 2)
                    if svf:
                        flow, _ = model((atlas_half, sub_half))
                    else:
                        flow = model((atlas_half, sub_half))
                    flow = F.interpolate(
                        flow, scale_factor=2, mode='trilinear',
                        align_corners=False) * 2

                warped_seg = spatial_trans_nn(atlas_seg_t, flow)
                warped_seg_np = warped_seg.squeeze().cpu().numpy()
                dsc_vals = utils.dice_val_labels(
                    warped_seg_np, subject_seg, eval_labels)
                dscs.append(np.mean(dsc_vals))

                # STSR (Dong et al. ICCV 2023)
                tm = tm_t.to(device, non_blocking=True)
                if (tm > 0).float().sum() > 0:
                    tm_bin = (tm > 0).float()
                    warped_tm = spatial_trans_bi(tm_bin, flow)
                    warped_org = spatial_trans_bi(organ_t, flow)
                    stsr_vals.append(eval_stsr(warped_tm, tm_bin, warped_org, organ_t))
    else:
        # Scan-to-scan validation: pair consecutive items from the loader
        with torch.no_grad():
            prev = None
            for vol_t, seg_t, tm_t in val_loader:
                if prev is None:
                    prev = (vol_t, seg_t, tm_t)
                    continue
                mov_t = prev[0].to(device, non_blocking=True)
                fix_t = vol_t.to(device, non_blocking=True)
                mov_seg = prev[1].squeeze().numpy()
                fix_seg = seg_t.squeeze().numpy()

                if model_type == 'transmorph':
                    x_in = torch.cat((mov_t, fix_t), dim=1)
                    _, flow = model(x_in)
                elif model_type == 'transmorph-diff':
                    _, deform_field, flow = model((mov_t, fix_t))
                else:
                    mov_half = F.avg_pool3d(mov_t, 2)
                    fix_half = F.avg_pool3d(fix_t, 2)
                    if svf:
                        flow, _ = model((mov_half, fix_half))
                    else:
                        flow = model((mov_half, fix_half))
                    flow = F.interpolate(
                        flow, scale_factor=2, mode='trilinear',
                        align_corners=False) * 2

                seg_t_gpu = torch.from_numpy(mov_seg).float()[None, None].to(device)
                warped_seg = spatial_trans_nn(seg_t_gpu, flow)
                warped_seg_np = warped_seg.squeeze().cpu().numpy()
                dsc_vals = utils.dice_val_labels(
                    warped_seg_np, fix_seg, eval_labels)
                dscs.append(np.mean(dsc_vals))
                prev = (vol_t, seg_t, tm_t)

    mean_dsc = np.mean(dscs) if dscs else 0.0
    mean_stsr = float(np.nanmean(stsr_vals)) if stsr_vals else float('nan')
    return mean_dsc, mean_stsr


# ============================================================
# Full evaluation
# ============================================================

def evaluate(model, npz_files, vol_shape, device, labels, num_pairs=100,
             svf=False, model_type='tvf', atlas_vol=None, atlas_seg=None,
             mode=None):
    """Evaluation: Dice + negative Jacobian %.

    If atlas_seg is provided, uses atlas-based Dice (warp atlas_seg with flow,
    compare with subject_seg). Otherwise, scan-to-scan Dice on pairs.

    When mode is 'scan-to-scan-intra' or 'scan-to-scan-inter', s2s pairs
    are sampled patient-grouped via common.pairs.generate_pairs (matches
    training-time sampling regime). When mode is None, falls back to
    fully random pair selection (legacy behavior).
    """
    files_with_seg = []
    for p in npz_files:
        try:
            with np.load(p) as d:
                if 'seg' in d:
                    files_with_seg.append(p)
        except Exception:
            pass

    model.eval()
    spatial_trans_nn = TransMorph.SpatialTransformer(
        vol_shape, mode='nearest').to(device)
    spatial_trans_bi = TransMorph.SpatialTransformer(vol_shape).to(device)

    all_dice = []
    all_baseline = []
    all_neg_jac = []
    all_stsr = []
    all_tvcf = []
    all_lvcr = []
    tvcf_eligible_pairs = 0  # pairs where both masks are non-empty

    if (atlas_seg is not None and atlas_vol is not None
            and mode not in ('scan-to-scan-intra', 'scan-to-scan-inter')):
        # Atlas-based evaluation
        if len(files_with_seg) < 1:
            raise RuntimeError(
                f"Need >= 1 .npz with seg, got {len(files_with_seg)}")

        atlas_seg_t = torch.from_numpy(atlas_seg).float()[None, None].to(device)
        atlas_t = torch.from_numpy(atlas_vol).float()[None, None].to(device)

        eval_count = min(num_pairs, len(files_with_seg))
        for idx in trange(eval_count, desc='Evaluating (atlas)'):
            d = np.load(files_with_seg[idx])
            subject = d['vol'].astype(np.float32)
            subject_seg = d['seg'].astype(np.float32)
            if subject.max() > 1:
                subject = subject / subject.max()

            # Baseline Dice (atlas_seg vs subject_seg, no registration)
            baseline = utils.dice_val_labels(atlas_seg, subject_seg, labels)
            all_baseline.append(baseline)

            sub_t = torch.from_numpy(subject).float()[None, None].to(device)

            with torch.no_grad():
                if model_type == 'transmorph':
                    x_in = torch.cat((atlas_t, sub_t), dim=1)
                    _, flow = model(x_in)
                elif model_type == 'transmorph-diff':
                    _, deform_field, flow = model((atlas_t, sub_t))
                else:
                    atlas_half = F.avg_pool3d(atlas_t, 2)
                    sub_half = F.avg_pool3d(sub_t, 2)
                    if svf:
                        flow, _ = model((atlas_half, sub_half))
                    else:
                        flow = model((atlas_half, sub_half))
                    flow = F.interpolate(
                        flow, scale_factor=2, mode='trilinear',
                        align_corners=False) * 2

                warped_seg = spatial_trans_nn(atlas_seg_t, flow)
            warped_seg_np = warped_seg.squeeze().cpu().numpy()

            dice_scores = utils.dice_val_labels(
                warped_seg_np, subject_seg, labels)
            all_dice.append(dice_scores)

            flow_np = flow.squeeze().cpu().numpy()
            jac_det = utils.jacobian_determinant_vxm(flow_np)
            neg_pct = 100.0 * np.sum(jac_det < 0) / jac_det.size
            all_neg_jac.append(neg_pct)

            # STSR (Dong et al. ICCV 2023) — only for subjects with tumor
            with torch.no_grad():
                tumor_mask_np = d['tumor_mask'] if 'tumor_mask' in d else None
                if tumor_mask_np is not None and tumor_mask_np.sum() > 0:
                    tm_t = (torch.from_numpy(tumor_mask_np.astype(np.float32))[None, None].to(device) > 0).float()
                    organ_t = (atlas_seg_t > 0).float()
                    warped_tm = spatial_trans_bi(tm_t, flow)
                    warped_org = spatial_trans_bi(organ_t, flow)
                    stsr_val = eval_stsr(warped_tm, tm_t, warped_org, organ_t)
                    all_stsr.append(stsr_val)
                    del tm_t, organ_t, warped_tm, warped_org

        actual_pairs = eval_count
    else:
        # Scan-to-scan evaluation
        if len(files_with_seg) < 2:
            raise RuntimeError(
                f"Need >= 2 .npz with seg, got {len(files_with_seg)}")

        bn_to_path = {os.path.basename(p): p for p in files_with_seg}
        if mode in ('scan-to-scan-intra', 'scan-to-scan-inter'):
            pair_basenames = generate_pairs(
                mode, [os.path.basename(p) for p in files_with_seg],
                max_pairs=num_pairs)[:num_pairs]
            print(f"  s2s eval mode={mode}: {len(pair_basenames)} pairs "
                  f"from {len(files_with_seg)} test files")
            pair_indices = [(files_with_seg.index(bn_to_path[a]),
                             files_with_seg.index(bn_to_path[b]))
                            for a, b in pair_basenames]
        else:
            rng = np.random.RandomState(0)
            pair_indices = []
            for _ in range(num_pairs):
                i, j = rng.choice(len(files_with_seg), size=2, replace=False)
                pair_indices.append((int(i), int(j)))

        for i, j in tqdm(pair_indices, desc='Evaluating'):
            dm = np.load(files_with_seg[i])
            df = np.load(files_with_seg[j])

            mov = dm['vol'].astype(np.float32)
            fix = df['vol'].astype(np.float32)
            mov_seg = dm['seg'].astype(np.float32)
            fix_seg = df['seg'].astype(np.float32)

            if mov.max() > 1:
                mov = mov / mov.max()
            if fix.max() > 1:
                fix = fix / fix.max()

            baseline = utils.dice_val_labels(mov_seg, fix_seg, labels)
            all_baseline.append(baseline)

            mov_t = torch.from_numpy(mov).float()[None, None].to(device)
            fix_t = torch.from_numpy(fix).float()[None, None].to(device)

            with torch.no_grad():
                if model_type == 'transmorph':
                    x_in = torch.cat((mov_t, fix_t), dim=1)
                    _, flow = model(x_in)
                else:
                    mov_half = F.avg_pool3d(mov_t, 2)
                    fix_half = F.avg_pool3d(fix_t, 2)
                    if svf:
                        flow, _ = model((mov_half, fix_half))
                    else:
                        flow = model((mov_half, fix_half))
                    flow = F.interpolate(
                        flow, scale_factor=2, mode='trilinear',
                        align_corners=False) * 2

            seg_t = torch.from_numpy(mov_seg).float()[None, None].to(device)
            with torch.no_grad():
                warped_seg = spatial_trans_nn(seg_t, flow)
            warped_seg_np = warped_seg.squeeze().cpu().numpy()

            dice_scores = utils.dice_val_labels(warped_seg_np, fix_seg, labels)
            all_dice.append(dice_scores)

            flow_np = flow.squeeze().cpu().numpy()
            jac_det = utils.jacobian_determinant_vxm(flow_np)
            neg_pct = 100.0 * np.sum(jac_det < 0) / jac_det.size
            all_neg_jac.append(neg_pct)

            # STSR Dong: tumor + organ from moving (Dong et al. ICCV 2023)
            with torch.no_grad():
                mov_tm_np = dm['tumor_mask'] if 'tumor_mask' in dm else None
                fix_tm_np = df['tumor_mask'] if 'tumor_mask' in df else None
                if mov_tm_np is not None and mov_tm_np.sum() > 0:
                    tm_t = (torch.from_numpy(mov_tm_np.astype(np.float32))[None, None].to(device) > 0).float()
                    organ_t = (torch.from_numpy(mov_seg)[None, None].to(device) > 0).float()
                    warped_tm = spatial_trans_bi(tm_t, flow)
                    warped_org = spatial_trans_bi(organ_t, flow)
                    all_stsr.append(eval_stsr(warped_tm, tm_t, warped_org, organ_t))

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

        actual_pairs = len(pair_indices)

    all_dice = np.array(all_dice)
    all_baseline = np.array(all_baseline)
    mean_dice_per_pair = np.nanmean(all_dice, axis=1)
    mean_baseline_per_pair = np.nanmean(all_baseline, axis=1)

    result = {
        'eval_mode': mode if mode is not None else ('atlas-to-scan' if atlas_seg is not None else 'scan-to-scan-random'),
        'dice_mean': float(np.nanmean(all_dice)),
        'dice_std': float(np.nanstd(mean_dice_per_pair)),
        'baseline_dice_mean': float(np.nanmean(all_baseline)),
        'baseline_dice_std': float(np.nanstd(mean_baseline_per_pair)),
        'dice_per_label': {int(l): float(d)
                           for l, d in zip(labels, np.nanmean(all_dice, axis=0))},
        'neg_jac_pct_mean': float(np.mean(all_neg_jac)),
        'neg_jac_pct_std': float(np.std(all_neg_jac)),
        'num_pairs': actual_pairs,
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
        sub_baseline = all_baseline[:, sub_idx]
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
# Checkpoint helpers
# ============================================================

def save_checkpoint(state, save_dir, filename, max_model_num=8):
    """Save checkpoint, prune old numbered checkpoints."""
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    # Prune old numbered checkpoints (not checkpoint.pt or tm_final.pt)
    numbered = natsorted(glob.glob(os.path.join(save_dir, 'tm_dsc*.pt')))
    while len(numbered) > max_model_num:
        os.remove(numbered[0])
        numbered = natsorted(glob.glob(os.path.join(save_dir, 'tm_dsc*.pt')))


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = ArgumentParser()
    # Mode
    parser.add_argument('--train', action='store_true',
                        help='Run training')
    parser.add_argument('--eval', action='store_true',
                        help='Post-training evaluation (Dice + Jacobian)')

    # Data
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..',
                                             'Voxelmorph', 'data',
                                             'fastsurfer_preprocessed'),
                        help='Directory of .npz files')
    parser.add_argument('--atlas', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             '..', 'Atlas',
                                             'mni_icbm152_t1_padded.nii.gz'),
                        help='Atlas path (.nii.gz or .npz)')
    parser.add_argument('--vol-shape', type=int, nargs=3,
                        default=None,
                        help='Volume shape (auto-detected from data if not set)')
    parser.add_argument('--atlas-seg', type=str, default=None,
                        help='Atlas segmentation path (.nii.gz or .npz) for '
                             'atlas-based Dice evaluation')

    # Model
    parser.add_argument('--model', type=str, default='tvf',
                        choices=['transmorph'],
                        help='Model variant: transmorph (original, full-res), '
                             'tvf (multi-res half-res pipeline, default), '
                             'or transmorph-diff (diffeomorphic, Chen 2022 Sec. 3.3)')
    parser.add_argument('--config', type=str, default=None,
                        choices=list(CONFIGS_TM.keys()),
                        help='Model config name (default: per --model)')
    parser.add_argument('--svf', action='store_true',
                        help='Diffeomorphic mode (SVF, tvf only)')
    parser.add_argument('--time-steps', type=int, default=None,
                        help='Time steps (default: 12 for SVF, 7 otherwise, tvf only)')

    # Training
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--steps-per-epoch', type=int, default=406)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sim-weight', type=float, default=1.0,
                        help='Similarity loss weight')
    parser.add_argument('--reg-weight', type=float, default=1.0,
                        help='Regularization loss weight')
    parser.add_argument('--reg-type', type=str, default='diffusion',
                        choices=['diffusion', 'bending'],
                        help='Regularizer type (default: diffusion)')
    parser.add_argument('--antifold-weight', type=float, default=0.0,
                        help='Anti-folding loss weight (0=disabled, paper: 100.0, Mok & Chung 2020)')
    parser.add_argument('--vp', action='store_true',
                        help='Enable volume preservation pipeline (Dong et al. ICCV 2023): '
                             'soft-weighted Pearson similarity + VP loss + Dong regularization')
    parser.add_argument('--mask-smooth', type=str, default='gaussian',
                        choices=['none', 'gaussian'],
                        help='Tumor mask smoothing: none (binary), '
                             'gaussian (Brett 2001 principle, mets-tuned 4mm FWHM). Default: gaussian')
    parser.add_argument('--dice-weight', type=float, default=0.0,
                        help='Dice auxiliary loss weight (0=disabled, paper: 1.0, TransMorph Eq.16)')
    parser.add_argument('--vol-pres-weight', type=float, default=0.0,
                        help='Volume preservation loss weight (0=disabled, paper: 0.1, Dong et al. 2023)')

    # Registration mode
    parser.add_argument('--mode', type=str, default='atlas-to-scan',
                        choices=['atlas-to-scan', 'scan-to-atlas',
                                 'scan-to-scan-intra', 'scan-to-scan-inter'],
                        help='Registration mode')
    parser.add_argument('--max-pairs', type=int, default=100,
                        help='Max pairs for scan-to-scan-inter eval (default: 100)')
    parser.add_argument('--pair-seed', type=int, default=42,
                        help='Random seed for scan-to-scan-inter pair selection')

    # Saving
    parser.add_argument('--save-dir', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'checkpoints'),
                        help='Directory for checkpoints and logs')
    parser.add_argument('--save-every', type=int, default=50,
                        help='Save numbered checkpoint every N epochs')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to checkpoint for resume/eval')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained weights (model only, no optimizer)')

    # Eval
    parser.add_argument('--eval-pairs', type=int, default=100,
                        help='Number of random pairs for evaluation')

    # Hardware
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision (float16) to reduce VRAM')

    args = parser.parse_args()

    # ── Default config per model type ──
    if args.config is None:
        if args.model == 'transmorph':
            args.config = 'TransMorph'
        else:
            args.config = 'TransMorph-3-LVL'

    # ── GPU setup ──
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available()
                          else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        print("Using CPU")
    torch.manual_seed(0)

    # ── Resolve data dir ──
    args.data_dir = os.path.abspath(args.data_dir)
    args.atlas = os.path.abspath(args.atlas)

    if args.train:
        # ==========================================================
        # Training
        # ==========================================================
        print("\n" + "=" * 60)
        print("  TransMorph — Training")
        print("=" * 60)

        os.makedirs(args.save_dir, exist_ok=True)

        # ── Collect data ──
        npz_files = collect_npz_files(args.data_dir)
        if len(npz_files) < 2:
            raise RuntimeError(
                f"Need at least 2 .npz volumes, found {len(npz_files)} "
                f"in {args.data_dir}")

        # ── Data split ──
        split_path = os.path.join(args.data_dir, 'data_split.json')
        split = get_or_create_split(npz_files, split_path)
        train_files = split['train']
        val_files = split['val']

        if len(train_files) < 2:
            raise RuntimeError(
                f"Need at least 2 train volumes, got {len(train_files)}")

        # ── Volume shape ──
        if args.vol_shape:
            vol_shape = tuple(args.vol_shape)
        else:
            vol_shape = get_vol_shape(npz_files)
        half_shape = tuple(s // 2 for s in vol_shape)

        # ── Load atlas ──
        atlas = load_atlas(args.atlas)
        print(f"Atlas: {args.atlas}, shape {atlas.shape}")
        assert atlas.shape == vol_shape, \
            f"Atlas shape {atlas.shape} != data vol_shape {vol_shape}. " \
            f"Crop/resample the atlas to match."

        # ── Load atlas segmentation (optional) ──
        atlas_seg_vol = None
        if args.atlas_seg:
            atlas_seg_path = os.path.abspath(args.atlas_seg)
            if atlas_seg_path.endswith('.npz'):
                atlas_seg_vol = np.load(atlas_seg_path)['seg'].astype(np.float32)
            else:
                atlas_seg_vol = nib.load(atlas_seg_path).get_fdata().astype(np.float32)
            assert atlas_seg_vol.shape == vol_shape, \
                f"Atlas seg shape {atlas_seg_vol.shape} != data vol_shape {vol_shape}. " \
                f"Crop/resample the atlas seg to match."
            print(f"Atlas seg: {atlas_seg_path}, shape {atlas_seg_vol.shape}")

        needs_atlas_seg = (args.vol_pres_weight > 0
                           and args.mode in ('atlas-to-scan', 'scan-to-atlas'))
        if needs_atlas_seg and not args.atlas_seg:
            parser.error('--atlas-seg is required when --vol-pres-weight > 0 '
                         'in atlas-based modes (organ mask = atlas brain mask). '
                         'In scan-to-scan modes the organ mask is the moving '
                         "subject's seg and --atlas-seg is unused.")

        # ── Configure model ──
        config = CONFIGS_TM[args.config]
        config.out_chan = 3

        assert vol_shape == tuple(config.img_size), \
            f"Data vol_shape {vol_shape} != config.img_size {tuple(config.img_size)}. " \
            f"Use matching data or --config."

        if args.model == 'transmorph':
            model = TransMorph.TransMorph(config)
            model.to(device)
            n_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
            print(f"Model: TransMorph ({args.config}), "
                  f"{n_params / 1e6:.2f}M params")
        elif args.model == 'transmorph-diff':
            # TransMorphDiff (Chen et al. 2022 Sec. 3.3)
            # Diffeomorphic via SVF + scaling-and-squaring (7 steps)
            # Has internal KL regularization — replaces external diffusion reg
            diff_config = CONFIGS_DIFF['TransMorphDiff']
            diff_config.img_size = vol_shape
            model = TransMorphDiff(diff_config)
            model.to(device)
            n_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
            print(f"Model: TransMorphDiff, {n_params / 1e6:.2f}M params, "
                  f"diffeomorphic (SVF, 7 int steps)")
            # Bilinear warper for seg (nearest) — used in evaluation
            diff_seg_warper = DiffBilinear(zero_boundary=True, mode='nearest').to(device)
            for p in diff_seg_warper.parameters():
                p.requires_grad = False
        else:
            config.img_size = half_shape
            time_steps = args.time_steps or (12 if args.svf else 7)
            model = TransMorph.TransMorphTVF(config, time_steps=time_steps,
                                             SVF=args.svf)
            model.to(device)
            n_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
            print(f"Model: TransMorphTVF ({args.config}), "
                  f"{n_params / 1e6:.2f}M params, "
                  f"time_steps={time_steps}, SVF={args.svf}")

        # External spatial transformer (needed for tvf, transmorph, and VP loss warping)
        spatial_trans = TransMorph.SpatialTransformer(vol_shape).to(device)

        # ── Load pretrained weights (model only) ──
        if args.pretrained:
            pretrained = torch.load(args.pretrained, map_location=device,
                                    weights_only=False)
            if 'state_dict' in pretrained:
                model.load_state_dict(pretrained['state_dict'])
            elif 'model_state_dict' in pretrained:
                model.load_state_dict(pretrained['model_state_dict'])
            else:
                model.load_state_dict(pretrained)
            print(f"Loaded pretrained weights: {args.pretrained}")

        # ── Optimizer ──
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=0, amsgrad=True)

        # ── AMP scaler ──
        scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

        # ── Losses ──
        if args.vp:
            criterion_sim = MaskedNCC().to(device)
        else:
            criterion_sim = losses.NCC_vxm().to(device)

        if args.reg_type == 'bending':
            criterion_reg = BendingEnergy().to(device)
            use_dong_reg = False
        else:
            # Dong et al. ICCV 2023 regularize_loss_3d (diffusion, divisor /2)
            # Used instead of TransMorph Grad3d (divisor /3) for consistency
            # with Dong et al. VP loss pipeline.
            use_dong_reg = True

        criterion_antifold = AntifoldLoss().to(device) if args.antifold_weight > 0 else None
        criterion_dice = DiceLoss().to(device) if args.dice_weight > 0 else None
        criterion_vol_pres = VolumePreservationLoss().to(device) if args.vol_pres_weight > 0 else None

        use_tumor_mask = args.vp or args.vol_pres_weight > 0
        use_seg = args.dice_weight > 0 or args.vol_pres_weight > 0

        # Spatial transformer for warping (tvf creates its own; transmorph needs one for Dice/moved)
        if args.model == 'transmorph':
            spatial_trans = TransMorph.SpatialTransformer(vol_shape).to(device)

        # Pre-compute atlas brain mask for VP loss (Dong et al. ICCV 2023)
        atlas_brain_mask_t = None
        if atlas_seg_vol is not None and args.vol_pres_weight > 0:
            atlas_brain_mask_t = torch.from_numpy(
                (atlas_seg_vol > 0).astype(np.float32)
            )[None, None].to(device)  # [1, 1, D, H, W]

        # atlas_seg_onehot_t is created after val_labels is defined (below)

        # ── Resume from checkpoint ──
        start_epoch = 1
        best_dsc = 0
        ckpt_path = os.path.join(args.save_dir, 'checkpoint.pt')

        if args.weights:
            ckpt = torch.load(args.weights, map_location=device,
                              weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scaler_state_dict' in ckpt:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_dsc = ckpt.get('best_dsc', 0)
            print(f"Resumed from epoch {ckpt['epoch']} ({args.weights})")
        elif os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device,
                              weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scaler_state_dict' in ckpt:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_dsc = ckpt.get('best_dsc', 0)
            print(f"Auto-resumed from epoch {ckpt['epoch']} ({ckpt_path})")

        # ── Transforms ──
        # No random flip: pathological brains are asymmetric (tumor location matters),
        # and flip would desync tumor_mask/seg. Float32 cast is redundant but harmless.
        train_composed = transforms.Compose([
            trans.NumpyType((np.float32, np.float32)),
        ])

        # ── Dataset & DataLoader ──
        mask_in_fixed_space = (args.mode == 'atlas-to-scan')
        if args.mode == 'scan-to-scan-inter':
            # Oncologist CONCERN #4: inter-patient s2s on a pathological
            # cohort is not a clinically meaningful evaluation target.
            print('[WARNING] inter-patient s2s on pathological cohort: '
                  'training-augmentation use only; not a clinically '
                  'meaningful evaluation target. (See FINDINGS.md §2.2.)')
        if args.mode in ('atlas-to-scan', 'scan-to-atlas'):
            train_set = datasets.NpzAtlasDataset(
                train_files, atlas, transforms=train_composed,
                load_tumor_mask=use_tumor_mask,
                load_seg=use_seg,
                atlas_seg=atlas_seg_vol if use_seg else None,
                reverse=(args.mode == 'scan-to-atlas'))
        else:
            # scan-to-scan: pair dataset (mov + fix tumor masks; mov + fix seg)
            train_set = datasets.NpzScanPairDataset(
                train_files,
                load_tumor_mask=use_tumor_mask,
                load_seg=use_seg,
                mode=args.mode,
                patient_id_fn=extract_patient_id)
        train_loader = DataLoader(
            train_set, batch_size=1, shuffle=True,
            num_workers=4, pin_memory=True)
        data_gen = infinite_loader(train_loader)

        # ── Validation DataLoader (multi-worker for pipelined I/O) ──
        val_dataset = datasets.NpzValDataset(val_files)
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=4, pin_memory=True)

        # ── Eval labels: use all 30 TransMorph labels present in atlas_seg ──
        if atlas_seg_vol is not None:
            atlas_labels_present = set(np.unique(atlas_seg_vol).astype(int)) - {0}
            val_labels = [l for l in EVAL_LABELS_30 if l in atlas_labels_present]
        else:
            val_labels = EVAL_LABELS_CEREBRA

        # Pre-compute one-hot atlas segmentation for Dice loss
        # (TransMorph Sec. 3.2, VoxelMorph Sec. 3: K-channel binary masks,
        # warped with bilinear interpolation for differentiable Dice).
        # In s2s the precompute is unused (mov + fix one-hots are built
        # per-batch from the subject segs), so skip it to save GPU memory.
        atlas_seg_onehot_t = None
        if (args.dice_weight > 0
                and atlas_seg_vol is not None
                and args.mode in ('atlas-to-scan', 'scan-to-atlas')):
            atlas_seg_onehot_t = _seg_to_onehot(
                atlas_seg_vol, val_labels
            ).to(device)  # [1, K, D, H, W]
            print(f"  Atlas seg one-hot: {atlas_seg_onehot_t.shape} "
                  f"({len(val_labels)} labels)")

        # ── Print config ──
        print(f"\n  Data dir:        {args.data_dir}")
        print(f"  Train volumes:   {len(train_files)}")
        print(f"  Val volumes:     {len(val_files)}")
        print(f"  Vol shape:       {vol_shape}")
        print(f"  Model type:      {args.model}")
        print(f"  Config:          {args.config}")
        print(f"  Epochs:          {args.epochs}")
        print(f"  Steps/epoch:     {args.steps_per_epoch}")
        print(f"  Learning rate:   {args.lr}")
        print(f"  Sim weight:      {args.sim_weight}")
        print(f"  Reg weight:      {args.reg_weight}")
        if args.model == 'tvf':
            print(f"  SVF:             {args.svf}")
            print(f"  Time steps:      {time_steps}")
        print(f"  AMP:             {args.amp}")
        print(f"  Save dir:        {args.save_dir}")
        print(f"  Val labels:      {len(val_labels)}")
        if atlas_seg_vol is not None:
            print(f"  Atlas seg:       {args.atlas_seg}")
        print()

        # ── Logging ──
        writer = None
        if HAS_TENSORBOARD:
            writer = SummaryWriter(
                log_dir=os.path.join(args.save_dir, 'logs'))

        log_path = os.path.join(args.save_dir, 'train_log.csv')
        log_exists = os.path.exists(log_path)
        log_file = open(log_path, 'a')
        if not log_exists:
            log_file.write('epoch,loss,sim,reg,lr,val_dsc,val_stsr\n')
            log_file.flush()

        # ── PowerMonitor ──
        power_log = os.path.join(args.save_dir, 'power_log.csv')
        monitor = PowerMonitor(filepath=power_log, gpu_index=args.gpu)

        # ── Training loop ──
        try:
            epoch_bar = trange(start_epoch, args.epochs + 1, desc='Training')
            for epoch in epoch_bar:
                model.train()
                adjust_learning_rate(optimizer, epoch, args.epochs, args.lr)

                loss_meter = utils.AverageMeter()
                sim_meter = utils.AverageMeter()
                reg_meter = utils.AverageMeter()

                monitor.start(epoch=epoch, step=0)

                step_bar = trange(args.steps_per_epoch,
                                  desc=f'Epoch {epoch}', leave=False)
                for step in step_bar:
                    batch = next(data_gen)
                    moving = batch[0].to(device)
                    fixed = batch[1].to(device)

                    # Batch layout depends on mode:
                    #   a2s (NpzAtlasDataset): (mov, fix, [tm_mov], [seg_mov, atlas_seg])
                    #   s2s (NpzScanPairDataset): (mov, fix, [tm_mov, tm_fix], [seg_mov, seg_fix])
                    if args.mode in ('atlas-to-scan', 'scan-to-atlas'):
                        tumor_mask = batch[2].to(device) if use_tumor_mask and len(batch) > 2 else None
                        tumor_mask_fix = None
                        subject_seg = batch[3].to(device) if use_seg and len(batch) > 3 else None
                        # batch[4] = atlas_seg from NpzAtlasDataset is unused in
                        # the training loop (Dice uses atlas_seg_onehot_t precomputed).
                        fixed_seg_t = None
                    else:
                        # s2s: tumor mask is two-element block, seg is two-element block
                        tumor_mask = batch[2].to(device) if use_tumor_mask and len(batch) > 2 else None
                        tumor_mask_fix = batch[3].to(device) if use_tumor_mask and len(batch) > 3 else None
                        seg_off = 4 if use_tumor_mask else 2
                        subject_seg = batch[seg_off].to(device) if use_seg and len(batch) > seg_off else None
                        fixed_seg_t = batch[seg_off + 1].to(device) if use_seg and len(batch) > seg_off + 1 else None
                    if tumor_mask is not None and args.vp:
                        tumor_mask = smooth_tumor_mask(tumor_mask, method=args.mask_smooth)
                    if tumor_mask_fix is not None and args.vp:
                        tumor_mask_fix = smooth_tumor_mask(tumor_mask_fix, method=args.mask_smooth)

                    with torch.amp.autocast('cuda', enabled=args.amp):
                        if args.model == 'transmorph':
                            # Full-resolution pipeline
                            x_in = torch.cat((moving, fixed), dim=1)
                            moved, flow = model(x_in)
                        elif args.model == 'transmorph-diff':
                            # Diffeomorphic pipeline (Chen 2022 Sec. 3.3)
                            # Returns (warped, deform_field, disp_field)
                            moved, deform_field, flow = model((moving, fixed))
                        else:
                            # Half-resolution pipeline (tvf)
                            mov_half = F.avg_pool3d(moving, 2)
                            fix_half = F.avg_pool3d(fixed, 2)
                            if args.svf:
                                flow, flow_inv = model((mov_half, fix_half))
                            else:
                                flow = model((mov_half, fix_half))
                            flow = F.interpolate(
                                flow, scale_factor=2, mode='trilinear',
                                align_corners=False) * 2
                            moved = spatial_trans(moving, flow)

                    # --- Resolve masks in fixed space (Dong et al. L365) ---
                    if args.vp and tumor_mask is not None:
                        if mask_in_fixed_space:
                            sim_mask = tumor_mask
                            vp_mask = tumor_mask
                        else:
                            with torch.no_grad():
                                warped_mask = spatial_trans(tumor_mask, flow)
                            sim_mask = warped_mask
                            vp_mask = warped_mask
                    else:
                        sim_mask = None
                        vp_mask = None

                    # Losses in float32 (NCC is unstable in fp16)
                    if args.vp and sim_mask is not None:
                        loss_sim = criterion_sim(moved.float(), fixed, mask=sim_mask) * args.sim_weight
                    else:
                        loss_sim = criterion_sim(moved.float(), fixed) * args.sim_weight

                    # Regularization
                    if args.model == 'transmorph-diff':
                        loss_reg = model.scale_reg_loss() * args.reg_weight
                    elif use_dong_reg:
                        loss_reg = regularize_loss_3d(flow.float()) * args.reg_weight
                    else:
                        loss_reg = criterion_reg(flow.float(), fixed) * args.reg_weight
                    loss = loss_sim + loss_reg

                    if criterion_antifold is not None:
                        loss = loss + args.antifold_weight * criterion_antifold(flow.float())

                    # VP Loss — organ mask depends on mode
                    if mask_in_fixed_space:
                        organ_mask_t = atlas_brain_mask_t
                    elif subject_seg is not None:
                        organ_mask_t = (subject_seg > 0).float()
                    else:
                        organ_mask_t = None

                    if criterion_vol_pres is not None and vp_mask is not None and organ_mask_t is not None:
                        with torch.no_grad():
                            warped_organ = spatial_trans(
                                organ_mask_t.expand(flow.shape[0], -1, -1, -1, -1), flow)
                        loss = loss + args.vol_pres_weight * criterion_vol_pres(
                            flow.float(), vp_mask,
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

                        warped_seg_onehot = spatial_trans(mov_seg_onehot, flow)

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
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    loss_meter.update(loss.item())
                    sim_meter.update(loss_sim.item())
                    reg_meter.update(loss_reg.item())

                    monitor.update_step(step)
                    step_bar.set_postfix(loss=f'{loss.item():.4f}')

                monitor.stop()

                current_lr = optimizer.param_groups[0]['lr']

                # ── Validation ──
                eval_dsc = 0.0
                eval_stsr_val = float('nan')
                if len(val_dataset) > 0 and val_labels and len(val_labels) > 0:
                    eval_dsc, eval_stsr_val = validate_epoch(
                        model, val_loader, vol_shape, device,
                        svf=args.svf,
                        labels=val_labels,
                        model_type=args.model,
                        atlas_vol=atlas if atlas_seg_vol is not None else None,
                        atlas_seg=atlas_seg_vol)

                # ── Logging ──
                stsr_str = f'{eval_stsr_val:.4f}' if not np.isnan(eval_stsr_val) else 'N/A'
                log_file.write(
                    f'{epoch},{loss_meter.avg:.6f},{sim_meter.avg:.6f},'
                    f'{reg_meter.avg:.6f},{current_lr},{eval_dsc:.4f},{stsr_str}\n')
                log_file.flush()

                if writer is not None:
                    writer.add_scalar('Loss/train', loss_meter.avg, epoch)
                    writer.add_scalar('Loss/sim', sim_meter.avg, epoch)
                    writer.add_scalar('Loss/reg', reg_meter.avg, epoch)
                    writer.add_scalar('LR', current_lr, epoch)
                    writer.add_scalar('DSC/validate', eval_dsc, epoch)
                    if not np.isnan(eval_stsr_val):
                        writer.add_scalar('STSR/validate', eval_stsr_val, epoch)

                epoch_bar.set_postfix(
                    loss=f'{loss_meter.avg:.4f}',
                    sim=f'{sim_meter.avg:.4f}',
                    reg=f'{reg_meter.avg:.4f}',
                    dsc=f'{eval_dsc:.4f}',
                    stsr=stsr_str,
                )

                # ── Checkpoint ──
                ckpt_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'config_name': args.config,
                    'model_type': args.model,
                    'svf': args.svf,
                    'time_steps': time_steps if args.model == 'tvf' else 0,
                    'best_dsc': best_dsc,
                    'best_stsr': eval_stsr_val,
                    'vol_shape': list(vol_shape),
                }
                torch.save(ckpt_state, ckpt_path)

                # Best model by DSC
                if eval_dsc > best_dsc:
                    best_dsc = eval_dsc
                    ckpt_state['best_dsc'] = best_dsc
                    save_checkpoint(
                        ckpt_state, args.save_dir,
                        f'tm_dsc{eval_dsc:.4f}.pt')
                    tqdm.write(
                        f"  -> New best DSC: {eval_dsc:.4f}")

                # Periodic save
                if epoch % args.save_every == 0:
                    numbered = os.path.join(
                        args.save_dir, f'tm_{epoch:04d}.pt')
                    torch.save(ckpt_state, numbered)
                    tqdm.write(f"  -> Saved {numbered}")

            # Final save
            final_path = os.path.join(args.save_dir, 'tm_final.pt')
            torch.save(ckpt_state, final_path)
            print(f"\nTraining complete. Final model: {final_path}")
            print(f"Best DSC: {best_dsc:.4f}")

        except KeyboardInterrupt:
            print("\n\nTraining interrupted — checkpoint already saved.")

        finally:
            log_file.close()
            if writer is not None:
                writer.close()
            monitor.close()
            s = monitor.summary()
            if s:
                gpu_avg = s['gpu']['avg'] if s['gpu']['avg'] is not None else '?'
                gpu_max = s['gpu']['max'] if s['gpu']['max'] is not None else '?'
                cpu_avg = s['cpu']['avg'] if s['cpu']['avg'] is not None else '?'
                print(f"\nPower summary: GPU {gpu_avg}W avg / "
                      f"{gpu_max}W peak | "
                      f"CPU {cpu_avg}W avg | "
                      f"{s['total_energy_wh']} Wh total")
                print(f"Full power log: {power_log}")

    elif args.eval:
        # ==========================================================
        # Evaluation
        # ==========================================================
        print("\n" + "=" * 60)
        print("  TransMorph — Post-Training Evaluation")
        print("=" * 60)

        # ── Find weights ──
        if not args.weights:
            fallback = os.path.join(args.save_dir, 'tm_final.pt')
            ckpt_fallback = os.path.join(args.save_dir, 'checkpoint.pt')
            if os.path.exists(fallback):
                args.weights = fallback
            elif os.path.exists(ckpt_fallback):
                args.weights = ckpt_fallback
            else:
                raise RuntimeError(
                    "--eval requires --weights or tm_final.pt in --save-dir")

        # ── Load checkpoint to get config ──
        ckpt = torch.load(args.weights, map_location=device,
                          weights_only=False)
        config_name = ckpt.get('config_name', args.config)
        model_type = ckpt.get('model_type', args.model)
        svf = ckpt.get('svf', args.svf)
        time_steps = ckpt.get('time_steps',
                              args.time_steps or (12 if svf else 7))

        # ── Collect data ──
        npz_files = collect_npz_files(args.data_dir)

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

        # ── Volume shape ──
        if args.vol_shape:
            vol_shape = tuple(args.vol_shape)
        elif 'vol_shape' in ckpt:
            vol_shape = tuple(ckpt['vol_shape'])
        else:
            vol_shape = get_vol_shape(npz_files)
        half_shape = tuple(s // 2 for s in vol_shape)

        # ── Build model ──
        config = CONFIGS_TM[config_name]
        config.out_chan = 3
        config.window_size = tuple(
            max(s // 32, 2) for s in vol_shape[:len(config.window_size)]
        )

        if model_type == 'transmorph':
            config.img_size = vol_shape
            model = TransMorph.TransMorph(config)
        else:
            config.img_size = half_shape
            model = TransMorph.TransMorphTVF(config, time_steps=time_steps,
                                             SVF=svf)

        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        print(f"Loaded weights: {args.weights} "
              f"(epoch {ckpt.get('epoch', '?')}, "
              f"model={model_type}, config={config_name}"
              f"{f', SVF={svf}' if model_type == 'tvf' else ''})")

        # ── Load atlas segmentation (optional) ──
        atlas_vol_eval = None
        atlas_seg_eval = None
        if args.atlas_seg:
            atlas_seg_path = os.path.abspath(args.atlas_seg)
            if atlas_seg_path.endswith('.npz'):
                atlas_seg_eval = np.load(atlas_seg_path)['seg'].astype(np.float32)
            else:
                atlas_seg_eval = nib.load(atlas_seg_path).get_fdata().astype(np.float32)
            if atlas_seg_eval.shape != vol_shape:
                from scipy.ndimage import zoom as zoom_fn
                zf = [vol_shape[i] / atlas_seg_eval.shape[i] for i in range(3)]
                atlas_seg_eval = zoom_fn(atlas_seg_eval, zf, order=0).astype(np.float32)

            # Also load atlas volume for forward pass
            atlas_vol_eval = load_atlas(args.atlas)
            if atlas_vol_eval.shape != vol_shape:
                from scipy.ndimage import zoom as zoom_fn2
                zf2 = [vol_shape[i] / atlas_vol_eval.shape[i] for i in range(3)]
                atlas_vol_eval = zoom_fn2(atlas_vol_eval, zf2, order=3).astype(np.float32)
                if atlas_vol_eval.max() > 0:
                    atlas_vol_eval = atlas_vol_eval / atlas_vol_eval.max()
            print(f"Atlas seg: {atlas_seg_path}, shape {atlas_seg_eval.shape}")

        # ── Eval labels: use all 30 TransMorph labels present in atlas_seg ──
        if atlas_seg_eval is not None:
            atlas_lp = set(np.unique(atlas_seg_eval).astype(int)) - {0}
            labels = [l for l in EVAL_LABELS_30 if l in atlas_lp]
        else:
            labels = EVAL_LABELS_CEREBRA
        sub_labels = labels
        print(f"\n  Vol shape:    {vol_shape}")
        print(f"  Model type:   {model_type}")
        print(f"  Files:        {len(npz_files)}")
        print(f"  Seg labels:   {len(labels)}")
        if sub_labels:
            print(f"  Subcortical:  {len(sub_labels)}")
        print(f"  Eval pairs:   {args.eval_pairs}")
        if atlas_seg_eval is not None:
            print(f"  Atlas seg:    {args.atlas_seg}")
        print()

        # ── Run evaluation ──
        results = evaluate(
            model, npz_files, vol_shape, device,
            labels=labels, num_pairs=args.eval_pairs, svf=svf,
            model_type=model_type,
            atlas_vol=atlas_vol_eval, atlas_seg=atlas_seg_eval,
            mode=args.mode)

        # ── Print results ──
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
            print(f"    Before registration:  "
                  f"{results['subcortical_baseline_dice_mean']:.4f} "
                  f"+/- {results['subcortical_baseline_dice_std']:.4f}")
            print(f"    After registration:   "
                  f"{results['subcortical_dice_mean']:.4f} "
                  f"+/- {results['subcortical_dice_std']:.4f}")

        print(f"\n  Neg Jac (%):            "
              f"{results['neg_jac_pct_mean']:.4f} "
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

        # ── Save results ──
        _mode_slug = (args.mode or 'atlas-to-scan').replace('-', '_')
        results_path = os.path.join(args.save_dir, f'eval_results_{_mode_slug}.json')
        os.makedirs(args.save_dir, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    else:
        parser.print_help()
        print("\nUse --train or --eval to select a mode.")
