#!/usr/bin/env python3
"""Batch runner for NODEO atlas-to-subject registration with CerebrA Dice.

Loads .npz subjects in-memory and runs the NODEO registration loop directly
(no subprocess, no temp files). Computes Dice on the CerebrA protocol and
Jacobian determinant statistics.

Usage:
    python run_nodeo_batch.py \
        --atlas-brain ../Atlas/mni_icbm152_t1_padded_160x192x224.nii.gz \
        --atlas-seg   ../Atlas/fastsurfer_seg_160x192x224.nii.gz \
        --data-dir    ../Voxelmorph/data/fastsurfer_preprocessed_mni_160 \
        --split-json  ../Voxelmorph/data/fastsurfer_preprocessed_mni_160/data_split.json \
        --device cuda
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# Import from NODEO's own modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Network import BrainNet
from Loss import *
from NeuralODE import *
from Utils import *

# Import shared eval labels
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'common'))
from labels import EVAL_LABELS_CEREBRA, EVAL_LABELS_30
from power_monitor import PowerMonitor
from pairs import extract_patient_id, generate_pairs
from losses import eval_tvcf, tvcf_pair_passes_filter, distribution_summary, format_distribution


# ---------------------------------------------------------------------------
# NODEO registration (inlined from original_example_from_notebook.py)
# ---------------------------------------------------------------------------

def registration(config, device, moving, fixed, tumor_mask=None,
                 organ_mask_moving=None, mask_in_fixed_space=True):
    """Register moving to fixed image via Neural ODE.

    :param config: namespace with NODEO hyperparameters.
    :param device: torch device.
    :param moving: numpy array (D, H, W) — moving image.
    :param fixed: numpy array (D, H, W) — fixed image.
    :param tumor_mask: optional numpy array (D, H, W) — binary tumor mask
                       for volume preservation (Dong et al. ICCV 2023).
    :param organ_mask_moving: optional tensor [1, 1, D, H, W] — brain mask
                              of moving image for organ ratio in VP loss.
    :param mask_in_fixed_space: if True (atlas-to-scan), mask is from fixed
                                image and already in output space — no warp.
                                If False (scan-to-atlas, scan-to-scan), mask
                                is from moving and must be warped to fixed
                                space each iteration (Dong et al. L365).
    :return: (best_df, best_df_with_grid, best_warped_moving) tensors.
    """
    im_shape = fixed.shape
    moving = torch.from_numpy(moving).to(device).float()
    fixed = torch.from_numpy(fixed).to(device).float()
    moving = moving.unsqueeze(0).unsqueeze(0)
    fixed = fixed.unsqueeze(0).unsqueeze(0)

    mask_t = None
    if tumor_mask is not None and config.lambda_vp > 0:
        mask_t = torch.from_numpy(tumor_mask).to(device).float()
        mask_t = mask_t.unsqueeze(0).unsqueeze(0)
        mask_t = (mask_t > 0).float()
        # Smooth binary mask (Brett 2001 / Dong 2023)
        mask_t = smooth_tumor_mask(mask_t, method=config.mask_smooth)

    Network = BrainNet(img_sz=im_shape,
                       smoothing_kernel=config.smoothing_kernel,
                       smoothing_win=config.smoothing_win,
                       smoothing_pass=config.smoothing_pass,
                       ds=config.ds,
                       bs=config.bs
                       ).to(device)

    ode_train = NeuralODE(Network, config.optimizer, config.STEP_SIZE).to(device)

    scale_factor = torch.tensor(im_shape).to(device).view(1, 3, 1, 1, 1) * 1.
    ST = SpatialTransformer(im_shape).to(device)
    grid = generate_grid3D_tensor(im_shape).unsqueeze(0).to(device)

    optimizer = torch.optim.Adam(ode_train.parameters(), lr=config.lr, amsgrad=True)

    # Similarity: Dong Pearson global (Eq. 11-12) when VP active,
    # NODEO NCC local (original paper) otherwise.
    if config.lambda_vp > 0:
        loss_sim_fn = MaskedNCC_Dong().to(device)
    else:
        loss_sim_fn = NCC(win=config.NCC_win)

    _vp_loss = VolumePreservationLoss().to(device) if config.lambda_vp > 0 else None
    BEST_loss_sim_loss_J = 1000
    for i in range(config.epoches):
        all_phi = ode_train(grid, Tensor(np.arange(config.time_steps)), return_whole_sequence=True)
        all_v = all_phi[1:] - all_phi[:-1]
        all_phi = (all_phi + 1.) / 2. * scale_factor
        phi = all_phi[-1]
        grid_voxel = (grid + 1.) / 2. * scale_factor
        df = phi - grid_voxel
        warped_moving, df_with_grid = ST(moving, df, return_phi=True)

        # --- Resolve masks in fixed space for similarity + VP ---
        # Dong et al. train_simple.py L365: mask warped to fixed space.
        # Jacobian is on the output/fixed grid, so VP mask must match.
        if mask_t is not None and config.lambda_vp > 0:
            if mask_in_fixed_space:
                # atlas-to-scan: mask from fixed (subject), already correct
                sim_mask = mask_t
                vp_mask = mask_t
            else:
                # scan-to-atlas / scan-to-scan: mask from moving, warp to fixed
                with torch.no_grad():
                    warped_mask = ST(mask_t, df, return_phi=False)
                sim_mask = warped_mask
                vp_mask = warped_mask
        else:
            sim_mask = None
            vp_mask = None

        # Similarity: Dong Pearson with soft mask (VP mode) or NODEO NCC
        if config.lambda_vp > 0:
            loss_sim = loss_sim_fn(warped_moving, fixed, mask=sim_mask)
        else:
            loss_sim = loss_sim_fn(warped_moving, fixed)
        warped_moving = warped_moving.squeeze(0).squeeze(0)
        loss_v = config.lambda_v * magnitude_loss(all_v)
        loss_J = config.lambda_J * neg_Jdet_loss(df_with_grid)
        loss_df = config.lambda_df * smoothloss_loss(df)
        loss = loss_sim + loss_v + loss_J + loss_df

        # Volume preservation (Dong et al. ICCV 2023 Eq. 10)
        # vp_mask is in fixed space (same grid as Jacobian)
        if vp_mask is not None and config.lambda_vp > 0 and organ_mask_moving is not None:
            with torch.no_grad():
                warped_organ = ST(organ_mask_moving, df, return_phi=False)
            loss_vp = config.lambda_vp * _vp_loss(
                df, vp_mask,
                organ_mask_moving=organ_mask_moving,
                organ_mask_warped=warped_organ)
            loss = loss + loss_vp

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 20 == 0:
            print("Iteration: {0} Loss_sim: {1:.3e} loss_J: {2:.3e}".format(
                i + 1, loss_sim.item(), loss_J.item()))
        if i > config.epoches - 50:
            loss_sim_loss_J = 1000 * loss_sim.item() * loss_J.item()
            if loss_sim_loss_J < BEST_loss_sim_loss_J:
                best_df = df.detach().clone()
                best_df_with_grid = df_with_grid.detach().clone()
                best_warped_moving = warped_moving.detach().clone()
    return best_df, best_df_with_grid, best_warped_moving


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def make_nodeo_config(epoches=500, lambda_vp=0.0, mask_smooth='gaussian'):
    """Build a config namespace with NODEO default hyperparameters.

    When lambda_vp > 0, enables Dong et al. ICCV 2023 pipeline:
      - Similarity switches from NODEO NCC local to Dong Pearson global (Eq. 11-12)
      - VP loss added (Eq. 10)
      - Tumor mask smoothed per mask_smooth method
    """
    return argparse.Namespace(
        smoothing_kernel='AK',
        smoothing_win=15,
        smoothing_pass=1,
        ds=2,
        bs=16,
        optimizer='Euler',
        STEP_SIZE=0.001,
        epoches=epoches,
        NCC_win=21,
        lr=0.005,
        lambda_J=2.5,
        lambda_df=0.05,
        lambda_v=0.00005,
        time_steps=2,
        lambda_vp=lambda_vp,
        mask_smooth=mask_smooth,
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def dice(array1, array2, labels):
    """Compute Dice overlap for a set of integer labels."""
    scores = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)
        scores[idx] = top / bottom
    return scores


def compute_jacobian_stats(df_with_grid):
    """Compute negative Jacobian ratio from df_with_grid tensor.

    df_with_grid: tensor from registration(), shape (1, D, H, W, 3).
    JacboianDet handles this shape natively (size(-1) == 3).
    """
    neg_jet = -1.0 * JacboianDet(df_with_grid)
    neg_jet = F.relu(neg_jet)
    num_neg = len(torch.where(neg_jet > 0)[0])
    total = neg_jet.numel()
    return 100.0 * num_neg / total


def warp_seg_and_dice(df, moving_seg_np, fixed_seg_np, device, labels):
    """Warp moving seg with displacement field and compute Dice vs fixed.

    df: tensor shape (1, 3, D, H, W) from registration().
    moving_seg_np: numpy (D, H, W) — moving segmentation.
    fixed_seg_np: numpy (D, H, W) — fixed segmentation (ground truth).
    labels: list of integer label IDs to evaluate.
    """
    seg_tensor = torch.from_numpy(moving_seg_np).float().unsqueeze(0).unsqueeze(0).to(device)
    st = SpatialTransformer(moving_seg_np.shape, mode='nearest').to(device)
    warped_seg = st(seg_tensor, df, return_phi=False)
    warped_seg_np = warped_seg.squeeze().cpu().numpy()
    dice_scores = dice(warped_seg_np, fixed_seg_np, labels)
    return dice_scores


# ---------------------------------------------------------------------------
# Subject list
# ---------------------------------------------------------------------------

def get_test_subjects(split_json_path):
    """Read data_split.json and return test .npz basenames."""
    with open(split_json_path) as f:
        split = json.load(f)
    return [os.path.basename(p) for p in split.get('test', [])]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Batch NODEO registration with Dice + STSR evaluation')
    parser.add_argument('--mode', type=str, default='atlas-to-scan',
                        choices=['atlas-to-scan', 'scan-to-atlas',
                                 'scan-to-scan-intra', 'scan-to-scan-inter'],
                        help='Registration mode')
    parser.add_argument('--atlas-brain', type=str, required=True,
                        help='Path to atlas brain .nii.gz')
    parser.add_argument('--atlas-seg', type=str, required=True,
                        help='Path to atlas segmentation .nii.gz')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory with subject .npz files (each has vol and seg)')
    parser.add_argument('--split-json', type=str, required=True,
                        help='Path to data_split.json')
    parser.add_argument('--output-dir', type=str, default='result/fastsurfer_batch',
                        help='Directory for results output')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (default: cuda:0)')
    parser.add_argument('--epoches', type=int, default=500,
                        help='NODEO training epochs per subject (default: 500)')
    parser.add_argument('--vol-pres-weight', type=float, default=0.0,
                        help='Volume preservation loss weight (0=disabled, paper: 0.1, Dong et al. 2023)')
    parser.add_argument('--mask-smooth', type=str, default='gaussian',
                        choices=['none', 'gaussian'],
                        help='Tumor mask smoothing: none (binary), '
                             'gaussian (Brett 2001 principle, mets-tuned 4mm FWHM). Default: gaussian')
    parser.add_argument('--max-pairs', type=int, default=100,
                        help='Max pairs for scan-to-scan-inter (default: 100)')
    parser.add_argument('--pair-seed', type=int, default=42,
                        help='Random seed for scan-to-scan-inter pair selection')
    parser.add_argument('--power-log', type=str, default=None,
                        help='Path for power log CSV (default: <output-dir>/power_log.csv)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    mode = args.mode

    # Get test subjects and generate pairs
    test_subjects = get_test_subjects(args.split_json)
    print(f"Found {len(test_subjects)} test subjects in {args.split_json}")
    pairs = generate_pairs(mode, test_subjects,
                           max_pairs=args.max_pairs, seed=args.pair_seed)
    n = len(pairs)
    print(f"Mode: {mode} — {n} pairs to evaluate")

    if not pairs:
        print("No pairs to evaluate.")
        return

    # Verify atlas files exist
    if not os.path.exists(args.atlas_brain):
        print(f"ERROR: Atlas brain not found: {args.atlas_brain}")
        sys.exit(1)
    if not os.path.exists(args.atlas_seg):
        print(f"ERROR: Atlas seg not found: {args.atlas_seg}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Power monitoring
    power_log = args.power_log or os.path.join(args.output_dir, 'power_log.csv')
    gpu_idx = int(args.device.split(':')[1]) if ':' in args.device else 0
    monitor = PowerMonitor(filepath=power_log, gpu_index=gpu_idx)

    # Load atlas once (NIfTI)
    atlas_brain_np = load_nii(args.atlas_brain)
    atlas_seg_np = load_nii(args.atlas_seg)

    # Eval labels: use all 30 TransMorph labels present in atlas_seg
    atlas_labels_present = set(np.unique(atlas_seg_np).astype(int)) - {0}
    eval_labels = [l for l in EVAL_LABELS_30 if l in atlas_labels_present]
    print(f"Eval labels: {len(eval_labels)} (of 30 TransMorph labels present in atlas)")

    # Pre-compute atlas brain mask (only needed as organ mask for atlas-to-scan)
    atlas_brain_mask_t = None
    if args.vol_pres_weight > 0 and mode == 'atlas-to-scan':
        atlas_brain_mask_t = torch.from_numpy(
            (atlas_seg_np > 0).astype(np.float32)
        )[None, None].to(device)  # [1, 1, D, H, W]

    # Build NODEO config once
    config = make_nodeo_config(
        epoches=args.epoches,
        lambda_vp=args.vol_pres_weight,
        mask_smooth=args.mask_smooth,
    )

    all_dice = []
    all_baseline = []
    all_neg_j = []
    all_runtimes = []
    all_tvcf = []
    all_lvcr = []
    tvcf_eligible_pairs = 0
    skipped = 0
    failed = 0
    cached = 0

    # Per-subject results directory for checkpointing
    subj_results_dir = os.path.join(args.output_dir, 'per_subject')
    os.makedirs(subj_results_dir, exist_ok=True)

    t0 = time.time()
    monitor.start()

    for i, (item_a, item_b) in enumerate(pairs):
        # --- Build pair_id for caching/logging ---
        name_a = item_a.replace('.npz', '')
        if mode == 'atlas-to-scan' or mode == 'scan-to-atlas':
            pair_id = name_a
        else:
            pair_id = f'{name_a}_to_{item_b.replace(".npz", "")}'
        subj_json = os.path.join(subj_results_dir, f'{pair_id}.json')

        # --- Check file existence ---
        path_a = os.path.join(args.data_dir, item_a)
        if not os.path.exists(path_a):
            print(f"  [{i+1}/{n}] SKIP {item_a} — not found")
            skipped += 1
            continue
        if item_b is not None:
            path_b = os.path.join(args.data_dir, item_b)
            if not os.path.exists(path_b):
                print(f"  [{i+1}/{n}] SKIP {item_b} — not found")
                skipped += 1
                continue

        # --- Check cache ---
        if os.path.exists(subj_json):
            try:
                with open(subj_json) as f:
                    prev = json.load(f)
                needs_tvcf = (mode in ('scan-to-scan-intra', 'scan-to-scan-inter')
                              and 'tvcf' not in prev)
                if not needs_tvcf:
                    all_baseline.append(np.array(prev['baseline_dice']))
                    all_dice.append(np.array(prev['dice_scores']))
                    all_neg_j.append(prev['neg_jac_pct'])
                    all_runtimes.append(prev['runtime_s'])
                    cached += 1
                    continue
            except Exception:
                pass

        # --- Resolve moving, fixed, masks per mode ---
        fix_tm_np = None
        if mode == 'atlas-to-scan':
            data_subj = np.load(path_a)
            moving_vol = atlas_brain_np.astype(np.float32)
            fixed_vol = data_subj['vol'].astype(np.float32)
            moving_seg_np = atlas_seg_np
            fixed_seg_np = data_subj['seg'].astype(np.int32)
            # Tumor mask from fixed (subject) — already in fixed space
            tumor_mask_np = data_subj['tumor_mask'].astype(np.float32) \
                if 'tumor_mask' in data_subj else None
            organ_mask_t = atlas_brain_mask_t  # pre-computed from atlas
            mask_in_fixed = True
            # STSR: tumor from fixed, organ from moving (atlas)
            stsr_tumor_np = tumor_mask_np
            stsr_organ_np = atlas_seg_np

        elif mode == 'scan-to-atlas':
            data_subj = np.load(path_a)
            moving_vol = data_subj['vol'].astype(np.float32)
            fixed_vol = atlas_brain_np.astype(np.float32)
            moving_seg_np = data_subj['seg'].astype(np.int32)
            fixed_seg_np = atlas_seg_np
            # Tumor mask from moving (subject) — needs warp to fixed space
            tumor_mask_np = data_subj['tumor_mask'].astype(np.float32) \
                if 'tumor_mask' in data_subj else None
            organ_mask_t = torch.from_numpy(
                (data_subj['seg'] > 0).astype(np.float32)
            )[None, None].to(device) if args.vol_pres_weight > 0 else None
            mask_in_fixed = False
            # STSR Dong: tumor + organ both from moving
            stsr_tumor_np = tumor_mask_np
            stsr_organ_np = moving_seg_np

        else:  # scan-to-scan-intra or scan-to-scan-inter
            data_a = np.load(path_a)
            data_b = np.load(path_b)
            moving_vol = data_a['vol'].astype(np.float32)
            fixed_vol = data_b['vol'].astype(np.float32)
            moving_seg_np = data_a['seg'].astype(np.int32)
            fixed_seg_np = data_b['seg'].astype(np.int32)
            # Tumor mask from moving (scan A) — needs warp to fixed space
            tumor_mask_np = data_a['tumor_mask'].astype(np.float32) \
                if 'tumor_mask' in data_a else None
            fix_tm_np = data_b['tumor_mask'].astype(np.float32) \
                if 'tumor_mask' in data_b else None
            organ_mask_t = torch.from_numpy(
                (data_a['seg'] > 0).astype(np.float32)
            )[None, None].to(device) if args.vol_pres_weight > 0 else None
            mask_in_fixed = False
            # STSR Dong: tumor + organ both from moving
            stsr_tumor_np = tumor_mask_np
            stsr_organ_np = moving_seg_np

        # Baseline Dice (before registration)
        baseline_scores = dice(moving_seg_np, fixed_seg_np, eval_labels)
        all_baseline.append(baseline_scores)

        # --- Registration ---
        print(f"  [{i+1}/{n}] {pair_id}...")
        t_sub = time.time()
        try:
            df, df_with_grid, warped_moving = registration(
                config, device, moving_vol, fixed_vol,
                tumor_mask=tumor_mask_np,
                organ_mask_moving=organ_mask_t,
                mask_in_fixed_space=mask_in_fixed)
        except Exception as e:
            print(f"    FAILED: {e}")
            failed += 1
            continue
        runtime = time.time() - t_sub
        all_runtimes.append(runtime)
        print(f"    Registration done in {runtime:.1f}s")

        # --- Evaluation ---
        try:
            with torch.no_grad():
                dice_scores = warp_seg_and_dice(
                    df, moving_seg_np, fixed_seg_np, device, eval_labels)
                neg_j_ratio = compute_jacobian_stats(df_with_grid)

                # STSR + TVCF (Dong et al. ICCV 2023)
                stsr_val = float('nan')
                tvcf_val = float('nan')
                lvcr_val = float('nan')
                if stsr_tumor_np is not None and stsr_tumor_np.sum() > 0:
                    tm_t = (torch.from_numpy(stsr_tumor_np).float()
                            [None, None].to(device) > 0).float()
                    organ_t = (torch.from_numpy(stsr_organ_np.astype(np.float32))
                               .float()[None, None].to(device) > 0).float()
                    ST_eval = SpatialTransformer(moving_vol.shape).to(device)
                    warped_tumor = ST_eval(tm_t, df, return_phi=False)
                    warped_organ = ST_eval(organ_t, df, return_phi=False)
                    stsr_val = eval_stsr(warped_tumor, tm_t,
                                         warped_organ, organ_t)
                    del organ_t, ST_eval, warped_organ  # keep warped_tumor, tm_t for TVCF

                    if (mode in ('scan-to-scan-intra', 'scan-to-scan-inter')
                            and fix_tm_np is not None and fix_tm_np.sum() > 0):
                        tvcf_eligible_pairs += 1
                        passes, _ = tvcf_pair_passes_filter(stsr_tumor_np, fix_tm_np)
                        if passes:
                            fix_tm_t = (torch.from_numpy(fix_tm_np)
                                        [None, None].to(device) > 0).float()
                            tvcf_val, lvcr_val = eval_tvcf(warped_tumor, tm_t, fix_tm_t)
                            if not np.isnan(tvcf_val):
                                all_tvcf.append(tvcf_val)
                                all_lvcr.append(lvcr_val)
                    del tm_t, warped_tumor

            stsr_type = 'adapted' if mode == 'atlas-to-scan' else 'dong'
            mean_dice = np.mean(dice_scores)
            all_dice.append(dice_scores)
            all_neg_j.append(neg_j_ratio)
            stsr_str = "%.4f" % stsr_val if not np.isnan(stsr_val) else "N/A"
            tvcf_str = "%.4f" % tvcf_val if not np.isnan(tvcf_val) else "N/A"
            print(f"    Dice: {mean_dice:.4f} | Neg J: {neg_j_ratio:.4f}% | "
                  f"STSR({stsr_type}): {stsr_str} | TVCF: {tvcf_str}")

            # Save per-pair checkpoint
            subj_result = {
                'pair_id': pair_id,
                'mode': mode,
                'stsr_type': stsr_type,
                'baseline_dice': baseline_scores.tolist(),
                'dice_scores': dice_scores.tolist(),
                'neg_jac_pct': float(neg_j_ratio),
                'stsr': stsr_val,
                'tvcf': tvcf_val,
                'lvcr': lvcr_val,
                'tumor_voxels': int(tumor_mask_np.sum()) if tumor_mask_np is not None else 0,
                'runtime_s': float(runtime),
            }
            with open(subj_json, 'w') as f:
                json.dump(subj_result, f)
        except Exception as e:
            print(f"    Eval FAILED: {e}")
            failed += 1

        # Free GPU memory
        del df, df_with_grid, warped_moving
        if organ_mask_t is not None and mode != 'atlas-to-scan':
            del organ_mask_t
        torch.cuda.empty_cache()

        monitor.update_step(i + 1)

    monitor.stop()
    total_time = time.time() - t0

    # Aggregate results
    stsr_type = 'adapted' if mode == 'atlas-to-scan' else 'dong'
    print(f"\n{'=' * 60}")
    print(f"  NODEO Batch Results — mode: {mode}")
    print(f"{'=' * 60}")
    print(f"  Total pairs:     {n}")
    print(f"  Evaluated:       {len(all_dice)}")
    print(f"  Cached:          {cached}")
    print(f"  Skipped:         {skipped}")
    print(f"  Failed:          {failed}")
    print(f"  Total time:      {total_time:.1f}s")

    if all_dice:
        all_dice = np.array(all_dice)  # (N, num_labels)
        all_baseline = np.array(all_baseline)
        mean_dice = np.mean(all_dice)
        std_dice = np.std(np.mean(all_dice, axis=1))
        baseline_dice_mean = np.mean(all_baseline)
        baseline_dice_std = np.std(np.mean(all_baseline, axis=1))
        mean_neg_j = np.mean(all_neg_j)
        std_neg_j = np.std(all_neg_j)

        # Read per-pair STSR / TVCF / LVCR from per-subject JSONs
        stsr_values = []
        tvcf_values = []
        lvcr_values = []
        for sj in sorted(os.listdir(subj_results_dir)):
            if not sj.endswith('.json'):
                continue
            with open(os.path.join(subj_results_dir, sj)) as f:
                sr = json.load(f)
            v = sr.get('stsr')
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                stsr_values.append(v)
            v = sr.get('tvcf')
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                tvcf_values.append(v)
                lvcr_values.append(sr.get('lvcr', float('nan')))

        n_labels = len(eval_labels)
        mean_per_pair_dice = np.mean(all_dice, axis=1)
        mean_per_pair_base = np.mean(all_baseline, axis=1)

        print(f"\n{'=' * 60}")
        print(f"  Results ({len(all_dice)} pairs, {n_labels} labels)")
        print(f"{'=' * 60}")

        if distribution_summary(mean_per_pair_base.tolist()) is not None:
            print('\n' + format_distribution(
                distribution_summary(mean_per_pair_base.tolist()),
                'Dice before', note=f'{n_labels} labels, per-pair mean'))
        if distribution_summary(mean_per_pair_dice.tolist()) is not None:
            print(format_distribution(
                distribution_summary(mean_per_pair_dice.tolist()),
                'Dice after', note=f'{n_labels} labels, per-pair mean'))

        print(f"\n  Neg Jac (%):  {mean_neg_j:.4f} +/- {std_neg_j:.4f}")

        stsr_dist = distribution_summary(stsr_values)
        if stsr_dist is not None:
            print('\n' + format_distribution(
                stsr_dist, f'STSR-{stsr_type}',
                note='1.0 = perfect tumor-volume preservation'))

        tvcf_dist = distribution_summary(tvcf_values)
        if tvcf_dist is not None:
            n_elig = tvcf_eligible_pairs
            ret = len(tvcf_values) / n_elig if n_elig > 0 else None
            ret_str = f"{ret*100:.1f}%" if ret is not None else 'N/A'
            print('\n' + format_distribution(
                tvcf_dist,
                f'TVCF (eligible={n_elig}, topology-filter retention={ret_str})',
                note='1.0 = predicted change matches truth'))

        lvcr_dist = distribution_summary(lvcr_values)
        if lvcr_dist is not None:
            lv = lvcr_dist['mean']
            sign = ('<0 under-deformed on average' if lv < 0
                    else '>0 over-deformed on average')
            print('\n' + format_distribution(
                lvcr_dist, 'LVCR (signed log volume-change ratio)', note=sign))

        if all_runtimes:
            print(f"\n  Avg runtime:  {np.mean(all_runtimes):.1f}s/subject")

        # Per-label Dice
        dice_per_label = {}
        for j, label in enumerate(eval_labels):
            dice_per_label[str(label)] = {
                'mean': float(np.mean(all_dice[:, j])),
                'std': float(np.std(all_dice[:, j])),
            }

        # Save results
        results = {
            'mode': mode,
            'stsr_type': stsr_type,
            'num_pairs': len(all_dice),
            'dice_mean': float(mean_dice),
            'dice_std': float(std_dice),
            'baseline_dice_mean': float(baseline_dice_mean),
            'baseline_dice_std': float(baseline_dice_std),
            'neg_jac_pct_mean': float(mean_neg_j),
            'neg_jac_pct_std': float(std_neg_j),
            'avg_runtime_s': float(np.mean(all_runtimes)) if all_runtimes else None,
            'total_time_s': float(total_time),
            'dice_per_label': dice_per_label,
            'eval_labels': eval_labels,
            'dice_distribution':      distribution_summary(mean_per_pair_dice.tolist()),
            'baseline_dice_distribution': distribution_summary(mean_per_pair_base.tolist()),
            'stsr_distribution':      stsr_dist,
            'tvcf_eligible':          tvcf_eligible_pairs,
            'tvcf_distribution':      tvcf_dist,
            'lvcr_distribution':      lvcr_dist,
        }

        # Add power summary
        power = monitor.summary()
        if power:
            results['power'] = power

        results_path = os.path.join(args.output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {results_path}")

    monitor.close()


if __name__ == '__main__':
    main()
