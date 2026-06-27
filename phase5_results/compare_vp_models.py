#!/usr/bin/env python3
"""Compare the Volume-Preserving (VP) technique across TransMorph, VoxelMorph
and NODEO on a *single* image pair.

For one pair (chosen by flag or drawn at random from the test split, restricted
to the registration mode) this runs six registrations sequentially on one GPU:

    TransMorph  VP-off / VP-on   (best-DSC checkpoint per config)
    VoxelMorph  VP-off / VP-on   (best-DSC checkpoint per config)
    NODEO       VP-off / VP-on   (per-pair optimizer, lambda_vp = 0 / 0.1)

VP-off vs VP-on for the learned models = a different *trained* checkpoint dir
(VP is a training-time loss, baked into the weights — the eval forward pass is
identical). For NODEO it is a runtime toggle.

Outputs (one folder per pair under --out-dir):
    pair_info.json                       resolved checkpoints + metrics table
    moving/fixed _vol/seg/tumor.nii.gz
    {tag}_warped_vol/seg/tumor.nii.gz    tag = tm_novp, tm_vp, vxm_novp, ...
    {tag}_flow_mag.nii.gz, {tag}_diff_after.nii.gz
    vp_compare_montage.png               3D-Slicer-like seg overlay comparison

Usage (from /home/leonardo/Documents/PUC/TCC):
    conda activate transmorph
    PYTORCH_ALLOC_CONF=expandable_segments:True python compare_vp_models.py \
        --mode scan-to-scan-intra
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import nibabel as nib
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse the proven per-model runners and IO helpers.
from compare_models_for_slicer import (
    REPO, DATA_DIR, ATLAS_VOL, ATLAS_SEG,
    _load_subject, _load_atlas, save_nifti,
    run_transmorph, run_voxelmorph, run_nodeo,
)

sys.path.insert(0, REPO)
from common.pairs import generate_pairs
from common.labels import EVAL_LABELS_30
from common.losses import (jacobian_det_3d, eval_stsr, eval_tvcf,
                           tvcf_pair_passes_filter)


# ----------------------------------------------------------------------
# Checkpoint resolution
# ----------------------------------------------------------------------

# VP-off / VP-on checkpoint dirs per mode (paths relative to REPO).
# Each entry: (transmorph_dir, voxelmorph_dir).
_CKPT_DIRS = {
    'scan-to-scan-intra': {
        'novp': ('Transmorph min/checkpoints_baseline_s2s_intra',
                 'checkpoints_s2s_intra'),
        'vp':   ('Transmorph min/checkpoints_vp_s2s_intra',
                 'checkpoints_vp_s2s_intra'),
    },
    # atlas-to-scan / scan-to-atlas share the atlas-trained checkpoints.
    'atlas': {
        'novp': ('Transmorph min/checkpoints_baseline', 'checkpoints_baseline'),
        'vp':   ('Transmorph min/checkpoints_vp',        'checkpoints_vp'),
    },
}


def default_ckpt_dirs(mode):
    """Return {'novp': (tm_dir, vxm_dir), 'vp': (tm_dir, vxm_dir)} for a mode."""
    key = mode if mode in _CKPT_DIRS else 'atlas'
    if key == 'atlas' and mode not in ('atlas-to-scan', 'scan-to-atlas'):
        # scan-to-scan-inter has no dedicated dirs by default.
        print(f'[warn] no default checkpoint dirs for mode {mode}; '
              f'pass --tm-*-dir/--vxm-*-dir explicitly.')
    out = {}
    for vp in ('novp', 'vp'):
        tm_d, vxm_d = _CKPT_DIRS[key][vp]
        out[vp] = (os.path.join(REPO, tm_d), os.path.join(REPO, vxm_d))
    return out


def pick_best_ckpt(ckpt_dir):
    """Return the highest-DSC checkpoint in a dir.

    Best = max value parsed from `*_dsc{X}.pt`; fallback to `*_final.pt`,
    then `checkpoint.pt`. Raises if nothing usable is found.
    """
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f'checkpoint dir not found: {ckpt_dir}')
    dsc = []
    for p in glob.glob(os.path.join(ckpt_dir, '*_dsc*.pt')):
        m = re.search(r'_dsc([0-9.]+)\.pt$', os.path.basename(p))
        if m:
            dsc.append((float(m.group(1)), p))
    if dsc:
        return max(dsc)[1]
    for fb in ('tm_final.pt', 'vxm_final.pt', 'checkpoint.pt'):
        p = os.path.join(ckpt_dir, fb)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'no *_dsc*.pt / *_final.pt / checkpoint.pt in {ckpt_dir}')


# ----------------------------------------------------------------------
# Pair resolution
# ----------------------------------------------------------------------

def _build_pair(mode, data_dir, atlas_path, atlas_seg_path, moving, fixed):
    """Build the pair dict (build_pair-compatible) for the 4 modes.

    Tumor-space semantics follow Dong et al.: the tumor lives in whichever of
    moving/fixed is the real subject scan.
    """
    atlas = lambda: _load_atlas(atlas_path, atlas_seg_path)
    subj = lambda name: _load_subject(os.path.join(data_dir, name))

    if mode == 'atlas-to-scan':
        a, s = atlas(), subj(fixed)
        mov, fix, in_fixed = a, s, True
        pair_id = f'a2s__{os.path.splitext(fixed)[0]}'
    elif mode == 'scan-to-atlas':
        s, a = subj(moving), atlas()
        mov, fix, in_fixed = s, a, False
        pair_id = f's2a__{os.path.splitext(moving)[0]}'
    else:  # scan-to-scan-intra / scan-to-scan-inter
        mov, fix, in_fixed = subj(moving), subj(fixed), False
        tag = 'intra' if mode == 'scan-to-scan-intra' else 'inter'
        pair_id = (f's2s_{tag}__{os.path.splitext(moving)[0]}'
                   f'__to__{os.path.splitext(fixed)[0]}')

    assert mov['vol'].shape == fix['vol'].shape, \
        f'shape mismatch {mov["vol"].shape} != {fix["vol"].shape}'
    return {
        'mode': mode,
        'mov_vol': mov['vol'], 'fix_vol': fix['vol'],
        'mov_seg': mov['seg'], 'fix_seg': fix['seg'],
        'mov_tumor': mov['tumor'], 'fix_tumor': fix['tumor'],
        'tumor_in_fixed_space': in_fixed,
        'pair_id': pair_id,
    }


def _load_tumor_mask(data_dir, name):
    """Load just the tumor mask of a subject .npz (None if absent)."""
    d = np.load(os.path.join(data_dir, name))
    return d['tumor_mask'].astype(bool) if 'tumor_mask' in d else None


def resolve_pair(args):
    """Resolve moving/fixed from explicit flags or a random mode-valid pair."""
    explicit = (args.subject or args.moving or args.fixed)
    if explicit:
        if args.mode == 'atlas-to-scan':
            fixed, moving = args.subject or args.fixed, None
        elif args.mode == 'scan-to-atlas':
            moving, fixed = args.subject or args.moving, None
        else:
            moving, fixed = args.moving, args.fixed
            if not (moving and fixed):
                sys.exit('scan-to-scan modes need both --moving and --fixed.')
    else:
        split = json.load(open(os.path.join(args.data_dir, 'data_split.json')))
        # generate_pairs needs a fixed seed only to enumerate inter-patient
        # pairs deterministically; the random *choice* below uses pair-seed
        # (None -> system entropy, so omitting --pair-seed is truly random).
        pairs = generate_pairs(args.mode, split['test'], seed=args.pair_seed or 42)
        if not pairs:
            sys.exit(f'no valid pairs for mode {args.mode} in test split.')
        # For scan-to-scan, drop pairs whose TVCF would be NaN — i.e. either
        # side lacks a tumor mask, or tumor topology changed between scans
        # (tvcf_pair_passes_filter gates the same cases eval excludes).
        if args.mode in ('scan-to-scan-intra', 'scan-to-scan-inter'):
            valid = []
            for a, b in pairs:
                ta = _load_tumor_mask(args.data_dir, a)
                tb = _load_tumor_mask(args.data_dir, b)
                if ta is None or tb is None:
                    continue
                if tvcf_pair_passes_filter(ta, tb)[0]:
                    valid.append((a, b))
            if not valid:
                sys.exit(f'no TVCF-valid pairs for mode {args.mode} '
                         f'(every candidate has a missing/incompatible tumor mask).')
            print(f'[pair] {len(valid)}/{len(pairs)} pairs are TVCF-valid')
            pairs = valid
        rng = np.random.RandomState(args.pair_seed)   # None => entropy-seeded
        a, b = pairs[rng.randint(len(pairs))]
        if args.mode in ('atlas-to-scan', 'scan-to-atlas'):
            moving = fixed = a            # subject; atlas side filled by builder
        else:
            moving, fixed = a, b
        print(f'[pair] random {args.mode}: moving={moving} fixed={fixed}')
    return _build_pair(args.mode, args.data_dir, args.atlas, args.atlas_seg,
                       moving, fixed)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def _dice(a, b, labels):
    vals = []
    for lab in labels:
        am, bm = (a == lab), (b == lab)
        denom = am.sum() + bm.sum()
        if denom > 0:
            vals.append(2.0 * (am & bm).sum() / denom)
    return float(np.mean(vals)) if vals else float('nan')


def compute_metrics(pair, run):
    """Dice (warped vs fixed seg), neg-Jac %, tumor STSR, and TVCF/LVCR.

    STSR uses common.losses.eval_stsr — identical to the eval pipelines (no
    per-pair filtering). TVCF/LVCR (signed volume-change fidelity) use
    eval_tvcf and the same tvcf_pair_passes_filter topology gate as eval;
    `tvcf_filter_pass`/`tvcf_filter_reason` flag whether eval would have
    counted this pair in its TVCF mean.
    """
    dice = _dice(np.rint(run['warped_seg']).astype(np.int32),
                 np.rint(pair['fix_seg']).astype(np.int32), EVAL_LABELS_30)

    flow = torch.from_numpy(run['flow'])[None].float()   # [1,3,D,H,W]
    det = jacobian_det_3d(flow)
    neg_jac_pct = float((det <= 0).float().mean() * 100.0)

    out = {'dice': dice, 'neg_jac_pct': neg_jac_pct, 'stsr': float('nan'),
           'tvcf': float('nan'), 'lvcr': float('nan'),
           'tvcf_filter_pass': None, 'tvcf_filter_reason': 'no_moving_tumor'}

    # Tumor metrics only when the moving side carries a tumor that we warp
    # (s2s, scan-to-atlas). In atlas-to-scan the moving atlas has none.
    mov_tumor = pair['mov_tumor']
    if (run.get('warped_tumor') is not None and mov_tumor is not None
            and mov_tumor.sum() > 0):
        wt = torch.from_numpy(run['warped_tumor'])[None, None]
        ot = torch.from_numpy((mov_tumor > 0).astype(np.float32))[None, None]
        wo = torch.from_numpy((run['warped_seg'] > 0).astype(np.float32))[None, None]
        oo = torch.from_numpy((pair['mov_seg'] > 0).astype(np.float32))[None, None]
        out['stsr'] = eval_stsr(wt, ot, wo, oo)

        # TVCF/LVCR need the fixed-space GT tumor (longitudinal s2s).
        fix_tumor = pair['fix_tumor']
        if fix_tumor is not None and fix_tumor.sum() > 0:
            ft = torch.from_numpy((fix_tumor > 0).astype(np.float32))[None, None]
            passes, reason = tvcf_pair_passes_filter(mov_tumor > 0, fix_tumor > 0)
            out['tvcf_filter_pass'] = bool(passes)
            out['tvcf_filter_reason'] = reason or ''
            out['tvcf'], out['lvcr'] = eval_tvcf(wt, ot, ft)
        else:
            out['tvcf_filter_reason'] = 'no_fixed_tumor'
    return out


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------

def _slice_idx(pair):
    """Axial (last-axis) slice through the densest part of the tumor mask
    (largest in-plane area), else volume center."""
    t = pair['fix_tumor'] if pair['fix_tumor'] is not None else pair['mov_tumor']
    if t is not None and t.sum() > 0:
        area = (t > 0).sum(axis=(0, 1))   # tumor voxels per axial slice
        return int(area.argmax())
    return pair['fix_vol'].shape[2] // 2


def _panel(ax, vol, seg, tumor, z, title):
    if vol is None:
        ax.axis('off'); return
    ax.imshow(vol[:, :, z].T, cmap='gray', origin='lower')
    sm = np.ma.masked_where(seg[:, :, z].T == 0, seg[:, :, z].T)
    ax.imshow(sm, cmap='nipy_spectral', alpha=0.4, origin='lower')
    if tumor is not None and tumor[:, :, z].sum() > 0:
        ax.contour(tumor[:, :, z].T > 0, colors='red', linewidths=1.2,
                   origin='lower')
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def _metric_line(met):
    return (f'Dice={met["dice"]:.3f}  STSR={met["stsr"]:.2f}  '
            f'negJ={met["neg_jac_pct"]:.2f}%')


def render_montage(pair, results, out_path):
    """2x4 grid:
        col 0      = moving (top) / fixed (bottom) reference
        cols 1..3  = TM / VXM / NODEO, VP-off (top) / VP-on (bottom)
    """
    z = _slice_idx(pair)
    models = ['tm', 'vxm', 'nodeo']
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), squeeze=False)

    # column 0: moving / fixed
    _panel(axes[0][0], pair['mov_vol'], pair['mov_seg'], pair['mov_tumor'],
           z, 'moving')
    _panel(axes[1][0], pair['fix_vol'], pair['fix_seg'], pair['fix_tumor'],
           z, 'fixed')

    # columns 1..3: one model each, VP-off top / VP-on bottom
    for c, m in enumerate(models, start=1):
        for r, vp in enumerate(('novp', 'vp')):
            tag, ax = f'{m}_{vp}', axes[r][c]
            if tag not in results:
                ax.axis('off'); continue
            run, met = results[tag]['run'], results[tag]['metrics']
            label = f'{m.upper()} {"VP-on" if vp == "vp" else "VP-off"}'
            _panel(ax, run['warped'], run['warped_seg'], run.get('warped_tumor'),
                   z, f'{label}\n{_metric_line(met)}')

    fig.suptitle(f'{pair["pair_id"]}  (axial z={z})', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def load_from_disk(out_dir):
    """Rebuild (pair, results) from saved NIfTI + pair_info.json metrics.

    Raw flow vectors are not persisted (only flow_mag), so per-run metrics are
    read back from pair_info.json rather than recomputed.
    """
    def _ld(name):
        p = os.path.join(out_dir, name)
        return nib.load(p).get_fdata().astype(np.float32) if os.path.exists(p) else None

    info = json.load(open(os.path.join(out_dir, 'pair_info.json')))
    pair = {
        'pair_id': info['pair_id'], 'mode': info['mode'],
        'mov_vol': _ld('moving_vol.nii.gz'), 'fix_vol': _ld('fixed_vol.nii.gz'),
        'mov_seg': _ld('moving_seg.nii.gz'), 'fix_seg': _ld('fixed_seg.nii.gz'),
        'mov_tumor': _ld('moving_tumor.nii.gz'), 'fix_tumor': _ld('fixed_tumor.nii.gz'),
    }
    results = {}
    for tag, met in info.get('runs', {}).items():
        warped = _ld(f'{tag}_warped_vol.nii.gz')
        if warped is None:
            continue
        results[tag] = {
            'run': {'warped': warped,
                    'warped_seg': _ld(f'{tag}_warped_seg.nii.gz'),
                    'warped_tumor': _ld(f'{tag}_warped_tumor.nii.gz')},
            'metrics': met,
        }
    return pair, results


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', default='scan-to-scan-intra',
                   choices=['atlas-to-scan', 'scan-to-atlas',
                            'scan-to-scan-intra', 'scan-to-scan-inter'])
    p.add_argument('--data-dir', default=DATA_DIR)
    p.add_argument('--atlas', default=ATLAS_VOL)
    p.add_argument('--atlas-seg', default=ATLAS_SEG)
    p.add_argument('--subject', default=None, help='atlas-mode subject .npz')
    p.add_argument('--moving', default=None, help='moving .npz (s2s)')
    p.add_argument('--fixed', default=None, help='fixed .npz (s2s)')
    p.add_argument('--pair-seed', type=int, default=None,
                   help='seed for random pair pick (omit = truly random)')
    p.add_argument('--tm-novp-dir', default=None)
    p.add_argument('--tm-vp-dir', default=None)
    p.add_argument('--vxm-novp-dir', default=None)
    p.add_argument('--vxm-vp-dir', default=None)
    p.add_argument('--nodeo-epoches', type=int, default=500)
    p.add_argument('--nodeo-lambda-vp', type=float, default=0.1,
                   help='VP-on lambda_vp for NODEO (Dong et al. 2023: 0.1)')
    p.add_argument('--skip-tm', action='store_true')
    p.add_argument('--skip-vxm', action='store_true')
    p.add_argument('--skip-nodeo', action='store_true')
    p.add_argument('--out-dir', default=os.path.join(REPO, 'result', 'vp_compare'))
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--montage-only', action='store_true',
                   help='skip registration; rebuild montage from existing NIfTI')
    p.add_argument('--montage-dir', default=None,
                   help='pair folder for --montage-only (default: out-dir/<pair_id>)')
    args = p.parse_args()

    # ---- montage-only: rebuild PNG from already-computed NIfTI ----
    if args.montage_only:
        out_dir = args.montage_dir
        if out_dir is None:
            pair = resolve_pair(args)
            out_dir = os.path.join(args.out_dir, pair['pair_id'])
        if not os.path.exists(os.path.join(out_dir, 'pair_info.json')):
            sys.exit(f'no pair_info.json in {out_dir} — run the pair first, '
                     f'or pass --montage-dir.')
        pair, results = load_from_disk(out_dir)
        png = os.path.join(out_dir, 'vp_compare_montage.png')
        render_montage(pair, results, png)
        print(f'Montage rebuilt: {png}')
        return

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}   Mode: {args.mode}')

    pair = resolve_pair(args)
    affine = nib.load(args.atlas).affine
    assert nib.load(args.atlas).shape == pair['mov_vol'].shape, \
        'atlas shape != data shape — affine would mislead Slicer.'

    dirs = default_ckpt_dirs(args.mode)
    tm_dirs = {'novp': args.tm_novp_dir or dirs['novp'][0],
               'vp':   args.tm_vp_dir   or dirs['vp'][0]}
    vxm_dirs = {'novp': args.vxm_novp_dir or dirs['novp'][1],
                'vp':   args.vxm_vp_dir   or dirs['vp'][1]}

    out_dir = os.path.join(args.out_dir, pair['pair_id'])
    os.makedirs(out_dir, exist_ok=True)
    print(f'Output: {out_dir}')

    # ---- baseline volumes ----
    save_nifti(pair['mov_vol'], os.path.join(out_dir, 'moving_vol.nii.gz'), affine)
    save_nifti(pair['fix_vol'], os.path.join(out_dir, 'fixed_vol.nii.gz'), affine)
    save_nifti(pair['mov_seg'], os.path.join(out_dir, 'moving_seg.nii.gz'), affine)
    save_nifti(pair['fix_seg'], os.path.join(out_dir, 'fixed_seg.nii.gz'), affine)
    if pair['mov_tumor'] is not None:
        save_nifti(pair['mov_tumor'], os.path.join(out_dir, 'moving_tumor.nii.gz'), affine)
    if pair['fix_tumor'] is not None:
        save_nifti(pair['fix_tumor'], os.path.join(out_dir, 'fixed_tumor.nii.gz'), affine)
    save_nifti(np.abs(pair['mov_vol'] - pair['fix_vol']),
               os.path.join(out_dir, 'diff_before.nii.gz'), affine)

    # ---- the six runs: model x {VP-off, VP-on} ----
    plan = []
    if not args.skip_tm:
        plan += [('tm_novp', run_transmorph, {'weights': pick_best_ckpt(tm_dirs['novp'])}),
                 ('tm_vp',   run_transmorph, {'weights': pick_best_ckpt(tm_dirs['vp'])})]
    if not args.skip_vxm:
        plan += [('vxm_novp', run_voxelmorph, {'weights': pick_best_ckpt(vxm_dirs['novp'])}),
                 ('vxm_vp',   run_voxelmorph, {'weights': pick_best_ckpt(vxm_dirs['vp'])})]
    if not args.skip_nodeo:
        plan += [('nodeo_novp', run_nodeo, {'epoches': args.nodeo_epoches, 'lambda_vp': 0.0}),
                 ('nodeo_vp',   run_nodeo, {'epoches': args.nodeo_epoches,
                                            'lambda_vp': args.nodeo_lambda_vp})]

    info = {
        'pair_id': pair['pair_id'], 'mode': args.mode,
        'data_dir': args.data_dir, 'atlas_for_affine': args.atlas,
        'vol_shape': list(pair['mov_vol'].shape),
        'moving_tumor_voxels': int(pair['mov_tumor'].sum()) if pair['mov_tumor'] is not None else 0,
        'fixed_tumor_voxels': int(pair['fix_tumor'].sum()) if pair['fix_tumor'] is not None else 0,
        'runs': {},
    }
    results = {}
    for tag, fn, kwargs in plan:
        print(f'\n=== {tag.upper()} ===')
        run = fn(pair, device, **kwargs)
        met = compute_metrics(pair, run)
        save_nifti(run['warped'], os.path.join(out_dir, f'{tag}_warped_vol.nii.gz'), affine)
        save_nifti(run['warped_seg'], os.path.join(out_dir, f'{tag}_warped_seg.nii.gz'), affine)
        if run.get('warped_tumor') is not None:
            save_nifti(run['warped_tumor'], os.path.join(out_dir, f'{tag}_warped_tumor.nii.gz'), affine)
        flow_mag = np.linalg.norm(run['flow'], axis=0)
        save_nifti(flow_mag, os.path.join(out_dir, f'{tag}_flow_mag.nii.gz'), affine)
        save_nifti(np.abs(run['warped'] - pair['fix_vol']),
                   os.path.join(out_dir, f'{tag}_diff_after.nii.gz'), affine)
        info['runs'][tag] = {
            'weights': run['weights'], 'runtime_s': run['runtime_s'],
            'dice': met['dice'], 'stsr': met['stsr'],
            'tvcf': met['tvcf'], 'lvcr': met['lvcr'],
            'tvcf_filter_pass': met['tvcf_filter_pass'],
            'tvcf_filter_reason': met['tvcf_filter_reason'],
            'neg_jac_pct': met['neg_jac_pct'],
            'flow_mag_max': float(flow_mag.max()), 'flow_mag_mean': float(flow_mag.mean()),
        }
        results[tag] = {'run': run, 'metrics': met}
        flag = '' if met['tvcf_filter_pass'] in (True, None) \
            else f' [TVCF-excluded: {met["tvcf_filter_reason"]}]'
        print(f'  runtime={run["runtime_s"]:.1f}s  Dice={met["dice"]:.3f}  '
              f'STSR={met["stsr"]:.3f}  TVCF={met["tvcf"]:.3f}  '
              f'LVCR={met["lvcr"]:.3f}  negJac={met["neg_jac_pct"]:.3f}%{flag}')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    render_montage(pair, results, os.path.join(out_dir, 'vp_compare_montage.png'))
    with open(os.path.join(out_dir, 'pair_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    # ---- compact VP-off vs VP-on table ----
    print('\n=== VP comparison (off -> on) ===')
    print(f'{"model":<8}{"Dice":>16}{"STSR":>16}{"negJac%":>16}')
    for m in ('tm', 'vxm', 'nodeo'):
        off, on = info['runs'].get(f'{m}_novp'), info['runs'].get(f'{m}_vp')
        if not (off and on):
            continue
        print(f'{m:<8}'
              f'{off["dice"]:.3f}->{on["dice"]:.3f}  '
              f'{off["stsr"]:.2f}->{on["stsr"]:.2f}  '
              f'{off["neg_jac_pct"]:.2f}->{on["neg_jac_pct"]:.2f}')
    print(f'\nDone. Montage + NIfTI in {out_dir}')


if __name__ == '__main__':
    main()
