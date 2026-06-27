#!/usr/bin/env python3
"""Run TransMorph, VoxelMorph and NODEO on the same image pair and dump every
volume as NIfTI for inspection in 3D Slicer.

Two registration modes:
  --mode atlas-to-scan   (a2s, the mode TM/VXM were actually trained on)
                         moving = MNI atlas, fixed = subject scan
  --mode scan-to-scan    (s2s-intra, what NODEO was evaluated on)
                         moving = earlier scan, fixed = later scan
                         NB: the TM/VXM weights here were trained a2s,
                         so s2s results are out-of-distribution — see
                         PIPELINE_PLAN.md Phase 4b.

Usage (from /home/leonardo/Documents/PUC/TCC):
    conda activate transmorph

    # Atlas-to-scan on a single subject (default mode)
    python compare_models_for_slicer.py --subject YG_9LLWYQOUVVVX_2019-09-07.npz

    # Scan-to-scan intra-patient
    python compare_models_for_slicer.py --mode scan-to-scan \
        --moving YG_9LLWYQOUVVVX_2019-06-08.npz \
        --fixed  YG_9LLWYQOUVVVX_2019-09-07.npz

    # Skip a model / shorten NODEO
    python compare_models_for_slicer.py --skip-nodeo
    python compare_models_for_slicer.py --nodeo-epoches 200

Outputs (one folder per pair, named after the registration):
    pair_info.json
    moving_vol/seg/tumor.nii.gz, fixed_vol/seg/tumor.nii.gz
    {tm,vxm,nodeo}_warped_vol.nii.gz
    {tm,vxm,nodeo}_warped_seg.nii.gz       (NN interp, same labels)
    {tm,vxm,nodeo}_warped_tumor.nii.gz
    {tm,vxm,nodeo}_flow_mag.nii.gz         (||flow||_2 in voxels, scalar)
    diff_before.nii.gz                     (|moving - fixed|)
    {tm,vxm,nodeo}_diff_after.nii.gz       (|warped - fixed|)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, 'Voxelmorph', 'data', 'yale_phase2_mni_160')
ATLAS_VOL = os.path.join(REPO, 'Atlas',
                         'mni_icbm152_t1_padded_160x192x224.nii.gz')
ATLAS_SEG = os.path.join(REPO, 'Atlas',
                         'fastsurfer_seg_160x192x224.nii.gz')

# Best VP checkpoints (atlas-to-scan training, Phase 4a).
TM_WEIGHTS = os.path.join(REPO, 'Transmorph min',
                          'checkpoints_vp', 'tm_dsc0.5644.pt')
VXM_WEIGHTS = os.path.join(REPO, 'checkpoints_vp', 'vxm_dsc0.5185.pt')


# ----------------------------------------------------------------------
# Pair / atlas loading
# ----------------------------------------------------------------------

def _load_subject(npz_path):
    """Return a dict with normalized vol + seg + tumor_mask for one .npz."""
    d = np.load(npz_path)
    v = d['vol'].astype(np.float32)
    if v.max() > 1: v = v / v.max()
    return {
        'vol':   v,
        'seg':   d['seg'].astype(np.float32),
        'tumor': d['tumor_mask'].astype(np.float32) if 'tumor_mask' in d else None,
    }


def _load_atlas(atlas_vol_path, atlas_seg_path):
    v = nib.load(atlas_vol_path).get_fdata().astype(np.float32)
    if v.max() > 1: v = v / v.max()
    s = nib.load(atlas_seg_path).get_fdata().astype(np.float32)
    return {'vol': v, 'seg': s, 'tumor': None}


def build_pair(mode, args):
    """Resolve moving/fixed dicts based on registration mode.

    a2s: moving = atlas, fixed = subject (tumor lives in fixed space)
    s2s: moving = subject A, fixed = subject B (tumor lives in moving space)
    """
    if mode == 'atlas-to-scan':
        atlas = _load_atlas(args.atlas, args.atlas_seg)
        subj_path = os.path.join(args.data_dir, args.subject)
        if not os.path.exists(subj_path):
            sys.exit(f'Subject not found: {subj_path}')
        subj = _load_subject(subj_path)
        assert atlas['vol'].shape == subj['vol'].shape, \
            f'Atlas {atlas["vol"].shape} != subject {subj["vol"].shape}'
        return {
            'mode':       'atlas-to-scan',
            'mov_vol':    atlas['vol'],
            'fix_vol':    subj['vol'],
            'mov_seg':    atlas['seg'],
            'fix_seg':    subj['seg'],
            'mov_tumor':  None,            # atlas has no tumor
            'fix_tumor':  subj['tumor'],   # tumor in subject = fixed space
            'tumor_in_fixed_space': True,
            'pair_id':    f'a2s__{os.path.splitext(args.subject)[0]}',
        }
    else:  # scan-to-scan
        mov_path = os.path.join(args.data_dir, args.moving)
        fix_path = os.path.join(args.data_dir, args.fixed)
        if not os.path.exists(mov_path):
            sys.exit(f'Moving not found: {mov_path}')
        if not os.path.exists(fix_path):
            sys.exit(f'Fixed not found: {fix_path}')
        m = _load_subject(mov_path)
        f = _load_subject(fix_path)
        return {
            'mode':       'scan-to-scan',
            'mov_vol':    m['vol'],
            'fix_vol':    f['vol'],
            'mov_seg':    m['seg'],
            'fix_seg':    f['seg'],
            'mov_tumor':  m['tumor'],
            'fix_tumor':  f['tumor'],
            'tumor_in_fixed_space': False,
            'pair_id':    (f's2s__{os.path.splitext(args.moving)[0]}'
                           f'__to__{os.path.splitext(args.fixed)[0]}'),
        }


def save_nifti(arr, path, affine):
    """Save a numpy volume as NIfTI with the given affine."""
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
    if arr.ndim == 4 and arr.shape[0] == 3:
        # vector field [3, D, H, W] → [D, H, W, 3] for nibabel
        arr = np.transpose(arr, (1, 2, 3, 0))
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), path)


# ----------------------------------------------------------------------
# TransMorph
# ----------------------------------------------------------------------

def run_transmorph(pair, device, weights=None):
    """Build TransMorph from the VP checkpoint and warp moving → fixed."""
    sys.path.insert(0, os.path.join(REPO, 'Transmorph min'))
    from TransMorph import CONFIGS as CONFIGS_TM
    import TransMorph as TM

    weights_path = weights or TM_WEIGHTS
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    config_name = ckpt.get('config_name', 'TransMorph')
    model_type = ckpt.get('model_type', 'transmorph')
    svf = ckpt.get('svf', False)
    time_steps = ckpt.get('time_steps', 7)
    vol_shape = tuple(ckpt.get('vol_shape', pair['mov_vol'].shape))

    config = CONFIGS_TM[config_name]
    config.out_chan = 3
    config.window_size = tuple(
        max(s // 32, 2) for s in vol_shape[:len(config.window_size)]
    )
    if model_type == 'transmorph':
        config.img_size = vol_shape
        model = TM.TransMorph(config)
    else:
        config.img_size = tuple(s // 2 for s in vol_shape)
        model = TM.TransMorphTVF(config, time_steps=time_steps, SVF=svf)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    spatial_lin = TM.SpatialTransformer(vol_shape, mode='bilinear').to(device)
    spatial_nn = TM.SpatialTransformer(vol_shape, mode='nearest').to(device)

    mov_t = torch.from_numpy(pair['mov_vol'])[None, None].to(device)
    fix_t = torch.from_numpy(pair['fix_vol'])[None, None].to(device)
    seg_t = torch.from_numpy(pair['mov_seg'])[None, None].to(device)
    has_tumor = pair['mov_tumor'] is not None and pair['mov_tumor'].sum() > 0

    t0 = time.time()
    with torch.no_grad():
        if model_type == 'transmorph':
            x = torch.cat((mov_t, fix_t), dim=1)
            warped, flow = model(x)
        else:
            mov_h = F.avg_pool3d(mov_t, 2)
            fix_h = F.avg_pool3d(fix_t, 2)
            if svf:
                flow, _ = model((mov_h, fix_h))
            else:
                flow = model((mov_h, fix_h))
            flow = F.interpolate(flow, scale_factor=2, mode='trilinear',
                                 align_corners=False) * 2
            warped = spatial_lin(mov_t, flow)
        warped_seg = spatial_nn(seg_t, flow)
        warped_tumor = None
        if has_tumor:
            tm_t = torch.from_numpy(
                (pair['mov_tumor'] > 0).astype(np.float32)
            )[None, None].to(device)
            warped_tumor = spatial_lin(tm_t, flow)
    runtime = time.time() - t0
    return {
        'warped': warped.squeeze().cpu().numpy(),
        'warped_seg': warped_seg.squeeze().cpu().numpy(),
        'warped_tumor': (warped_tumor.squeeze().cpu().numpy()
                         if warped_tumor is not None else None),
        'flow': flow.squeeze().cpu().numpy(),  # [3, D, H, W]
        'runtime_s': runtime,
        'weights': weights_path,
    }


# ----------------------------------------------------------------------
# VoxelMorph
# ----------------------------------------------------------------------

def run_voxelmorph(pair, device, weights=None):
    os.environ['VXM_BACKEND'] = 'pytorch'
    import voxelmorph as vxm

    vol_shape = pair['mov_vol'].shape
    nb_features = [
        [16, 32, 32, 32],
        [32, 32, 32, 32, 32, 16, 16],
    ]
    weights_path = weights or VXM_WEIGHTS
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    sd = ckpt['model_state_dict']
    int_steps = 7 if any(k.startswith('integrate.') for k in sd) else 0
    model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=int_steps)
    model.load_state_dict(sd)
    model.to(device).eval()

    spatial_nn = vxm.layers.SpatialTransformer(vol_shape, mode='nearest').to(device)
    spatial_lin = vxm.layers.SpatialTransformer(vol_shape).to(device)

    mov_t = torch.from_numpy(pair['mov_vol'])[None, None].to(device)
    fix_t = torch.from_numpy(pair['fix_vol'])[None, None].to(device)
    seg_t = torch.from_numpy(pair['mov_seg'])[None, None].to(device)
    has_tumor = pair['mov_tumor'] is not None and pair['mov_tumor'].sum() > 0

    t0 = time.time()
    with torch.no_grad():
        warped, flow = model(mov_t, fix_t)
        warped_seg = spatial_nn(seg_t, flow)
        warped_tumor = None
        if has_tumor:
            tm_t = torch.from_numpy(
                (pair['mov_tumor'] > 0).astype(np.float32)
            )[None, None].to(device)
            warped_tumor = spatial_lin(tm_t, flow)
    runtime = time.time() - t0
    return {
        'warped': warped.squeeze().cpu().numpy(),
        'warped_seg': warped_seg.squeeze().cpu().numpy(),
        'warped_tumor': (warped_tumor.squeeze().cpu().numpy()
                         if warped_tumor is not None else None),
        'flow': flow.squeeze().cpu().numpy(),
        'runtime_s': runtime,
        'weights': weights_path,
    }


# ----------------------------------------------------------------------
# NODEO (per-pair optimization, no pretrained weights)
# ----------------------------------------------------------------------

def run_nodeo(pair, device, epoches, lambda_vp):
    sys.path.insert(0, os.path.join(REPO, 'NODEO-DIR'))
    from run_nodeo_batch import registration, make_nodeo_config
    from Utils import SpatialTransformer

    # Tumor mask source depends on mode (Dong et al. ICCV 2023):
    #   a2s — tumor in fixed (subject), already aligned with Jacobian grid
    #   s2s — tumor in moving, must be warped each iteration
    in_fixed = pair['tumor_in_fixed_space']
    tumor_np = pair['fix_tumor'] if in_fixed else pair['mov_tumor']
    has_tumor = tumor_np is not None and tumor_np.sum() > 0

    # Organ mask (Dong VP loss). a2s: brain mask of atlas (== moving).
    # s2s: brain mask of moving (subject A).
    organ_mask_t = None
    if lambda_vp > 0:
        organ_mask_t = torch.from_numpy(
            (pair['mov_seg'] > 0).astype(np.float32)
        )[None, None].to(device)

    cfg = make_nodeo_config(epoches=epoches, lambda_vp=lambda_vp,
                            mask_smooth='gaussian')

    t0 = time.time()
    df, df_with_grid, _ = registration(
        cfg, device, pair['mov_vol'], pair['fix_vol'],
        tumor_mask=tumor_np if has_tumor else None,
        organ_mask_moving=organ_mask_t,
        mask_in_fixed_space=in_fixed,
    )
    runtime = time.time() - t0

    vol_shape = pair['mov_vol'].shape
    ST_lin = SpatialTransformer(vol_shape, mode='bilinear').to(device)
    ST_nn = SpatialTransformer(vol_shape, mode='nearest').to(device)

    mov_t = torch.from_numpy(pair['mov_vol'])[None, None].to(device)
    seg_t = torch.from_numpy(pair['mov_seg'])[None, None].to(device)

    with torch.no_grad():
        warped = ST_lin(mov_t, df)
        warped_seg = ST_nn(seg_t, df)
        warped_tumor = None
        # Warp the moving tumor only in s2s — in a2s the atlas has no tumor;
        # the user inspects fixed_tumor.nii.gz against the warped atlas seg.
        if pair['mov_tumor'] is not None and pair['mov_tumor'].sum() > 0:
            tm_t = torch.from_numpy(
                (pair['mov_tumor'] > 0).astype(np.float32)
            )[None, None].to(device)
            warped_tumor = ST_lin(tm_t, df)
    return {
        'warped': warped.squeeze().cpu().numpy(),
        'warped_seg': warped_seg.squeeze().cpu().numpy(),
        'warped_tumor': (warped_tumor.squeeze().cpu().numpy()
                         if warped_tumor is not None else None),
        'flow': df.squeeze().cpu().numpy(),  # [3, D, H, W] (z,y,x order in NODEO)
        'runtime_s': runtime,
        'weights': f'optimized {epoches} epochs (lambda_vp={lambda_vp})',
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['atlas-to-scan', 'scan-to-scan'],
                        default='atlas-to-scan',
                        help='Registration mode (default: atlas-to-scan, the '
                             'mode TM/VXM weights were trained for)')
    parser.add_argument('--subject', default='YG_9LLWYQOUVVVX_2019-09-07.npz',
                        help='Subject .npz (atlas-to-scan only)')
    parser.add_argument('--moving', default='YG_9LLWYQOUVVVX_2019-06-08.npz',
                        help='Moving .npz (scan-to-scan only)')
    parser.add_argument('--fixed',  default='YG_9LLWYQOUVVVX_2019-09-07.npz',
                        help='Fixed .npz (scan-to-scan only)')
    parser.add_argument('--data-dir', default=DATA_DIR)
    parser.add_argument('--atlas', default=ATLAS_VOL,
                        help='Atlas .nii.gz (also defines the output affine)')
    parser.add_argument('--atlas-seg', default=ATLAS_SEG,
                        help='Atlas segmentation .nii.gz (a2s only)')
    parser.add_argument('--out-dir', default=os.path.join(REPO, 'result',
                                                          'slicer_compare'))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--nodeo-epoches', type=int, default=500)
    parser.add_argument('--nodeo-lambda-vp', type=float, default=0.1,
                        help='VP loss weight for NODEO (Dong et al. 2023: 0.1)')
    parser.add_argument('--skip-tm',    action='store_true')
    parser.add_argument('--skip-vxm',   action='store_true')
    parser.add_argument('--skip-nodeo', action='store_true')
    parser.add_argument('--tm-weights',  type=str, default=None,
                        help=f'Override TM checkpoint path (default: {TM_WEIGHTS})')
    parser.add_argument('--vxm-weights', type=str, default=None,
                        help=f'Override VXM checkpoint path (default: {VXM_WEIGHTS})')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available()
                          else 'cpu')
    print(f'Device: {device}   Mode: {args.mode}')

    pair = build_pair(args.mode, args)
    affine = nib.load(args.atlas).affine
    assert nib.load(args.atlas).shape == pair['mov_vol'].shape, \
        f'Atlas shape {nib.load(args.atlas).shape} ≠ data shape ' \
        f'{pair["mov_vol"].shape} — affine would mislead Slicer.'

    out_dir = os.path.join(args.out_dir, pair['pair_id'])
    os.makedirs(out_dir, exist_ok=True)
    print(f'Output: {out_dir}')

    # ---- baseline volumes ----
    save_nifti(pair['mov_vol'],   os.path.join(out_dir, 'moving_vol.nii.gz'),   affine)
    save_nifti(pair['fix_vol'],   os.path.join(out_dir, 'fixed_vol.nii.gz'),    affine)
    save_nifti(pair['mov_seg'],   os.path.join(out_dir, 'moving_seg.nii.gz'),   affine)
    save_nifti(pair['fix_seg'],   os.path.join(out_dir, 'fixed_seg.nii.gz'),    affine)
    if pair['mov_tumor'] is not None:
        save_nifti(pair['mov_tumor'], os.path.join(out_dir, 'moving_tumor.nii.gz'), affine)
    if pair['fix_tumor'] is not None:
        save_nifti(pair['fix_tumor'], os.path.join(out_dir, 'fixed_tumor.nii.gz'), affine)
    save_nifti(np.abs(pair['mov_vol'] - pair['fix_vol']),
               os.path.join(out_dir, 'diff_before.nii.gz'), affine)

    info = {
        'pair_id': pair['pair_id'],
        'mode': pair['mode'],
        'data_dir': args.data_dir,
        'atlas_for_affine': args.atlas,
        'vol_shape': list(pair['mov_vol'].shape),
        'moving_tumor_voxels': (int(pair['mov_tumor'].sum())
                                if pair['mov_tumor'] is not None else 0),
        'fixed_tumor_voxels':  (int(pair['fix_tumor'].sum())
                                if pair['fix_tumor'] is not None else 0),
        'models': {},
    }
    if pair['mode'] == 'atlas-to-scan':
        info['subject'] = args.subject
    else:
        info['moving'] = args.moving
        info['fixed']  = args.fixed

    runs = []
    if not args.skip_tm:    runs.append(('tm',    run_transmorph,   {'weights': args.tm_weights}))
    if not args.skip_vxm:   runs.append(('vxm',   run_voxelmorph,   {'weights': args.vxm_weights}))
    if not args.skip_nodeo: runs.append(('nodeo', run_nodeo,        {
        'epoches': args.nodeo_epoches,
        'lambda_vp': args.nodeo_lambda_vp,
    }))

    for tag, fn, kwargs in runs:
        print(f'\n=== {tag.upper()} ===')
        out = fn(pair, device, **kwargs)
        save_nifti(out['warped'],
                   os.path.join(out_dir, f'{tag}_warped_vol.nii.gz'), affine)
        save_nifti(out['warped_seg'],
                   os.path.join(out_dir, f'{tag}_warped_seg.nii.gz'), affine)
        if out['warped_tumor'] is not None:
            save_nifti(out['warped_tumor'],
                       os.path.join(out_dir, f'{tag}_warped_tumor.nii.gz'), affine)
        flow_mag = np.linalg.norm(out['flow'], axis=0)
        save_nifti(flow_mag,
                   os.path.join(out_dir, f'{tag}_flow_mag.nii.gz'), affine)
        save_nifti(np.abs(out['warped'] - pair['fix_vol']),
                   os.path.join(out_dir, f'{tag}_diff_after.nii.gz'), affine)
        info['models'][tag] = {
            'weights': out['weights'],
            'runtime_s': out['runtime_s'],
            'flow_mag_max': float(flow_mag.max()),
            'flow_mag_mean': float(flow_mag.mean()),
        }
        print(f'  runtime: {out["runtime_s"]:.1f}s   '
              f'|flow| max={flow_mag.max():.2f}vox  mean={flow_mag.mean():.2f}vox')
        # release model GPU memory before the next one
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(os.path.join(out_dir, 'pair_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    print(f'\nDone. Load {out_dir} into 3D Slicer.')


if __name__ == '__main__':
    main()
