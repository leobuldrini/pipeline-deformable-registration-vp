"""
Phase 3 loss functions for deformable image registration with pathology.

All losses are backed by published literature:
- BendingEnergy: Rueckert et al. 1999; TransMorph Eq. 15
- AntifoldLoss: VoxelMorph-diff (Dalca et al. 2019), SYMNet (Mok & Chung 2020)
- MaskedNCC: Dong et al. ICCV 2023 (soft-weighted global Pearson correlation)
- DiceLoss: wrapper around voxelmorph.torch.losses.Dice (Balakrishnan et al. 2019)
  with one-hot encoding per TransMorph Eq. 16-17 (Chen et al. 2022 Sec. 3.2)
- regularize_loss_3d: Dong et al. ICCV 2023 (diffusion reg, divisor /2)
- VolumePreservationLoss: Dong et al. ICCV 2023 Eq. 10
- compute_stsr: Dong et al. ICCV 2023 (evaluation metric)
- eval_tvcf: longitudinal-fidelity adaptation of STSR's symmetric ratio
  (Dong et al. ICCV 2023) to predicted-vs-GT volume change in
  longitudinal tumor registration (Sarkar et al. IJROBP 83(3):1038, 2011).
"""

import math
import os
import importlib.util

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
#  Import published losses directly from paper repositories
# ---------------------------------------------------------------------------

# VoxelMorph Dice (Balakrishnan et al. 2019) — bypass voxelmorph __init__
# which requires TensorFlow. The torch losses module is self-contained.
_vxm_losses_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'miniconda3', 'envs',
    'transmorph', 'lib', 'python3.11', 'site-packages',
    'voxelmorph', 'torch', 'losses.py')
# Fallback: try common conda path
if not os.path.exists(_vxm_losses_path):
    import sysconfig
    _vxm_losses_path = os.path.join(
        sysconfig.get_path('purelib'), 'voxelmorph', 'torch', 'losses.py')
_vxm_spec = importlib.util.spec_from_file_location("_vxm_losses", _vxm_losses_path)
_vxm_losses = importlib.util.module_from_spec(_vxm_spec)
_vxm_spec.loader.exec_module(_vxm_losses)
_VxmDice = _vxm_losses.Dice   # expects [B, K, D, H, W] multi-channel soft inputs

# Dong et al. ICCV 2023 — regularize_loss_3d (diffusion, divisor /2)
# Cannot import directly from Dong repo due to transitive dependency on frnn.
# Function is reproduced verbatim from:
#   Medical-Reg-with-Volume-Preserving/metrics/losses.py lines 116-126
#   https://github.com/dddraxxx/Medical-Reg-with-Volume-Preserving
import torch as _torch

def regularize_loss_3d(flow):
    """Dong et al. ICCV 2023 diffusion regularization (divisor /2.0).

    Source: Medical-Reg-with-Volume-Preserving/metrics/losses.py:116-126
    Note: differs from TransMorph Grad3d which uses divisor /3.0.
    """
    dy = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
    dx = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
    dz = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
    d = _torch.mean(dx**2, dim=(1, 2, 3, 4)) + _torch.mean(dy**2, dim=(1, 2, 3, 4)) + _torch.mean(dz**2, dim=(1, 2, 3, 4))
    return d.sum() / 2.0


# ---------------------------------------------------------------------------
#  Jacobian determinant utility (PyTorch, differentiable)
# ---------------------------------------------------------------------------

def jacobian_det_3d(flow, return_det=False):
    """Compute Jacobian determinant of a 3D displacement field.

    Uses the finite-difference convention from Dong et al. (ICCV 2023):
    identity is added directly to displacement differences, avoiding
    allocation of a full identity grid.

    Reference: Dong et al., "Preserving Tumor Volumes for Unsupervised
    Medical Image Registration," ICCV 2023. Code: metrics/losses.py.

    Args:
        flow: displacement field [B, 3, D, H, W]
        return_det: if True, return raw determinant map

    Returns:
        Jacobian determinant [B, D-1, H-1, W-1]
    """
    dx = (flow[:, :, 1:, 1:, 1:] - flow[:, :, :-1, 1:, 1:]
          + flow.new_tensor([1., 0., 0.]).view(1, 3, 1, 1, 1))
    dy = (flow[:, :, 1:, 1:, 1:] - flow[:, :, 1:, :-1, 1:]
          + flow.new_tensor([0., 1., 0.]).view(1, 3, 1, 1, 1))
    dz = (flow[:, :, 1:, 1:, 1:] - flow[:, :, 1:, 1:, :-1]
          + flow.new_tensor([0., 0., 1.]).view(1, 3, 1, 1, 1))

    jac = torch.stack([dx, dy, dz], dim=1).permute(0, 3, 4, 5, 1, 2)
    det = torch.det(jac)

    return det


# ---------------------------------------------------------------------------
#  3b. Bending Energy (Rueckert et al. 1999; TransMorph Eq. 15)
# ---------------------------------------------------------------------------

class BendingEnergy(nn.Module):
    """Bending energy regularizer on displacement field.

    Penalizes curvature (second-order derivatives) rather than gradients.
    More permissive of large smooth deformations (e.g., tumor mass effect).

    Reference: Rueckert et al. 1999; TransMorph paper Eq. 15.
    """

    def _gradient_dx(self, fv):
        return (fv[:, :, 2:, 1:-1, 1:-1] - fv[:, :, :-2, 1:-1, 1:-1]) / 2

    def _gradient_dy(self, fv):
        return (fv[:, :, 1:-1, 2:, 1:-1] - fv[:, :, 1:-1, :-2, 1:-1]) / 2

    def _gradient_dz(self, fv):
        return (fv[:, :, 1:-1, 1:-1, 2:] - fv[:, :, 1:-1, 1:-1, :-2]) / 2

    def _gradient_txyz(self, t, fn):
        return torch.stack([fn(t[:, i:i+1]) for i in range(3)], dim=1).squeeze(2)

    def forward(self, flow, _=None):
        """
        Args:
            flow: displacement field [B, 3, D, H, W]
            _: unused (for API compatibility with Grad3d)
        """
        dTdx = self._gradient_txyz(flow, self._gradient_dx)
        dTdy = self._gradient_txyz(flow, self._gradient_dy)
        dTdz = self._gradient_txyz(flow, self._gradient_dz)

        dTdxx = self._gradient_txyz(dTdx, self._gradient_dx)
        dTdyy = self._gradient_txyz(dTdy, self._gradient_dy)
        dTdzz = self._gradient_txyz(dTdz, self._gradient_dz)
        dTdxy = self._gradient_txyz(dTdx, self._gradient_dy)
        dTdyz = self._gradient_txyz(dTdy, self._gradient_dz)
        dTdxz = self._gradient_txyz(dTdx, self._gradient_dz)

        return torch.mean(
            dTdxx**2 + dTdyy**2 + dTdzz**2
            + 2 * dTdxy**2 + 2 * dTdxz**2 + 2 * dTdyz**2
        )


# ---------------------------------------------------------------------------
#  3d. Anti-folding (VoxelMorph-diff, SYMNet)
# ---------------------------------------------------------------------------

class AntifoldLoss(nn.Module):
    """Penalize negative Jacobian determinants (folding).

    Follows SYMNet (Mok & Chung, CVPR 2020) formulation exactly:
    mean(relu(-det(J))). Default weight in SYMNet is 100.0.

    Reference: Mok & Chung, "Fast Symmetric Diffeomorphic Image Registration
    with Convolutional Neural Networks," CVPR 2020.
    """

    def forward(self, flow):
        """
        Args:
            flow: displacement field [B, 3, D, H, W]

        Returns:
            mean(relu(-det(J)))
        """
        jac_det = jacobian_det_3d(flow)
        neg_det = F.relu(-jac_det)
        return torch.mean(neg_det)


# ---------------------------------------------------------------------------
#  Mask smoothing utilities
# ---------------------------------------------------------------------------

# NOTE: Dong et al. ICCV 2023's "soft tumor mask" (STM, Eq. 9) is derived from
# a Stage-1 similarity-only network's Jacobian, not from the GT tumor segmentation.
# Since we operate in the supervised setting (BraTS25 nnU-Net masks available),
# we substitute Brett 2001's Gaussian-smoothed binary mask as the soft weighting.
# See FINDINGS.md sections C1, C2.
def smooth_tumor_mask(mask, method='gaussian', sigma_mm=4.0):
    """Convert binary tumor mask to soft mask for cost function masking.

    Args:
        mask: binary tumor mask [B, 1, D, H, W] (1=tumor, 0=healthy)
        method: 'none' (binary, no smoothing) or 'gaussian' (Brett 2001).
        sigma_mm: Gaussian FWHM in mm. Default 4.0 (mets-tuned: σ ≈ 1.7
                  voxels at 1 mm iso, FWHM = 2.355 * σ). This is a
                  deliberate departure from Brett 2001's 8 mm value, which
                  was tuned for cm-scale stroke lesions; metastases are
                  typically 5–15 mm and an 8 mm halo dilates well past the
                  lesion. The Brett 2001 *principle* (Gaussian-soft cost-
                  function masking) is what we keep; the kernel size is
                  mets-tuned. See plan note "Mets FWHM rationale" and
                  FINDINGS.md C1/C2.

    Returns:
        soft mask [B, 1, D, H, W] with values in [0, 1]
        (1 = definite tumor, 0 = definite healthy, gradual at boundaries)

    References:
        Brett et al., "Spatial Normalization of Brain Images with Focal
        Lesions Using Cost Function Masking," NeuroImage 2001.
    """
    if method == 'none':
        return mask.float()

    if method == 'gaussian':
        # Brett et al. 2001: Gaussian smoothing of binary mask
        # Default 4 mm FWHM at 1 mm iso -> sigma = 4 / 2.355 ≈ 1.7 voxels (mets-tuned; see docstring)
        sigma = sigma_mm / 2.355
        kernel_size = int(6 * sigma + 1) | 1  # ensure odd
        # 3D Gaussian kernel (separable)
        ax = torch.arange(kernel_size, dtype=mask.dtype, device=mask.device) - kernel_size // 2
        gauss_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        # Apply separable 3D convolution
        pad = kernel_size // 2
        smooth = mask.float()
        for dim in range(2, 5):  # D, H, W
            shape = [1] * 5
            shape[dim] = kernel_size
            kernel = gauss_1d.view(shape).expand(1, 1, -1 if dim == 2 else 1,
                                                  -1 if dim == 3 else 1,
                                                  -1 if dim == 4 else 1)
            # Reshape for conv3d: kernel [out, in, kD, kH, kW]
            k3d = torch.ones(1, 1, 1, 1, 1, dtype=mask.dtype, device=mask.device)
            if dim == 2:
                k3d = gauss_1d.view(1, 1, -1, 1, 1)
            elif dim == 3:
                k3d = gauss_1d.view(1, 1, 1, -1, 1)
            else:
                k3d = gauss_1d.view(1, 1, 1, 1, -1)
            smooth = F.conv3d(smooth, k3d, padding=[pad if d == dim else 0
                                                     for d in range(2, 5)])
        return smooth.clamp(0, 1)

    raise ValueError(f"Unknown mask smoothing method: {method!r}. Must be 'none' or 'gaussian'.")


# ---------------------------------------------------------------------------
#  3a. Soft-weighted similarity (Dong et al. ICCV 2023)
# ---------------------------------------------------------------------------

class MaskedNCC(nn.Module):
    """Soft-weighted global Pearson correlation for similarity loss.

    When mask is provided (1=tumor, 0=healthy), uses soft_weight = 1 - mask
    to down-weight tumor voxels in the covariance computation while keeping
    them in the global mean/variance.

    Reference: Dong et al., "Preserving Tumor Volumes for Unsupervised
    Medical Image Registration," ICCV 2023, Eq. 11-12.
    Code: metrics/losses.py sim_loss() lines 36-68.
    """

    def __init__(self, win=None):
        super().__init__()
        # win accepted for API compatibility but unused (global correlation)

    def forward(self, y_true, y_pred, mask=None):
        """
        Args:
            y_true: fixed image [B, 1, D, H, W]
            y_pred: moved image [B, 1, D, H, W]
            mask: binary tumor mask [B, 1, D, H, W] (1=tumor, 0=healthy).
                  If None, computes unweighted global Pearson correlation.

        Returns:
            scalar loss in [0, 4] (higher = worse alignment)
        """
        I = y_true.flatten(start_dim=1)   # [B, N]
        J = y_pred.flatten(start_dim=1)   # [B, N]

        mu_I = I.mean(dim=1, keepdim=True)
        mu_J = J.mean(dim=1, keepdim=True)
        var_I = ((I - mu_I) ** 2).mean(dim=1, keepdim=True)
        var_J = ((J - mu_J) ** 2).mean(dim=1, keepdim=True)

        if mask is not None:
            sw = (1.0 - mask).flatten(start_dim=1)   # [B, N]
        else:
            sw = I.new_ones((1, 1))

        # Weighted covariance (Dong et al. sim_loss lines 30-31)
        cov = ((I - mu_I) * (J - mu_J) * sw).mean(dim=1, keepdim=True)

        eps = 1e-6
        pearson = cov / torch.sqrt((var_I + eps) * (var_J + eps))
        # Normalize by mean soft weight (Dong et al. sim_loss line 32)
        pearson = pearson / sw.mean(dim=1, keepdim=True)

        return (1 - pearson).mean() * 4


# ---------------------------------------------------------------------------
#  3c. Dice auxiliary loss (TransMorph Eq. 16-17, VoxelMorph Sec. 3)
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Multi-label soft Dice loss for anatomical structure alignment.

    Expects ALREADY one-hot encoded and warped segmentations as input:
      - y_pred: [B, K, D, H, W] soft (bilinear-warped) atlas segmentation
      - y_true: [B, K, D, H, W] hard (one-hot) subject segmentation

    The one-hot encoding and warping must be done in the training loop
    BEFORE calling this loss, following TransMorph Sec. 3.2:
      "s_m ∘ φ is computed by warping the K-channel s_m with φ using
       linear interpolation so that the gradients of L_seg can be
       backpropagated into the network."

    Internally uses voxelmorph.torch.losses.Dice (Balakrishnan et al. 2019).

    References:
    - TransMorph (Chen et al. 2022) Eq. 16-17, Sec. 3.2
    - VoxelMorph (Balakrishnan et al. 2019) Sec. 3
    """

    def __init__(self):
        super().__init__()
        self._dice = _VxmDice()

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: warped one-hot atlas seg [B, K, D, H, W] (soft, from bilinear warp)
            y_true: one-hot subject seg [B, K, D, H, W] (hard binary)

        Returns:
            1 - mean(Dice) across K channels (scalar, differentiable)
        """
        # vxm.Dice.loss returns negative Dice (for minimization)
        # we return 1 + (-dice) = 1 - dice to match standard convention
        return 1.0 + self._dice.loss(y_true, y_pred)


# ---------------------------------------------------------------------------
#  3e. Volume Preservation in tumor regions (Dong et al. ICCV 2023)
# ---------------------------------------------------------------------------

class VolumePreservationLoss(nn.Module):
    """Organ-ratio-normalized, symmetric volume-preserving loss.

    Penalizes Jacobian determinant deviations from 1 inside the tumor
    region, normalized by the overall organ (brain) volume change ratio
    so that global size differences are not penalized.

    Uses symmetric penalty max(|J*r|, 1/|J*r|) with clamping for
    stability.

    Reference: Dong et al., "Preserving Tumor Volumes for Unsupervised
    Medical Image Registration," ICCV 2023, Eq. 10.
    Code: train_simple.py lines 388-430.
    """

    def forward(self, flow, tumor_mask, organ_mask_moving, organ_mask_warped):
        """
        Args:
            flow: displacement field [B, 3, D, H, W]
            tumor_mask: binary tumor mask [B, 1, D, H, W] (1=tumor)
            organ_mask_moving: binary brain mask of moving image [B, 1, D, H, W]
            organ_mask_warped: binary brain mask after warping [B, 1, D, H, W]

        Returns:
            scalar loss (0.0 if no tumor voxels present)
        """
        bs = flow.shape[0]

        # Organ volume ratio per batch element (Dong et al. train_simple.py:401)
        moving_vol = (organ_mask_moving > 0.5).reshape(bs, -1).sum(1).float()
        warped_vol = (organ_mask_warped > 0.5).reshape(bs, -1).sum(1).float()
        ratio = (warped_vol / moving_vol.clamp(min=1)).reshape(-1, 1, 1, 1)

        # Jacobian determinant scaled by organ ratio, clamped (train_simple.py:414)
        det = jacobian_det_3d(flow, return_det=True)  # [B, D-1, H-1, W-1]
        det_scaled = (det.abs() * ratio).clamp(min=1.0 / 3, max=3.0)

        # Interpolate tumor mask to match Jacobian spatial size (train_simple.py:415)
        vp_mask = F.interpolate(
            tumor_mask.float(), size=det_scaled.shape[-3:],
            mode='trilinear', align_corners=False
        ).squeeze(1)  # [B, D-1, H-1, W-1]

        mask_sum = vp_mask.sum()
        if mask_sum == 0:
            return torch.tensor(0.0, device=flow.device, requires_grad=True)

        # Symmetric penalty: max(d, 1/d) (train_simple.py:417)
        adet = torch.where(det_scaled > 1, det_scaled, 1.0 / det_scaled)

        # Masked mean (train_simple.py:420)
        return (adet * vp_mask).sum() / mask_sum


# ---------------------------------------------------------------------------
#  STSR metric (Dong et al. ICCV 2023) — evaluation only, not a loss
# ---------------------------------------------------------------------------

def compute_stsr(warped_tumor_vol, orig_tumor_vol,
                 warped_organ_vol, orig_organ_vol):
    """Symmetric Tumor-to-organ Size Ratio (STSR).

    Measures how well tumor volume is preserved relative to the organ
    after registration. STSR = 1.0 means perfect preservation.

    Formula: STSR = max(TSR_w / TSR_o, TSR_o / TSR_w)²
    where TSR = |tumor| / |organ|.

    Reference: Dong et al., "Preserving Tumor Volumes for Unsupervised
    Medical Image Registration," ICCV 2023.
    Code: eval/eval.py lines 275-285.

    Args:
        warped_tumor_vol: scalar or tensor, warped tumor voxel count
        orig_tumor_vol: scalar or tensor, original tumor voxel count
        warped_organ_vol: scalar or tensor, warped organ voxel count
        orig_organ_vol: scalar or tensor, original organ voxel count

    Returns:
        STSR value (>=1.0, lower is better, 1.0 = perfect)
    """
    eps = 1e-8
    tsr_orig = orig_tumor_vol / (orig_organ_vol + eps)
    tsr_warped = warped_tumor_vol / (warped_organ_vol + eps)
    ratio = tsr_warped / (tsr_orig + eps)
    stsr = torch.where(ratio > 1, ratio, 1.0 / ratio) ** 2
    return stsr


def eval_stsr(warped_tumor_mask, orig_tumor_mask,
              warped_organ_mask, orig_organ_mask):
    """Compute STSR from binary mask tensors.

    Convenience wrapper around compute_stsr that takes mask tensors
    and counts voxels. Used by all evaluation pipelines (TransMorph,
    VoxelMorph, NODEO).

    Reference: Dong et al. ICCV 2023, eval/eval.py lines 275-285.

    Args:
        warped_tumor_mask: [B, 1, D, H, W] or [B, D, H, W] — warped tumor mask
        orig_tumor_mask:   same shape — original tumor mask
        warped_organ_mask: same shape — warped brain mask
        orig_organ_mask:   same shape — original brain mask

    Returns:
        float STSR value (NaN if no tumor voxels)
    """
    wt = (warped_tumor_mask > 0.5).float().sum()
    ot = (orig_tumor_mask > 0.5).float().sum()
    wo = (warped_organ_mask > 0.5).float().sum()
    oo = (orig_organ_mask > 0.5).float().sum()
    if ot == 0:
        return float('nan')
    return float(compute_stsr(wt, ot, wo, oo).item())


# ---------------------------------------------------------------------------
#  TVCF metric — longitudinal s2s tumor-volume-change fidelity
# ---------------------------------------------------------------------------
#  Algebraic form: symmetric squared ratio (Dong et al. ICCV 2023, STSR).
#  Conceptual basis: predicted-vs-GT tumor volume change in longitudinal
#  registration (Sarkar et al., IJROBP 83(3):1038-1046, 2011) — but using
#  mask warping rather than integrated Jacobian, on real follow-up masks
#  rather than synthetic GT.

def eval_tvcf(warped_tumor_mask, mov_tumor_mask, fix_tumor_mask):
    """Tumor Volume Change Fidelity for longitudinal s2s registration.

    Compares the model-predicted volume change ratio against the
    ground-truth ratio observed between two same-patient timepoints:

        VCR_true = |T_fix| / |T_mov|     ground-truth volume change
        VCR_pred = |T_warp| / |T_mov|    model-predicted change
        TVCF     = max(VCR_pred/VCR_true, VCR_true/VCR_pred) ** 2
        LVCR     = log(VCR_pred / VCR_true)

    TVCF >= 1.0; lower is better; 1.0 = predicted change matches truth.
    LVCR sign distinguishes under-deformation (<0) from over-deformation
    (>0).

    Caller MUST pre-filter pairs with tvcf_pair_passes_filter to exclude
    topology-changing events (lesion appearance/resection), which a
    diffeomorphic deformation cannot represent.

    Args:
        warped_tumor_mask: [B,1,D,H,W] or [B,D,H,W] — warped moving tumor
        mov_tumor_mask:    same shape — original moving tumor
        fix_tumor_mask:    same shape — target/fixed tumor mask

    Returns:
        (tvcf, lvcr) tuple of floats. (NaN, NaN) if any input is empty.
    """
    eps = 1e-8
    wt = float((warped_tumor_mask > 0.5).float().sum().item())
    mt = float((mov_tumor_mask > 0.5).float().sum().item())
    ft = float((fix_tumor_mask > 0.5).float().sum().item())
    if mt == 0 or ft == 0 or wt == 0:
        return float('nan'), float('nan')
    vcr_true = ft / mt
    vcr_pred = wt / mt
    ratio = vcr_pred / (vcr_true + eps)
    if ratio <= 0:
        return float('nan'), float('nan')
    tvcf = max(ratio, 1.0 / ratio) ** 2
    lvcr = math.log(ratio)
    return float(tvcf), float(lvcr)


def tvcf_pair_passes_filter(mov_mask_np, fix_mask_np,
                            v_min=100, ratio_lo=0.1, ratio_hi=10.0,
                            centroid_max_vox=20.0):
    """Topology-preservation filter for TVCF eligibility.

    Excludes pairs where TVCF would not measure registration quality
    because tumor topology changed between scans (lesion appearance,
    resection, or different lesions sampled at the two timepoints).
    Defaults assume MNI152 1 mm isotropic spacing (centroid_max_vox = mm).

    Args:
        mov_mask_np: 3D ndarray — moving tumor mask
        fix_mask_np: 3D ndarray — fixed tumor mask
        v_min: minimum voxel count for either mask
        ratio_lo, ratio_hi: |T_fix|/|T_mov| must lie in [lo, hi]
        centroid_max_vox: max centroid Euclidean distance (vox)

    Returns:
        (passes, reason). reason='' on pass else a short tag.
    """
    mov = mov_mask_np > 0
    fix = fix_mask_np > 0
    mv = int(mov.sum())
    fv = int(fix.sum())
    if mv < v_min or fv < v_min:
        return False, f'volume_below_v_min(mov={mv},fix={fv})'
    ratio = fv / mv
    if ratio < ratio_lo or ratio > ratio_hi:
        return False, f'ratio_outside_bounds({ratio:.3f})'
    mc = np.asarray(np.nonzero(mov), dtype=np.float64).mean(axis=1)
    fc = np.asarray(np.nonzero(fix), dtype=np.float64).mean(axis=1)
    d = float(np.linalg.norm(mc - fc))
    if d > centroid_max_vox:
        return False, f'centroid_distance({d:.2f}vox)'
    return True, ''


# ---------------------------------------------------------------------------
#  Distribution summary helper for eval reporting
# ---------------------------------------------------------------------------
#  Five-number summary + Tukey 1.5*IQR outlier counts (Tukey 1977,
#  Exploratory Data Analysis, Addison-Wesley).

def distribution_summary(values):
    """Five-number summary + IQR outlier counts for scalar samples.

    Returns dict with: n, mean, std, min, q1, median, q3, max, iqr,
    outlier_low, outlier_high, outlier_low_thr, outlier_high_thr.
    Outlier rule: x < Q1 - 1.5*IQR or x > Q3 + 1.5*IQR
    (Tukey 1977, Exploratory Data Analysis).

    Args:
        values: iterable of scalars (None / NaN are dropped).

    Returns:
        dict, or None if no finite values.
    """
    arr = np.asarray(
        [v for v in values
         if v is not None and not (isinstance(v, float) and math.isnan(v))],
        dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    iqr = float(q3 - q1)
    lo_thr = float(q1 - 1.5 * iqr)
    hi_thr = float(q3 + 1.5 * iqr)
    inlier_mask = (arr >= lo_thr) & (arr <= hi_thr)
    inliers = arr[inlier_mask]
    return {
        'n': int(arr.size),
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'min': float(arr.min()),
        'q1': float(q1),
        'median': float(med),
        'q3': float(q3),
        'max': float(arr.max()),
        'iqr': iqr,
        'outlier_low': int((arr < lo_thr).sum()),
        'outlier_high': int((arr > hi_thr).sum()),
        'outlier_low_thr': lo_thr,
        'outlier_high_thr': hi_thr,
        'n_trimmed': int(inliers.size),
        'mean_trimmed': float(inliers.mean()) if inliers.size else float('nan'),
        'std_trimmed':  float(inliers.std())  if inliers.size else float('nan'),
    }


def format_distribution(stats, label, fmt='.4f', note=None):
    """Render a distribution_summary dict as a multi-line eval-print block.

    Args:
        stats: dict returned by distribution_summary, or None
        label: header label ("Dice", "STSR", ...)
        fmt: format spec for floats (default '.4f')
        note: optional trailing line (e.g., "1.0 = perfect")

    Returns:
        multi-line string (no leading newline).
    """
    if stats is None:
        return f'  {label}: no data'
    n_lo, n_hi = stats['outlier_low'], stats['outlier_high']
    has_out = (n_lo + n_hi) > 0
    header = f'  {label} (n={stats["n"]})'
    if note:
        header += f' — {note}'
    central = (f'    median {stats["median"]:{fmt}}   '
               f'mean {stats["mean"]:{fmt}} +/- {stats["std"]:{fmt}}')
    if has_out:
        central += (f'   trimmed(n={stats["n_trimmed"]}) '
                    f'{stats["mean_trimmed"]:{fmt}} '
                    f'+/- {stats["std_trimmed"]:{fmt}}')
    spread = (f'    Q1/Q3 {stats["q1"]:{fmt}} / {stats["q3"]:{fmt}}   '
              f'range [{stats["min"]:{fmt}}, {stats["max"]:{fmt}}]')
    lines = [header, central, spread]
    if has_out:
        bits = []
        if n_lo > 0:
            bits.append(f'low={n_lo} (<{stats["outlier_low_thr"]:{fmt}})')
        if n_hi > 0:
            bits.append(f'high={n_hi} (>{stats["outlier_high_thr"]:{fmt}})')
        lines.append(f'    outliers (Tukey 1.5*IQR): ' + ', '.join(bits))
    return '\n'.join(lines)
