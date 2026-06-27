import os, glob
import datetime  # for date-delta debug helper (P2_8 smoke test)
import torch, sys
from collections import defaultdict
from torch.utils.data import Dataset
from .data_utils import pkload, npzload
import numpy as np


class IXIBrainDataset(Dataset):
    """Original IXI atlas-based dataset loading .pkl files."""
    def __init__(self, data_path, atlas_path, transforms):
        self.paths = data_path
        self.atlas_path = atlas_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class IXIBrainInferDataset(Dataset):
    """Original IXI inference dataset loading .pkl files with segmentations."""
    def __init__(self, data_path, atlas_path, transforms):
        self.atlas_path = atlas_path
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class NpzAtlasDataset(Dataset):
    """Atlas-based training dataset loading .npz files.

    Returns (moving, fixed, [tumor_mask], [seg], [atlas_seg]) tensors.
    When reverse=False (atlas-to-scan): moving=atlas, fixed=subject.
    When reverse=True (scan-to-atlas): moving=subject, fixed=atlas.
    Tumor mask always comes from the subject.
    """
    def __init__(self, npz_paths, atlas_vol, transforms=None,
                 load_tumor_mask=False, load_seg=False, atlas_seg=None,
                 reverse=False):
        """
        Args:
            npz_paths: list of .npz file paths (subjects)
            atlas_vol: numpy array [H, W, D], normalized to [0, 1]
            transforms: torchvision Compose of trans.* transforms
            load_tumor_mask: if True, also returns tumor_mask from .npz
            load_seg: if True, also returns subject seg from .npz
            atlas_seg: numpy array [H, W, D] atlas segmentation (for Dice loss)
            reverse: if True, swap moving/fixed (scan-to-atlas mode)
        """
        self.paths = npz_paths
        self.atlas = atlas_vol.astype(np.float32)
        self.transforms = transforms
        self.load_tumor_mask = load_tumor_mask
        self.load_seg = load_seg
        self.atlas_seg = atlas_seg
        self.reverse = reverse

    def __getitem__(self, index):
        path = self.paths[index]
        d = np.load(path)
        subject = d['vol'].astype(np.float32)
        if subject.max() > 1:
            subject = subject / subject.max()

        atlas = self.atlas.copy()

        # Add channel dim: [1, H, W, D]
        atlas = atlas[None, ...]
        subject = subject[None, ...]

        if self.transforms:
            atlas, subject = self.transforms([atlas, subject])

        atlas = np.ascontiguousarray(atlas)
        subject = np.ascontiguousarray(subject)

        if self.reverse:
            # scan-to-atlas: moving=subject, fixed=atlas
            result = [torch.from_numpy(subject).float(), torch.from_numpy(atlas).float()]
        else:
            # atlas-to-scan: moving=atlas, fixed=subject
            result = [torch.from_numpy(atlas).float(), torch.from_numpy(subject).float()]

        if self.load_tumor_mask:
            if 'tumor_mask' in d:
                tm = d['tumor_mask'].astype(np.float32)[None, ...]
            else:
                tm = np.zeros_like(subject)
            result.append(torch.from_numpy(tm).float())

        if self.load_seg:
            seg = d['seg'].astype(np.float32)[None, ...]
            result.append(torch.from_numpy(seg).float())
            if self.atlas_seg is not None:
                a_seg = self.atlas_seg.astype(np.float32)[None, ...]
                result.append(torch.from_numpy(a_seg).float())

        return tuple(result)

    def __len__(self):
        return len(self.paths)


class NpzScanPairDataset(Dataset):
    """Scan-to-scan training dataset.

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
    """
    def __init__(self, npz_paths, load_tumor_mask=False, load_seg=False,
                 mode='scan-to-scan-intra', patient_id_fn=None):
        self.load_tumor_mask = load_tumor_mask
        self.load_seg = load_seg
        self.mode = mode
        self.patient_id_fn = patient_id_fn

        # Pre-bin paths by patient
        self._patient_of = {}
        self._by_patient = defaultdict(list)
        self.paths = list(npz_paths)
        if patient_id_fn is not None:
            for i, p in enumerate(self.paths):
                pid = patient_id_fn(os.path.basename(p))
                self._patient_of[i] = pid
                self._by_patient[pid].append(i)

        # CHANGED from sub-plan: in intra mode, drop single-scan patients from
        # the index entirely (oncologist BLOCKER → CONCERN reduction). Avoids
        # silent inter contamination.
        if (mode == 'scan-to-scan-intra'
                and patient_id_fn is not None):
            keep_pids = {pid for pid, idxs in self._by_patient.items()
                         if len(idxs) >= 2}
            n_dropped = sum(1 for pid in self._by_patient
                            if pid not in keep_pids)
            if n_dropped > 0:
                print(f"NpzScanPairDataset: dropped {n_dropped} "
                      f"single-scan patients from intra index")
            kept_paths = [self.paths[i] for i, p in enumerate(self.paths)
                          if self._patient_of[i] in keep_pids]
            # Rebuild paths and per-patient index from kept patients only
            self.paths = kept_paths
            self._patient_of = {}
            self._by_patient = defaultdict(list)
            for i, p in enumerate(self.paths):
                pid = patient_id_fn(os.path.basename(p))
                self._patient_of[i] = pid
                self._by_patient[pid].append(i)
            if len(self.paths) == 0:
                raise RuntimeError(
                    'scan-to-scan-intra requires at least one patient '
                    'with >= 2 scans in the training split.')

    def _pick_fixed_index(self, i):
        n = len(self.paths)
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
        # Backward-compatible random pairing
        j = i
        while j == i:
            j = np.random.randint(n)
        return j

    def _date_of(self, index):
        """Parse date from filename `YG_<id>_<YYYY-MM-DD>.npz`. Smoke-test
        helper for clinical visibility (oncologist CONCERN #1).
        """
        name = os.path.basename(self.paths[index]).replace('.npz', '')
        parts = name.split('_')
        # date is the trailing token in YYYY-MM-DD form
        return datetime.date.fromisoformat(parts[-1])

    def __getitem__(self, index):
        d_mov = np.load(self.paths[index])
        j = self._pick_fixed_index(index)
        d_fix = np.load(self.paths[j])

        mov = d_mov['vol'].astype(np.float32)
        fix = d_fix['vol'].astype(np.float32)
        if mov.max() > 1:
            mov = mov / mov.max()
        if fix.max() > 1:
            fix = fix / fix.max()

        result = [torch.from_numpy(mov[None]).float(),
                  torch.from_numpy(fix[None]).float()]

        if self.load_tumor_mask:
            # CHANGED from sub-plan: load BOTH masks; fail loud if either
            # is missing (oncologist BLOCKER, user decision 2026-04-26).
            tm_mov = d_mov['tumor_mask'].astype(np.float32)[None, ...]
            tm_fix = d_fix['tumor_mask'].astype(np.float32)[None, ...]
            result.append(torch.from_numpy(tm_mov).float())
            result.append(torch.from_numpy(tm_fix).float())

        if self.load_seg:
            mov_seg = d_mov['seg'].astype(np.float32)[None, ...]
            fix_seg = d_fix['seg'].astype(np.float32)[None, ...]
            result.append(torch.from_numpy(mov_seg).float())
            result.append(torch.from_numpy(fix_seg).float())

        return tuple(result)

    def __len__(self):
        return len(self.paths)


class NpzInferDataset(Dataset):
    """Scan-to-scan evaluation dataset loading .npz files with segmentations.

    Returns (moving_vol, fixed_vol, moving_seg, fixed_seg) tensors.
    Only includes files that have both 'vol' and 'seg' keys.
    """
    def __init__(self, npz_paths, transforms=None):
        """
        Args:
            npz_paths: list of .npz file paths (must have 'vol' and 'seg')
            transforms: torchvision Compose of trans.* transforms
        """
        # Filter to only files with seg
        self.paths = []
        for p in npz_paths:
            try:
                with np.load(p) as d:
                    if 'seg' in d:
                        self.paths.append(p)
            except Exception:
                pass
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        d = np.load(path)
        vol = d['vol'].astype(np.float32)
        seg = d['seg'].astype(np.float32)

        if vol.max() > 1:
            vol = vol / vol.max()

        vol = vol[None, ...]
        seg = seg[None, ...]

        if self.transforms:
            vol, seg = self.transforms([vol, seg])

        vol = np.ascontiguousarray(vol)
        seg = np.ascontiguousarray(seg)

        return torch.from_numpy(vol).float(), torch.from_numpy(seg).float()

    def __len__(self):
        return len(self.paths)


class NpzValDataset(Dataset):
    """Validation dataset for atlas-based registration.

    Returns (subject_vol, subject_seg) tensors of shape [1, H, W, D].
    Filters to only files that contain a 'seg' key.
    Used with a multi-worker DataLoader to pipeline disk I/O during validation.
    """
    def __init__(self, npz_paths):
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
        if vol.max() > 1:
            vol = vol / vol.max()
        tm = d['tumor_mask'].astype(np.float32) if 'tumor_mask' in d else np.zeros_like(vol)
        return (torch.from_numpy(vol).float().unsqueeze(0),
                torch.from_numpy(seg).float().unsqueeze(0),
                torch.from_numpy(tm).float().unsqueeze(0))

    def __len__(self):
        return len(self.paths)
