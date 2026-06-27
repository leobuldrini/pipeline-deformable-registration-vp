"""Pair generation utilities for multi-mode registration evaluation.

Used by NODEO, TransMorph, and VoxelMorph batch runners.
"""

import numpy as np
from collections import defaultdict


def extract_patient_id(npz_basename):
    """Extract patient ID from filename: YG_XXXXX_YYYY-MM-DD.npz -> YG_XXXXX."""
    name = npz_basename.replace('.npz', '')
    parts = name.split('_')
    return '_'.join(parts[:2])


def generate_pairs(mode, test_subjects, max_pairs=100, seed=42):
    """Generate (item_a, item_b) pairs based on registration mode.

    For atlas modes, item_b is None (atlas loaded separately).
    For scan-to-scan, both are .npz basenames.
    """
    if mode in ('atlas-to-scan', 'scan-to-atlas'):
        return [(subj, None) for subj in test_subjects]

    patient_scans = defaultdict(list)
    for subj in test_subjects:
        patient_scans[extract_patient_id(subj)].append(subj)

    if mode == 'scan-to-scan-intra':
        pairs = []
        for pid in sorted(patient_scans.keys()):
            scans = sorted(patient_scans[pid])
            if len(scans) < 2:
                continue
            for a in range(len(scans)):
                for b in range(len(scans)):
                    if a != b:
                        pairs.append((scans[a], scans[b]))
        return pairs

    if mode == 'scan-to-scan-inter':
        rng = np.random.RandomState(seed)
        all_scans = list(test_subjects)
        pairs, seen = [], set()
        attempts = 0
        while len(pairs) < max_pairs and attempts < max_pairs * 20:
            attempts += 1
            i, j = rng.choice(len(all_scans), size=2, replace=False)
            a, b = all_scans[i], all_scans[j]
            if extract_patient_id(a) == extract_patient_id(b):
                continue
            if (a, b) in seen:
                continue
            seen.add((a, b))
            pairs.append((a, b))
        return pairs

    raise ValueError(f"Unknown mode: {mode}")
