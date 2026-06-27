"""Shared label definitions for evaluation across all registration models.

The 30-label set follows the TransMorph paper protocol (Chen et al.) and
matches the structures listed in IXI/Anatomical_Structures.md.
"""

EVAL_LABELS_30 = [
    2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    24, 28, 31, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 60, 63,
]

# 25-label subset for CerebrA atlas (FreeSurfer IDs).
# CerebrA lacks: 2/41 (Cerebral WM), 24 (CSF), 31/63 (Choroid Plexus).
EVAL_LABELS_CEREBRA = [
    3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    28, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 60,
]
