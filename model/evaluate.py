# Created by Micah
# Date: 7/31/25
# Time: 10:18 PM
# Project: Gerrymandering
# File: evaluate.py

import numpy as np
from scipy.ndimage import label

def check_connectivity(mask):
    """
    Returns True if the mask represents a single contiguous region.
    Uses 4-connectivity (no diagonal links).
    """
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]])
    labeled_array, num_features = label(mask, structure=structure)
    return num_features == 1

def evaluate_flip(candidate, target_district):
    print(f"Evaluating flip of pixel {candidate} into district {target_district}")
    # TODO: Add loss scoring logic later
