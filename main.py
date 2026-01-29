# Created by Micah
# Date: 7/31/25
# Time: 10:18 PM
# Project: Gerrymandering
# File: main.py

'''

"Redistricting is like an election in reverse. Instead of letting the
voters pick the politicians, the politicians pick the voters!"

- Thomas Hoeffeler

Calculating gerrymandered districts using a monte carlo simulation

'''

from ml_dtypes import uint4
from maps.init_squareland import initialize_map
from model.evaluate import evaluate_flip, check_connectivity
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import gc
from scipy.signal import convolve2d
import matplotlib.animation as animation

def visualize_districts(district_masks, centers):
# ---
# Visualize districts in distinct colors. Dots indicate the center
# ---

    n = district_masks[0].shape[0]
    district_map = np.zeros((n, n), dtype=np.uint8)
    for k, mask in enumerate(district_masks):
        district_map += (mask.astype(np.uint8)) * (k + 1)

    plt.imshow(district_map, cmap='tab20')
    for center in centers:
        plt.plot(center[0], center[1], 'ko')
    #plt.colorbar(label='District ID')
    plt.title('District Map')
    plt.show()
    #plt.close('all')

def compute_edge_neighbourliness(district_masks, sigma=1):
    # ---
    # Compute a matrix indicating how many outgroup neighbors each edge pixel has,
    # weighted by sigma for diagonal neighbors.
    #
    # Parameters:
    #     district_masks: binary masks for all districts (1=in, 0=out)
    #     sigma: weighting factor for diagonal neighbors
    #
    # Returns:
    #     neighbourliness: nxn array with values > 0 only at true edges (excluding map boundary)
    # ---
    n = district_masks[0].shape[0]
    neighbourliness = np.zeros((n, n), dtype=int)

    # Define neighbor kernel with sigma for diagonals, edge kernel to prevent triggering on the side of the map rather than border between districts
    neighbor_kernel = np.array([[sigma, 4, sigma],
                                [4,   0, 4],
                                [sigma, 4, sigma]])

    edge_kernel = np.array([[1, 1, 1],
                            [1, -8, 1],
                            [1, 1, 1]])

    # Only keep neighbourliness where this district has a true edge
    for mask in district_masks:
        edge_map = convolve2d(mask, edge_kernel, mode='same', boundary='fill', fillvalue=0)
        edge_mask = (edge_map < 0)
        outgroup_mask = 1 - mask
        weighted_neighbors = convolve2d(outgroup_mask, neighbor_kernel, mode='same', boundary='fill', fillvalue=0)
        neighbourliness += edge_mask * weighted_neighbors

    return neighbourliness

def select_candidate(neighbourliness):
    # ---
    # Select a random non-zero element from the neighbourliness map.
    # Returns its coordinates and value.
    # ---
    nonzero_coords = np.argwhere(neighbourliness > 0)
    if len(nonzero_coords) == 0:
        return None, None  # No edge candidates
    candidate_idx = np.random.choice(len(nonzero_coords))
    candidate = tuple(nonzero_coords[candidate_idx])
    return candidate, neighbourliness[candidate]


def flip(candidate, candidate_score, district_masks):
    # """
    # Attempt to flip the candidate pixel based on its neighbourliness score.
    # Ensures that the flip does not break district continuity.
    # """
    flip_chance = determine_flip_probability(candidate_score)
    if np.random.rand() > flip_chance:
        print(f"Flip rejected: candidate {candidate} (score {candidate_score:.3f})")
        return district_masks

    print(f"Flip accepted: candidate {candidate} (score {candidate_score:.3f})")

    # Determine which district(s) the candidate borders
    i, j = candidate
    neighbor_ids = []
    for k, mask in enumerate(district_masks):
        neighborhood = mask[max(0, i - 1):i + 2, max(0, j - 1):j + 2]
        if np.any(neighborhood == 1):
            neighbor_ids.append(k)

    # Identify current district and exclude from flip targets
    current_district = next((k for k, mask in enumerate(district_masks) if mask[i, j] == 1), None)
    # Filter out the current district so we only flip into a different one
    neighbor_ids = [k for k in neighbor_ids if k != current_district]
    if not neighbor_ids:
        print("No adjacent district found to flip into.")
        return district_masks

    # Pick one at random for now
    target_district = np.random.choice(neighbor_ids)

    # Identify the current district of the candidate
    current_district = next((k for k, mask in enumerate(district_masks) if mask[i, j] == 1), None)
    if current_district is None:
        print("Candidate not in any district.")
        return district_masks

    # Create a tentative update of the masks
    new_masks = district_masks.copy()
    new_masks[current_district] = new_masks[current_district].copy()
    new_masks[target_district] = new_masks[target_district].copy()
    new_masks[current_district][i, j] = 0
    new_masks[target_district][i, j] = 1

    if not continuity_check_and_evaluate(candidate, current_district, target_district, new_masks):
        return district_masks
    return new_masks
    gc.collect()

def determine_flip_probability(candidate_score, max_score=20.0):
    if candidate_score <= 0:
        return 0.0
    if candidate_score >= max_score:
        return 1.0
    return candidate_score / max_score

def continuity_check_and_evaluate(candidate, current_district, target_district, new_masks):
    if not (check_connectivity(new_masks[current_district]) and check_connectivity(new_masks[target_district])):
        print("Flip would break district continuity. Rejected.")
        return False
    print(f"Flip confirmed: pixel {candidate} moved from district {current_district} to {target_district}.")
    evaluate_flip(candidate, target_district)
    return True


def update(frame):
    print(f"Frame {frame}")
    nonlocal district_masks
    neighbourliness = compute_edge_neighbourliness(district_masks)
    candidate, score = select_candidate(neighbourliness)
    if candidate:
        district_masks = flip(candidate, score, district_masks)
        updated_map = np.zeros_like(district_masks[0], dtype=np.uint8)
        for k, mask in enumerate(district_masks):
            updated_map += mask.astype(np.uint8) * (k + 1)
        im.set_data(updated_map)
    return [im]

def main():
    _, _, district_masks, centers = initialize_map()
    n_flips = 50  # or however many you want

    for _ in range(n_flips):
        neighbourliness = compute_edge_neighbourliness(district_masks)
        candidate, score = select_candidate(neighbourliness)
        if candidate:
            district_masks = flip(candidate, score, district_masks)
            visualize_districts(district_masks, centers)
            plt.show()
            #plt.close()  # clean up memory

if __name__ == '__main__':
    main()
