# Created by Micah
# Date: 7/31/25
# Time: 10:19 PM
# Project: Gerrymandering
# File: init.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def initialize_map(n=25, num_cities=2, num_farms=4, city_pop=1000, farm_pop=300, city_std=2, farm_std=5, num_districts=5):
    # """
    # Initialize the Squareland population, swing, and district map.
    # Returns:
    #     pops: nxn array of population density
    #     swing: nxn array of swing values from -1 (red) to 1 (blue)
    #     district_masks: list of nxn binary arrays for each district
    #     centers: list of (x, y) tuples for each district center
    # """
    pops = np.zeros((n, n))

    # Add Gaussian "city" centers
    for _ in range(num_cities):
        cx, cy = np.random.randint(0, n, size=2)
        for i in range(n):
            for j in range(n):
                pops[i, j] += city_pop * np.exp(-((i - cx)**2 + (j - cy)**2) / (2 * city_std**2))

    # Add Gaussian "farm" centers
    for _ in range(num_farms):
        fx, fy = np.random.randint(0, n, size=2)
        for i in range(n):
            for j in range(n):
                pops[i, j] += farm_pop * np.exp(-((i - fx)**2 + (j - fy)**2) / (2 * farm_std**2))

    # Normalize population
    pops = pops / np.max(pops)

    # Random swing values between -1 and 1
    swing = np.random.uniform(-1, 1, size=(n, n))

    # Generate random district centers
    centers = np.random.randint(0, n, size=(num_districts, 2))

    # Assign each cell to the nearest district center
    district_indices = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            dists = [np.linalg.norm([i - int(cx), j - int(cy)]) for cx, cy in centers]
            district_indices[i, j] = np.argmin(dists)

    # Create binary masks for each district
    district_masks = [(district_indices == k).astype(bool) for k in range(num_districts)]

    return pops, swing, district_masks, centers


def show_swing(swing):
    # """
    # Display the swing map using a red-blue colorscale.
    # """
    cmap = plt.get_cmap('bwr')  # blue-white-red colormap
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    plt.imshow(swing, cmap=cmap, norm=norm)
    plt.colorbar(label='Swing (-1 = Red, 1 = Blue)')
    plt.title('Swing Map')
    plt.show()