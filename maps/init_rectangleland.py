from __future__ import annotations

import numpy as np

from maps.init_squareland import launch_map_overview


def initialize_map(
    n: int = 25,
    num_cities: int = 0,
    num_farms: int = 0,
    city_pop: float = 0,
    farm_pop: float = 0,
    city_std: float = 0,
    farm_std: float = 0,
    num_districts: int = 5,
):
    """
    Initialize a simple map with the left and center thirds blue and the right third red.

    Unused parameters are kept for compatibility with the Squareland initializer.
    """
    del num_cities, num_farms, city_pop, farm_pop, city_std, farm_std

    row_indices, col_indices = np.indices((n, n))
    pops = np.ones((n, n), dtype=float)
    swing = np.ones((n, n), dtype=float)
    swing[:, (2 * n) // 3 :] = -1.0

    centers = np.random.randint(0, n, size=(num_districts, 2), dtype=int)
    dist2 = (row_indices[None, :, :] - centers[:, 0][:, None, None]) ** 2 + (
        col_indices[None, :, :] - centers[:, 1][:, None, None]
    ) ** 2
    labels = np.argmin(dist2, axis=0).astype(np.int16)
    district_masks = [(labels == district_id) for district_id in range(num_districts)]

    return pops, swing, district_masks, centers
