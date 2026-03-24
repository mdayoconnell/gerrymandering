# Project: Gerrymandering in Python
# Module: maps/init_squareland.py

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def initialize_map(
    n: int = 25,
    num_cities: int = 2,
    num_farms: int = 4,
    city_pop: float = 1000,
    farm_pop: float = 300,
    city_std: float = 2,
    farm_std: float = 5,
    num_districts: int = 5,
):
    """
    Initialize the Squareland population, swing, and district map.

    Returns:
        pops: (n, n) float array of population density (normalized to max=1)
        swing: (n, n) float array of swing values in [-1, 1]
        district_masks: list of length num_districts, each a (n, n) boolean mask
        centers: (num_districts, 2) int array of (row, col) district centers
    """
    # Coordinate grids (row=i, col=j)
    I, J = np.indices((n, n))
    pops = np.zeros((n, n), dtype=float)
    city_centers = []
    farm_centers = []

    # Add Gaussian "city" centers: big magnitude, small std
    for _ in range(num_cities):
        cx, cy = np.random.randint(0, n, size=2)
        city_centers.append((cx, cy))
        r2 = (I - cx) ** 2 + (J - cy) ** 2
        pops += city_pop * np.exp(-r2 / (2.0 * (city_std**2)))

    # Add Gaussian "farm" centers: smaller magnitude, larger std
    for _ in range(num_farms):
        fx, fy = np.random.randint(0, n, size=2)
        farm_centers.append((fx, fy))
        r2 = (I - fx) ** 2 + (J - fy) ** 2
        pops += farm_pop * np.exp(-r2 / (2.0 * (farm_std**2)))

    # Normalize population to max=1 (robust to pathological all-zero case)
    max_pop = float(np.max(pops))
    if max_pop > 0:
        pops = pops / max_pop

    # Swing is driven by the same Gaussian centers:
    # cities vote blue (+), farms vote red (-), then normalize to [-1, 1].
    swing = np.zeros((n, n), dtype=float)

    for cx, cy in city_centers:
        r2 = (I - cx) ** 2 + (J - cy) ** 2
        swing += np.exp(-r2 / (2.0 * (city_std**2)))

    for fx, fy in farm_centers:
        r2 = (I - fx) ** 2 + (J - fy) ** 2
        swing -= np.exp(-r2 / (2.0 * (farm_std**2)))

    max_abs = float(np.max(np.abs(swing)))
    if max_abs > 0:
        swing = swing / max_abs

    # Random district centers (row, col)
    centers = np.random.randint(0, n, size=(num_districts, 2), dtype=int)

    # Vectorized Voronoi assignment:
    # dist2 has shape (K, n, n)
    dist2 = (I[None, :, :] - centers[:, 0][:, None, None]) ** 2 + (J[None, :, :] - centers[:, 1][:, None, None]) ** 2
    labels = np.argmin(dist2, axis=0).astype(np.int16)  # (n, n), labels 0..K-1

    # One boolean mask per district
    district_masks = [(labels == k) for k in range(num_districts)]

    return pops, swing, district_masks, centers


def district_indices_from_masks(district_masks) -> np.ndarray:
    """
    Convert list of boolean masks into a single (n, n) integer label map.
    Labels are 0..K-1.
    """
    labels = np.full(district_masks[0].shape, fill_value=-1, dtype=np.int16)
    for k, mask in enumerate(district_masks):
        labels[mask] = k
    return labels


def overlay_district_boundaries(ax, labels: np.ndarray, linewidth: float = 1.5):
    """
    Draw district boundaries as black line segments along the borders between pixels.

    This avoids "black boundary pixels" that would visually erase parts of the underlying map.
    Uses 4-neighborhood adjacencies only (no diagonals).
    """
    nrows, ncols = labels.shape
    segments = []

    # Vertical borders between (i, j) and (i, j+1): x = j+0.5, y spans [i-0.5, i+0.5]
    diff_h = labels[:, :-1] != labels[:, 1:]
    ii, jj = np.nonzero(diff_h)
    for i, j in zip(ii, jj):
        x = j + 0.5
        segments.append([(x, i - 0.5), (x, i + 0.5)])

    # Horizontal borders between (i, j) and (i+1, j): y = i+0.5, x spans [j-0.5, j+0.5]
    diff_v = labels[:-1, :] != labels[1:, :]
    ii, jj = np.nonzero(diff_v)
    for i, j in zip(ii, jj):
        y = i + 0.5
        segments.append([(j - 0.5, y), (j + 0.5, y)])

    if not segments:
        return

    lc = LineCollection(segments, colors="k", linewidths=linewidth, zorder=10)
    ax.add_collection(lc)
    ax.autoscale(False)

def compute_district_stats(pops: np.ndarray, swing: np.ndarray, labels: np.ndarray, num_districts: int):
    """
    Compute per-district population and population-weighted swing.

    Returns:
        district_pop: (K,) total population (sum of pops) per district
        district_vote: (K,) population-weighted vote sum (sum pops*swing) per district
        district_mean_swing: (K,) mean swing = district_vote / district_pop in [-1, 1]
    """
    flat_labels = labels.ravel()
    flat_pops = pops.ravel()
    flat_vote = (pops * swing).ravel()

    district_pop = np.bincount(flat_labels, weights=flat_pops, minlength=num_districts).astype(float)
    district_vote = np.bincount(flat_labels, weights=flat_vote, minlength=num_districts).astype(float)

    district_mean_swing = np.zeros(num_districts, dtype=float)
    nz = district_pop > 0
    district_mean_swing[nz] = district_vote[nz] / district_pop[nz]

    # Clamp just in case of tiny numerical drift
    district_mean_swing = np.clip(district_mean_swing, -1.0, 1.0)

    return district_pop, district_vote, district_mean_swing


def visualize_district_swing(labels: np.ndarray, pops: np.ndarray, swing: np.ndarray, centers: np.ndarray,
                             ax=None, show_colorbar: bool = True, overlay_boundaries: bool = True):
    """
    Show a 'swing Voronoi' view: each district is colored by its population-weighted mean swing.
    """
    if ax is None:
        ax = plt.gca()

    num_districts = int(labels.max()) + 1
    district_pop, district_vote, district_mean = compute_district_stats(pops, swing, labels, num_districts)

    swing_map = district_mean[labels]  # broadcast district mean to pixels
    im = ax.imshow(swing_map, cmap="bwr", vmin=-1, vmax=1)

    # centers are (row, col) so plot(col, row)
    ax.plot(centers[:, 1], centers[:, 0], "ko", markersize=4)
    ax.set_title("District Mean Swing (pop-weighted)")
    ax.set_xticks([])
    ax.set_yticks([])

    if overlay_boundaries:
        overlay_district_boundaries(ax, labels)

    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Return stats for downstream election visuals if desired
    return district_pop, district_mean


# ---- Electoral College Bar Panel ----
def visualize_electoral_college_bar(
    district_pop: np.ndarray,
    district_mean_swing: np.ndarray,
    ax=None,
    total_ev: int = 100,
    show_numbers: bool = True,
    total_population: float | None = None,
):
    """
    Electoral-college style bar:
      - Each district gets electoral votes proportional to population (sums to total_ev).
      - Each segment is colored red/blue based on district_mean_swing sign.
      - Segment width is its EV weight.

    Returns:
        ev: (K,) int electoral votes per district summing to total_ev
        blue_ev: total EV for blue-winning districts
        red_ev: total EV for red-winning districts
    """
    if ax is None:
        ax = plt.gca()

    K = len(district_pop)
    pop = district_pop.astype(float)

    # Allocate EV proportional to population, with largest-remainder rounding to hit exact total_ev.
    weights = pop / pop.sum() if pop.sum() > 0 else np.full(K, 1.0 / K)
    raw = weights * float(total_ev)
    ev = np.floor(raw).astype(int)
    remainder = raw - ev

    missing = int(total_ev - ev.sum())
    if missing > 0:
        order = np.argsort(-remainder)  # descending remainder
        ev[order[:missing]] += 1
    elif missing < 0:
        order = np.argsort(remainder)   # ascending remainder
        take = min(-missing, K)
        ev[order[:take]] -= 1

    # Ensure nonnegative
    ev = np.maximum(ev, 0)

    # Group districts: blues on the left, reds on the right.
    # Order within each group by |mean swing| descending (stronger margins first).
    blue_idx = np.where(district_mean_swing > 0)[0]
    red_idx = np.where(district_mean_swing <= 0)[0]

    blue_order = blue_idx[np.argsort(-np.abs(district_mean_swing[blue_idx]))] if blue_idx.size else blue_idx
    red_order = red_idx[np.argsort(-np.abs(district_mean_swing[red_idx]))] if red_idx.size else red_idx

    left = 0
    # Draw blues first
    for k in blue_order:
        w = int(ev[k])
        if w <= 0:
            continue
        ax.barh([0], [w], left=left, color="#2b6cb0", edgecolor="k", linewidth=0.6, height=0.6)
        if show_numbers and w >= 4:
            ax.text(left + w / 2.0, 0, str(w), ha="center", va="center", fontsize=9, color="white")
        left += w

    blue_right_edge = left  # divider position

    # Draw reds next
    for k in red_order:
        w = int(ev[k])
        if w <= 0:
            continue
        ax.barh([0], [w], left=left, color="#c53030", edgecolor="k", linewidth=0.6, height=0.6)
        if show_numbers and w >= 4:
            ax.text(left + w / 2.0, 0, str(w), ha="center", va="center", fontsize=9, color="white")
        left += w

    # Visual divider between sides (only if both sides exist)
    if blue_idx.size and red_idx.size:
        ax.axvline(blue_right_edge, color="k", linewidth=1.0, alpha=0.7)

    blue_ev = int(ev[district_mean_swing > 0].sum())
    red_ev = int(ev[district_mean_swing <= 0].sum())

    ax.set_xlim(0, total_ev)
    ax.set_yticks([])
    ax.set_xlabel("Electoral Votes")
    title = f"Electoral College Bar (Blue {blue_ev} – Red {red_ev})"
    if total_population is not None:
        title += f"\nTotal Pop: {total_population:,.2f}"
    ax.set_title(title)

    # Clean look
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    return ev, blue_ev, red_ev


def visualize_districts(district_masks, centers, ax=None, show_colorbar: bool = False):
    """
    Visualize districts in distinct colors with black dots at center locations.
    """
    if ax is None:
        ax = plt.gca()

    labels = district_indices_from_masks(district_masks)  # 0..K-1
    im = ax.imshow(labels, cmap="tab20", vmin=0, vmax=len(district_masks) - 1)

    # centers are (row, col) so plot(col, row)
    ax.plot(centers[:, 1], centers[:, 0], "ko", markersize=4)
    ax.set_title("Voronoi District Labels")
    ax.set_xticks([])
    ax.set_yticks([])

    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def launch_map_overview(
    pops: np.ndarray,
    swing: np.ndarray,
    district_masks,
    centers: np.ndarray,
    electoral_total_ev: int = 100,
):
    """
    Render a compact 2x2 overview of the current map state.
    """
    labels = district_indices_from_masks(district_masks)
    num_districts = int(labels.max()) + 1
    district_pop, _, district_mean = compute_district_stats(pops, swing, labels, num_districts)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    population_ax, swing_ax = axes[0]
    outcome_ax, districts_ax = axes[1]

    population_im = population_ax.imshow(pops, cmap="viridis", vmin=0, vmax=1)
    population_ax.set_title("Population (normalized)")
    population_ax.set_xticks([])
    population_ax.set_yticks([])
    overlay_district_boundaries(population_ax, labels)
    plt.colorbar(population_im, ax=population_ax, fraction=0.046, pad=0.04)

    swing_im = swing_ax.imshow(swing, cmap="bwr", vmin=-1, vmax=1)
    swing_ax.set_title("Swing (-1 red, +1 blue)")
    swing_ax.set_xticks([])
    swing_ax.set_yticks([])
    overlay_district_boundaries(swing_ax, labels)
    plt.colorbar(swing_im, ax=swing_ax, fraction=0.046, pad=0.04)

    visualize_electoral_college_bar(
        district_pop=district_pop,
        district_mean_swing=district_mean,
        ax=outcome_ax,
        total_ev=electoral_total_ev,
        show_numbers=True,
        total_population=float(np.sum(pops)),
    )

    visualize_districts(district_masks, centers, ax=districts_ax, show_colorbar=False)
    districts_ax.set_title("District Labels")

    return fig, axes


def main(
    show_population: bool = True,
    show_swing: bool = True,
    show_voronoi: bool = False,
    show_district_swing: bool = True,
    show_electoral_college: bool = True,
    overlay_lines_on_population: bool = False,
    overlay_lines_on_swing: bool = False,
    overlay_lines_on_district_swing: bool = True,
    electoral_total_ev: int = 100,
    show_colorbars: bool = True,
    n: int = 100,
    num_cities: int = 6,
    num_farms: int = 20,
    city_pop: float = 10000,
    farm_pop: float = 700,
    city_std: float = 18,
    farm_std: float = 24,
    num_districts: int = 10,            # Values matter in parser arguments in main, not here
):
    pops, swing, district_masks, centers = initialize_map(
        n=n,
        num_cities=num_cities,
        num_farms=num_farms,
        city_pop=city_pop,
        farm_pop=farm_pop,
        city_std=city_std,
        farm_std=farm_std,
        num_districts=num_districts,
    )

    labels = district_indices_from_masks(district_masks)

    panels = []
    if show_population:
        panels.append("population")
    if show_swing:
        panels.append("swing")
    if show_district_swing:
        panels.append("district_swing")
    if show_electoral_college:
        panels.append("electoral_college")
    if show_voronoi:
        panels.append("voronoi")

    # Robust behavior: if user turned everything off, default to all panels
    if not panels:
        panels = ["population", "swing", "district_swing", "electoral_college"]

    fig, axs = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axs = [axs]

    for ax, name in zip(axs, panels):
        if name == "population":
            im = ax.imshow(pops, cmap="viridis", vmin=0, vmax=1)
            ax.set_title("Population (normalized)")
            ax.set_xticks([])
            ax.set_yticks([])
            if overlay_lines_on_population:
                overlay_district_boundaries(ax, labels)
            if show_colorbars:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        elif name == "swing":
            im = ax.imshow(swing, cmap="bwr", vmin=-1, vmax=1)
            ax.set_title("Swing (-1 red, +1 blue)")
            ax.set_xticks([])
            ax.set_yticks([])
            if overlay_lines_on_swing:
                overlay_district_boundaries(ax, labels)
            if show_colorbars:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        elif name == "district_swing":
            # This panel shows district-level (population-weighted) swing, broadcast onto pixels
            district_pop, district_mean = visualize_district_swing(
                labels, pops, swing, centers, ax=ax,
                show_colorbar=show_colorbars,
                overlay_boundaries=overlay_lines_on_district_swing,
            )

        elif name == "electoral_college":
            # Need district stats; compute once from current labels/pops/swing
            num_districts_here = int(labels.max()) + 1
            district_pop, _, district_mean = compute_district_stats(pops, swing, labels, num_districts_here)
            visualize_electoral_college_bar(
                district_pop=district_pop,
                district_mean_swing=district_mean,
                ax=ax,
                total_ev=electoral_total_ev,
                show_numbers=True,
                total_population=float(np.sum(pops)),
            )

        elif name == "voronoi":
            visualize_districts(district_masks, centers, ax=ax, show_colorbar=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Squareland map initialization debug plots.")

    # What to show
    parser.add_argument("--population", action="store_true", help="Show population heatmap panel.")
    parser.add_argument("--swing", action="store_true", help="Show swing heatmap panel.")
    parser.add_argument("--voronoi", action="store_true", help="Show Voronoi district label panel.")
    parser.add_argument("--district-swing", action="store_true", help="Show district-level (pop-weighted) swing panel.")
    parser.add_argument("--electoral-college", action="store_true", help="Show electoral-college bar panel.")
    parser.add_argument("--lines-on-population", action="store_true",
                    help="Overlay district boundaries on population panel.") 

    # Overlays   
    parser.add_argument("--lines-on-swing", action="store_true", help="Overlay district boundaries on swing panel.")
    parser.add_argument("--lines-on-district-swing", action="store_true", help="Overlay district boundaries on district-swing panel.")

    # Display options
    parser.add_argument("--no-colorbars", action="store_true", help="Disable colorbars on plots.")
    parser.add_argument("--electoral-total-ev", type=int, default=100, help="Total EV to allocate in electoral-college bar.")

    # Map params (optional for quick experimentation)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--num-cities", type=int, default=10)
    parser.add_argument("--num-farms", type=int, default=10)
    parser.add_argument("--city-pop", type=float, default=1000)
    parser.add_argument("--farm-pop", type=float, default=1000)
    parser.add_argument("--city-std", type=float, default=10)
    parser.add_argument("--farm-std", type=float, default=10)
    parser.add_argument("--num-districts", type=int, default=10)

    args = parser.parse_args()

    # If the user doesn't specify any of the panels, default to all
    any_panels = args.population or args.swing or args.voronoi or args.district_swing or args.electoral_college

    main(
        show_population=args.population or (not any_panels),
        show_swing=args.swing or (not any_panels),
        show_voronoi=args.voronoi,
        show_district_swing=args.district_swing or (not any_panels),
        show_electoral_college=args.electoral_college or (not any_panels),
        overlay_lines_on_population=args.lines_on_population,
        overlay_lines_on_swing=args.lines_on_swing,
        overlay_lines_on_district_swing=args.lines_on_district_swing or args.district_swing,
        show_colorbars=not args.no_colorbars,
        electoral_total_ev=args.electoral_total_ev,
        n=args.n,
        num_cities=args.num_cities,
        num_farms=args.num_farms,
        city_pop=args.city_pop,
        farm_pop=args.farm_pop,
        city_std=args.city_std,
        farm_std=args.farm_std,
        num_districts=args.num_districts,
    )
