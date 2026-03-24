from __future__ import annotations

from dataclasses import asdict
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

from maps.init_squareland import initialize_map, launch_map_overview
from model.evaluate import check_connectivity
from model.loss import LossWeights, compute_total_loss, district_indices_from_masks
from user_interface.custom_loss_ui import get_loss_weights, launch_loss_weight_sliders
from user_interface.flip_history_ui import FlipHistoryFrame, launch_flip_history_viewer


def compute_edge_neighbourliness(district_masks, sigma: int = 1) -> np.ndarray:
    """
    Score edge pixels by the amount of nearby out-group territory.
    Only pixels on a district boundary receive non-zero scores.
    """

    n = district_masks[0].shape[0]
    neighbourliness = np.zeros((n, n), dtype=int)

    neighbor_kernel = np.array(
        [
            [sigma, 4, sigma],
            [4, 0, 4],
            [sigma, 4, sigma],
        ]
    )
    edge_kernel = np.array(
        [
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1],
        ]
    )

    for mask in district_masks:
        edge_map = convolve2d(mask, edge_kernel, mode="same", boundary="fill", fillvalue=0)
        edge_mask = edge_map < 0
        outgroup_mask = 1 - mask
        weighted_neighbors = convolve2d(
            outgroup_mask,
            neighbor_kernel,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )
        neighbourliness += edge_mask * weighted_neighbors

    return neighbourliness


def select_candidate(neighbourliness: np.ndarray) -> tuple[int, int] | None:
    nonzero_coords = np.argwhere(neighbourliness > 0)
    if len(nonzero_coords) == 0:
        return None
    candidate_idx = np.random.choice(len(nonzero_coords))
    row, col = nonzero_coords[candidate_idx]
    return int(row), int(col)


def get_current_district(candidate: tuple[int, int], district_masks) -> int | None:
    row, col = candidate
    return next((idx for idx, mask in enumerate(district_masks) if mask[row, col]), None)


def get_adjacent_district_ids(candidate: tuple[int, int], district_masks) -> list[int]:
    row, col = candidate
    nrows, ncols = district_masks[0].shape
    adjacent_ids: set[int] = set()

    for next_row, next_col in (
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
    ):
        if not (0 <= next_row < nrows and 0 <= next_col < ncols):
            continue
        for district_id, mask in enumerate(district_masks):
            if mask[next_row, next_col]:
                adjacent_ids.add(district_id)
                break

    current_district = get_current_district(candidate, district_masks)
    if current_district is not None and current_district in adjacent_ids:
        adjacent_ids.remove(current_district)

    return sorted(adjacent_ids)


def build_proposed_masks(
    district_masks,
    candidate: tuple[int, int],
    current_district: int,
    target_district: int,
):
    row, col = candidate
    new_masks = district_masks.copy()
    new_masks[current_district] = new_masks[current_district].copy()
    new_masks[target_district] = new_masks[target_district].copy()
    new_masks[current_district][row, col] = 0
    new_masks[target_district][row, col] = 1
    return new_masks


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def acceptance_probability(loss_delta: float, temperature: float) -> float:
    if loss_delta <= 0:
        return 1.0
    return sigmoid(temperature / loss_delta)


def prompt_yes_no(prompt: str) -> bool:
    while True:
        response = input(prompt).strip().lower()
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")


def prompt_int(prompt: str, default: int) -> int:
    while True:
        response = input(prompt).strip()
        if not response:
            return default
        try:
            value = int(response)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if value <= 0:
            print("Please enter a positive whole number.")
            continue
        return value


def prompt_float(prompt: str, default: float) -> float:
    while True:
        response = input(prompt).strip()
        if not response:
            return default
        try:
            return float(response)
        except ValueError:
            print("Please enter a number.")


def prompt_for_loss_weights() -> LossWeights:
    current_weights = get_loss_weights()

    while True:
        fig, sliders = launch_loss_weight_sliders(initial=current_weights)
        plt.show()
        # Keep widget references alive until the window closes.
        del fig, sliders
        current_weights = get_loss_weights()

        print("\nCurrent loss weights:")
        for key, value in asdict(current_weights).items():
            print(f"  {key}: {value}")

        if prompt_yes_no("Use these weights? [y/n]: "):
            return current_weights


def capture_flip_history_frame(
    district_masks,
    accepted_flips: int,
    attempts: int,
    total_loss: float,
    loss_delta: float,
    candidate: tuple[int, int] | None = None,
    current_district: int | None = None,
    target_district: int | None = None,
) -> FlipHistoryFrame:
    return FlipHistoryFrame(
        labels=district_indices_from_masks(district_masks).copy(),
        accepted_flips=accepted_flips,
        attempts=attempts,
        total_loss=total_loss,
        loss_delta=loss_delta,
        candidate=candidate,
        current_district=current_district,
        target_district=target_district,
    )


def run_monte_carlo(
    pops: np.ndarray,
    swing: np.ndarray,
    district_masks,
    weights: LossWeights,
    successful_flips_target: int,
    temperature: float,
):
    current_loss = compute_total_loss(pops, swing, district_masks, weights=weights)
    accepted_flips = 0
    attempts = 0
    history_frames = [
        capture_flip_history_frame(
            district_masks=district_masks,
            accepted_flips=accepted_flips,
            attempts=attempts,
            total_loss=current_loss,
            loss_delta=0.0,
        )
    ]

    print(f"\nInitial total loss: {current_loss:.6f}")

    while accepted_flips < successful_flips_target:
        attempts += 1

        neighbourliness = compute_edge_neighbourliness(district_masks)
        candidate = select_candidate(neighbourliness)
        if candidate is None:
            raise RuntimeError("No valid edge candidates remain.")

        current_district = get_current_district(candidate, district_masks)
        if current_district is None:
            continue

        target_options = get_adjacent_district_ids(candidate, district_masks)
        if not target_options:
            continue

        target_district = int(np.random.choice(target_options))
        proposed_masks = build_proposed_masks(
            district_masks=district_masks,
            candidate=candidate,
            current_district=current_district,
            target_district=target_district,
        )

        if not (
            check_connectivity(proposed_masks[current_district])
            and check_connectivity(proposed_masks[target_district])
        ):
            continue

        proposed_loss = compute_total_loss(pops, swing, proposed_masks, weights=weights)
        loss_delta = proposed_loss - current_loss
        accept_probability = acceptance_probability(loss_delta, temperature)

        if loss_delta <= 0 or np.random.random() < accept_probability:
            district_masks = proposed_masks
            current_loss = proposed_loss
            accepted_flips += 1
            history_frames.append(
                capture_flip_history_frame(
                    district_masks=district_masks,
                    accepted_flips=accepted_flips,
                    attempts=attempts,
                    total_loss=current_loss,
                    loss_delta=loss_delta,
                    candidate=candidate,
                    current_district=current_district,
                    target_district=target_district,
                )
            )
            print(
                f"Accepted {accepted_flips}/{successful_flips_target} "
                f"after attempt {attempts}: pixel {candidate} "
                f"{current_district}->{target_district}, "
                f"delta={loss_delta:.6f}, total={current_loss:.6f}, "
                f"p={accept_probability:.6f}"
            )

    print(f"\nCompleted {accepted_flips} accepted flips in {attempts} attempts.")
    print(f"Final total loss: {current_loss:.6f}")
    return district_masks, current_loss, history_frames


def main() -> None:
    pops, swing, district_masks, centers = initialize_map()
    overview_fig, overview_axes = launch_map_overview(
        pops=pops,
        swing=swing,
        district_masks=district_masks,
        centers=centers,
    )
    plt.show()
    del overview_fig, overview_axes

    weights = prompt_for_loss_weights()
    successful_flips_target = prompt_int("Number of accepted flips to complete [10]: ", default=10)
    temperature = prompt_float("Temperature for uphill moves [1.0]: ", default=1.0)

    _, _, history_frames = run_monte_carlo(
        pops=pops,
        swing=swing,
        district_masks=district_masks,
        weights=weights,
        successful_flips_target=successful_flips_target,
        temperature=temperature,
    )

    history_fig, history_slider = launch_flip_history_viewer(
        pops=pops,
        swing=swing,
        frames=history_frames,
    )
    plt.show()
    del history_fig, history_slider


if __name__ == "__main__":
    main()
