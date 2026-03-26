from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

from maps import init_rectangleland, init_squareland
from model.evaluate import check_connectivity
from model.loss import LossWeights, compute_total_loss, district_indices_from_masks
from user_interface.custom_loss_ui import (
    get_loss_weights,
    get_uphill_move_probability,
    launch_loss_weight_sliders,
)
from user_interface.flip_history_ui import (
    FlipHistoryFrame,
    export_flip_history_movie,
    launch_flip_history_viewer,
)


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


def acceptance_probability(loss_delta: float, uphill_move_probability: float) -> float:
    if loss_delta <= 0:
        return 1.0
    return float(np.clip(uphill_move_probability, 0.0, 1.0))


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


def prompt_for_run_controls() -> tuple[LossWeights, float]:
    current_weights = get_loss_weights()
    current_uphill_move_probability = get_uphill_move_probability()

    while True:
        fig, sliders = launch_loss_weight_sliders(
            initial=current_weights,
            initial_uphill_move_probability=current_uphill_move_probability,
        )
        plt.show()
        # Keep widget references alive until the window closes.
        del fig, sliders
        current_weights = get_loss_weights()
        current_uphill_move_probability = get_uphill_move_probability()

        print("\nCurrent loss weights:")
        for key, value in asdict(current_weights).items():
            print(f"  {key}: {value}")
        print(f"  uphill_move_probability: {current_uphill_move_probability}")

        if prompt_yes_no("Use these settings? [y/n]: "):
            return current_weights, current_uphill_move_probability


def _artifacts_dir() -> Path:
    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def write_run_specs_artifact(
    *,
    timestamp: str,
    map_name: str,
    n: int,
    num_districts: int,
    num_cities: int,
    num_farms: int,
    require_continuity: bool,
    weights: LossWeights,
    successful_flips_target: int,
    uphill_move_probability: float,
    pops: np.ndarray,
    swing: np.ndarray,
    history_frames: list[FlipHistoryFrame],
    video_filename: str,
) -> Path:
    specs_path = _artifacts_dir() / f"run_specs_{timestamp}.json"
    payload = {
        "title": f"gerrymandering_run_{timestamp}",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "map": map_name,
        "n": n,
        "num_districts": num_districts,
        "num_cities": num_cities,
        "num_farms": num_farms,
        "require_continuity": require_continuity,
        "successful_flips_target": successful_flips_target,
        "uphill_move_probability": uphill_move_probability,
        "weights": asdict(weights),
        "grid_shape": [int(pops.shape[0]), int(pops.shape[1])],
        "total_population": float(np.sum(pops)),
        "swing_range": [float(np.min(swing)), float(np.max(swing))],
        "frame_count": len(history_frames),
        "accepted_flips": int(history_frames[-1].accepted_flips),
        "attempts": int(history_frames[-1].attempts),
        "initial_total_loss": float(history_frames[0].total_loss),
        "final_total_loss": float(history_frames[-1].total_loss),
        "video_filename": video_filename,
    }
    specs_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return specs_path


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
    uphill_move_probability: float,
    require_continuity: bool = True,
    max_attempts_without_accept: int | None = None,
):
    current_loss = compute_total_loss(pops, swing, district_masks, weights=weights)
    accepted_flips = 0
    attempts = 0
    attempts_since_accept = 0
    if max_attempts_without_accept is None:
        max_attempts_without_accept = max(2000, 200 * successful_flips_target, 400 * len(district_masks))

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
    stop_reason: str | None = None

    try:
        while accepted_flips < successful_flips_target:
            attempts += 1
            attempts_since_accept += 1

            neighbourliness = compute_edge_neighbourliness(district_masks)
            candidate = select_candidate(neighbourliness)
            if candidate is None:
                stop_reason = "No valid edge candidates remain."
                break

            current_district = get_current_district(candidate, district_masks)
            if current_district is None:
                if attempts_since_accept >= max_attempts_without_accept:
                    stop_reason = (
                        f"Stopping early after {attempts_since_accept} consecutive attempts without an accepted flip."
                    )
                    break
                continue

            target_options = get_adjacent_district_ids(candidate, district_masks)
            if not target_options:
                if attempts_since_accept >= max_attempts_without_accept:
                    stop_reason = (
                        f"Stopping early after {attempts_since_accept} consecutive attempts without an accepted flip."
                    )
                    break
                continue

            target_district = int(np.random.choice(target_options))
            proposed_masks = build_proposed_masks(
                district_masks=district_masks,
                candidate=candidate,
                current_district=current_district,
                target_district=target_district,
            )

            if require_continuity and not (
                check_connectivity(proposed_masks[current_district])
                and check_connectivity(proposed_masks[target_district])
            ):
                if attempts_since_accept >= max_attempts_without_accept:
                    stop_reason = (
                        f"Stopping early after {attempts_since_accept} consecutive attempts without an accepted flip."
                    )
                    break
                continue

            proposed_loss = compute_total_loss(pops, swing, proposed_masks, weights=weights)
            loss_delta = proposed_loss - current_loss
            accept_probability = acceptance_probability(loss_delta, uphill_move_probability)

            if loss_delta <= 0 or np.random.random() < accept_probability:
                district_masks = proposed_masks
                current_loss = proposed_loss
                accepted_flips += 1
                attempts_since_accept = 0
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
            elif attempts_since_accept >= max_attempts_without_accept:
                stop_reason = (
                    f"Stopping early after {attempts_since_accept} consecutive attempts without an accepted flip. "
                    "Likely reached a local minimum at the current uphill-move probability."
                )
                break
    except KeyboardInterrupt:
        stop_reason = (
            f"Interrupted by user after {attempts} attempts. "
            "Returning the history collected so far."
        )

    if stop_reason is not None:
        print(f"\n{stop_reason}")
        print(f"Accepted {accepted_flips}/{successful_flips_target} flips before stopping.")

    if accepted_flips >= successful_flips_target:
        print(f"\nCompleted {accepted_flips} accepted flips in {attempts} attempts.")
    else:
        print(f"\nFinished with {accepted_flips} accepted flips in {attempts} attempts.")
    print(f"Final total loss: {current_loss:.6f}")
    return district_masks, current_loss, history_frames


def main(
    map_name: str = "squareland",
    require_continuity: bool = True,
    n: int = 25,
    num_districts: int = 5,
    num_cities: int = 2,
    num_farms: int = 4,
) -> None:
    map_module = init_squareland if map_name == "squareland" else init_rectangleland

    pops, swing, district_masks, centers = map_module.initialize_map(
        n=n,
        num_districts=num_districts,
        num_cities=num_cities,
        num_farms=num_farms,
    )
    overview_fig, overview_axes = map_module.launch_map_overview(
        pops=pops,
        swing=swing,
        district_masks=district_masks,
        centers=centers,
    )
    plt.show()
    del overview_fig, overview_axes

    weights, uphill_move_probability = prompt_for_run_controls()
    successful_flips_target = prompt_int("Number of accepted flips to complete [10]: ", default=10)
    print("Press Ctrl+C during the Monte Carlo run to stop early and open the history viewer with partial results.")

    _, _, history_frames = run_monte_carlo(
        pops=pops,
        swing=swing,
        district_masks=district_masks,
        weights=weights,
        successful_flips_target=successful_flips_target,
        uphill_move_probability=uphill_move_probability,
        require_continuity=require_continuity,
    )

    history_fig, history_slider = launch_flip_history_viewer(
        pops=pops,
        swing=swing,
        frames=history_frames,
    )
    plt.show()
    del history_fig, history_slider

    if prompt_yes_no("Export as video? [y/n]: "):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = _artifacts_dir() / f"flip_history_{timestamp}.mov"
        specs_path = write_run_specs_artifact(
            timestamp=timestamp,
            map_name=map_name,
            n=n,
            num_districts=num_districts,
            num_cities=num_cities,
            num_farms=num_farms,
            require_continuity=require_continuity,
            weights=weights,
            successful_flips_target=successful_flips_target,
            uphill_move_probability=uphill_move_probability,
            pops=pops,
            swing=swing,
            history_frames=history_frames,
            video_filename=video_path.name,
        )
        print(f"Wrote run specs to {specs_path}")
        try:
            export_flip_history_movie(
                pops=pops,
                swing=swing,
                frames=history_frames,
                output_path=video_path,
            )
        except RuntimeError as exc:
            print(f"Video export failed: {exc}")
        else:
            print(f"Wrote video to {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the gerrymandering simulation.")
    parser.add_argument(
        "--map",
        choices=("squareland", "rectangleland"),
        default="squareland",
        help="Choose which map to initialize.",
    )
    parser.add_argument("--n", type=int, default=25, help="Map width and height in cells.")
    parser.add_argument("--num-districts", type=int, default=5, help="Number of districts.")
    parser.add_argument("--num-cities", type=int, default=2, help="Number of city centers.")
    parser.add_argument("--num-farms", type=int, default=4, help="Number of farm centers.")
    parser.add_argument(
        "--require-continuity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require both affected districts to remain contiguous before accepting a flip.",
    )
    args = parser.parse_args()

    main(
        map_name=args.map,
        require_continuity=args.require_continuity,
        n=args.n,
        num_districts=args.num_districts,
        num_cities=args.num_cities,
        num_farms=args.num_farms,
    )
