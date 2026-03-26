from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

import numpy as np

PreferredParty = Literal["blue", "red"]
ElectionMetric = Literal["seat_share", "mean_margin", "hybrid", "tipping_band"]
BalanceMetric = Literal["mae", "rmse", "max", "kurtosis"]

_EPS = 1e-12


@dataclass(frozen=True)
class LossWeights:
    """
    Relative weights for each loss category.

    `election_outcome` uses sign and magnitude:
    - sign: preferred side (+ => blue, - => red) when config does not set `preferred_party`
    - magnitude: weight in total loss
    """

    election_outcome: float = 1.0
    fault_tolerance: float = 1.0
    district_cleanliness: float = 1.0
    district_balance: float = 1.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, float]) -> "LossWeights":
        return cls(
            election_outcome=float(value.get("election_outcome", 1.0)),
            fault_tolerance=float(value.get("fault_tolerance", 1.0)),
            district_cleanliness=float(value.get("district_cleanliness", 1.0)),
            district_balance=float(value.get("district_balance", 1.0)),
        )


@dataclass(frozen=True)
class FaultToleranceConfig:
    enabled: bool = False
    trials: int = 200
    district_noise_std: float = 0.08
    ci_alpha: float = 0.10
    random_seed: int | None = None


@dataclass(frozen=True)
class LossConfig:
    preferred_party: PreferredParty | None = None
    election_metric: ElectionMetric = "tipping_band"
    progress_tau: float = 0.10
    tipping_band_radius: int = 1
    balance_metric: BalanceMetric = "kurtosis"
    fault_tolerance: FaultToleranceConfig = field(default_factory=FaultToleranceConfig)


@dataclass(frozen=True)
class LossBreakdown:
    total: float
    election_outcome: float
    fault_tolerance: float
    district_cleanliness: float
    district_balance: float
    weighted_election_outcome: float
    weighted_fault_tolerance: float
    weighted_district_cleanliness: float
    weighted_district_balance: float

    def as_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "election_outcome": self.election_outcome,
            "fault_tolerance": self.fault_tolerance,
            "district_cleanliness": self.district_cleanliness,
            "district_balance": self.district_balance,
            "weighted_election_outcome": self.weighted_election_outcome,
            "weighted_fault_tolerance": self.weighted_fault_tolerance,
            "weighted_district_cleanliness": self.weighted_district_cleanliness,
            "weighted_district_balance": self.weighted_district_balance,
        }


def district_indices_from_masks(district_masks: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """
    Convert district masks to a label map.
    Pixels with no district assignment are labeled -1.
    If overlaps exist, the lowest-index district wins tie-break.
    """

    mask_stack = _stack_masks(district_masks)
    labels, _ = _labels_from_mask_stack(mask_stack)
    return labels


def compute_district_stats_from_masks(
    pops: np.ndarray,
    swing: np.ndarray,
    district_masks: Sequence[np.ndarray] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return per-district stats from masks and cell-level data.

    Returns:
        district_pop: total population per district
        district_vote: population-weighted swing sum per district
        district_mean_swing: district_vote / district_pop in [-1, 1]
    """

    mask_stack = _stack_masks(district_masks)
    pops_arr, swing_arr = _validate_grid_inputs(pops, swing, mask_stack.shape[1:])
    labels, _ = _labels_from_mask_stack(mask_stack)
    num_districts = mask_stack.shape[0]
    return _district_stats_from_labels(labels, pops_arr, swing_arr, num_districts)


def compute_loss_breakdown(
    pops: np.ndarray,
    swing: np.ndarray,
    district_masks: Sequence[np.ndarray] | np.ndarray,
    weights: LossWeights | Mapping[str, float] | None = None,
    config: LossConfig | None = None,
) -> LossBreakdown:
    """
    Compute component losses and weighted total loss from district masks.

    All component losses are normalized to [0, 1], where 0 is best.
    """

    cfg = config or LossConfig()
    w = _coerce_weights(weights)

    mask_stack = _stack_masks(district_masks)
    pops_arr, swing_arr = _validate_grid_inputs(pops, swing, mask_stack.shape[1:])
    labels, _ = _labels_from_mask_stack(mask_stack)
    district_pop, _, district_mean_swing = _district_stats_from_labels(
        labels=labels,
        pops=pops_arr,
        swing=swing_arr,
        num_districts=mask_stack.shape[0],
    )

    preferred_party = cfg.preferred_party or ("blue" if w.election_outcome >= 0.0 else "red")
    election_outcome_loss = _election_outcome_loss(
        district_pop=district_pop,
        district_mean_swing=district_mean_swing,
        preferred_party=preferred_party,
        metric=cfg.election_metric,
        progress_tau=cfg.progress_tau,
        tipping_band_radius=cfg.tipping_band_radius,
    )
    fault_tolerance_loss = _fault_tolerance_loss(
        district_mean_swing=district_mean_swing,
        cfg=cfg.fault_tolerance,
    )
    district_cleanliness_loss = _district_cleanliness_loss(
        labels=labels,
        pops=pops_arr,
        num_districts=mask_stack.shape[0],
    )
    district_balance_loss = _district_balance_loss(
        district_pop=district_pop,
        metric=cfg.balance_metric,
    )

    weighted_election = abs(float(w.election_outcome)) * election_outcome_loss
    weighted_fault = max(0.0, float(w.fault_tolerance)) * fault_tolerance_loss
    weighted_cleanliness = max(0.0, float(w.district_cleanliness)) * district_cleanliness_loss
    weighted_balance = max(0.0, float(w.district_balance)) * district_balance_loss

    total = weighted_election + weighted_fault + weighted_cleanliness + weighted_balance

    return LossBreakdown(
        total=float(total),
        election_outcome=float(election_outcome_loss),
        fault_tolerance=float(fault_tolerance_loss),
        district_cleanliness=float(district_cleanliness_loss),
        district_balance=float(district_balance_loss),
        weighted_election_outcome=float(weighted_election),
        weighted_fault_tolerance=float(weighted_fault),
        weighted_district_cleanliness=float(weighted_cleanliness),
        weighted_district_balance=float(weighted_balance),
    )


def compute_total_loss(
    pops: np.ndarray,
    swing: np.ndarray,
    district_masks: Sequence[np.ndarray] | np.ndarray,
    weights: LossWeights | Mapping[str, float] | None = None,
    config: LossConfig | None = None,
) -> float:
    return compute_loss_breakdown(
        pops=pops,
        swing=swing,
        district_masks=district_masks,
        weights=weights,
        config=config,
    ).total


def _coerce_weights(weights: LossWeights | Mapping[str, float] | None) -> LossWeights:
    if weights is None:
        return LossWeights()
    if isinstance(weights, LossWeights):
        return weights
    return LossWeights.from_mapping(weights)


def _stack_masks(district_masks: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    if isinstance(district_masks, np.ndarray):
        if district_masks.ndim != 3:
            raise ValueError("district_masks ndarray must have shape (num_districts, rows, cols).")
        if district_masks.shape[0] == 0:
            raise ValueError("district_masks must include at least one district.")
        return district_masks.astype(bool, copy=False)

    if not district_masks:
        raise ValueError("district_masks must include at least one district.")

    masks = [np.asarray(mask, dtype=bool) for mask in district_masks]
    shape = masks[0].shape
    if len(shape) != 2:
        raise ValueError("each district mask must be 2D.")
    for idx, mask in enumerate(masks[1:], start=1):
        if mask.shape != shape:
            raise ValueError(f"district mask at index {idx} has shape {mask.shape}, expected {shape}.")
    return np.stack(masks, axis=0)


def _validate_grid_inputs(
    pops: np.ndarray,
    swing: np.ndarray,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    pops_arr = np.asarray(pops, dtype=float)
    swing_arr = np.asarray(swing, dtype=float)

    if pops_arr.shape != expected_shape:
        raise ValueError(f"pops shape {pops_arr.shape} does not match mask shape {expected_shape}.")
    if swing_arr.shape != expected_shape:
        raise ValueError(f"swing shape {swing_arr.shape} does not match mask shape {expected_shape}.")
    return pops_arr, swing_arr


def _labels_from_mask_stack(mask_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coverage_count = mask_stack.sum(axis=0)
    labels = np.argmax(mask_stack, axis=0).astype(np.int16)
    labels[coverage_count == 0] = -1
    return labels, coverage_count


def _district_stats_from_labels(
    labels: np.ndarray,
    pops: np.ndarray,
    swing: np.ndarray,
    num_districts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = labels >= 0
    if not np.any(valid):
        zeros = np.zeros(num_districts, dtype=float)
        return zeros, zeros, zeros

    flat_labels = labels[valid].ravel()
    flat_pops = pops[valid].ravel()
    flat_vote = (pops * swing)[valid].ravel()

    district_pop = np.bincount(flat_labels, weights=flat_pops, minlength=num_districts).astype(float)
    district_vote = np.bincount(flat_labels, weights=flat_vote, minlength=num_districts).astype(float)

    district_mean_swing = np.zeros(num_districts, dtype=float)
    nonzero = district_pop > _EPS
    district_mean_swing[nonzero] = district_vote[nonzero] / district_pop[nonzero]
    district_mean_swing = np.clip(district_mean_swing, -1.0, 1.0)

    return district_pop, district_vote, district_mean_swing


def _election_outcome_loss(
    district_pop: np.ndarray,
    district_mean_swing: np.ndarray,
    preferred_party: PreferredParty,
    metric: ElectionMetric,
    progress_tau: float,
    tipping_band_radius: int,
) -> float:
    if district_mean_swing.size == 0:
        return 1.0

    signed_margin, preferred_wins = _preferred_party_view(district_mean_swing, preferred_party)

    seat_share_loss = 1.0 - float(np.mean(preferred_wins))
    margin_loss = 0.5 * (1.0 - float(np.mean(np.clip(signed_margin, -1.0, 1.0))))

    if metric == "seat_share":
        return float(np.clip(seat_share_loss, 0.0, 1.0))
    if metric == "mean_margin":
        return float(np.clip(margin_loss, 0.0, 1.0))
    if metric == "hybrid":
        # Seat wins matter most, but margin discourages razor-thin maps.
        return float(np.clip(0.8 * seat_share_loss + 0.2 * margin_loss, 0.0, 1.0))
    if metric == "tipping_band":
        return _tipping_band_loss(
            district_pop=district_pop,
            signed_margin=signed_margin,
            tau=progress_tau,
            band_radius=tipping_band_radius,
        )
    raise ValueError(f"Unsupported election metric: {metric}")


def _preferred_party_view(
    district_mean_swing: np.ndarray,
    preferred_party: PreferredParty,
) -> tuple[np.ndarray, np.ndarray]:
    if preferred_party == "blue":
        signed_margin = np.asarray(district_mean_swing, dtype=float)
        preferred_wins = signed_margin > 0.0
    else:
        signed_margin = -np.asarray(district_mean_swing, dtype=float)
        # Preserve the legacy convention where ties break toward red.
        preferred_wins = signed_margin >= 0.0

    return signed_margin, preferred_wins


def _tipping_band_loss(
    district_pop: np.ndarray,
    signed_margin: np.ndarray,
    tau: float,
    band_radius: int,
) -> float:
    electoral_weight = _normalized_electoral_weights(district_pop)
    sort_order = np.argsort(-np.asarray(signed_margin, dtype=float))
    sorted_margin = np.asarray(signed_margin, dtype=float)[sort_order]
    sorted_weight = electoral_weight[sort_order]
    num_districts = sorted_margin.size
    if num_districts == 0:
        return 1.0

    cumulative_weight = np.cumsum(sorted_weight)
    decisive_idx = int(np.searchsorted(cumulative_weight, 0.5, side="left"))
    decisive_idx = min(decisive_idx, num_districts - 1)
    radius = max(int(band_radius), 0)

    start = max(0, decisive_idx - radius)
    stop = min(num_districts, decisive_idx + radius + 1)
    band_indices = np.arange(start, stop)
    proximity_weight = (radius + 1) - np.abs(band_indices - decisive_idx)
    weights = proximity_weight * sorted_weight[start:stop]
    if not np.any(weights > 0.0):
        weights = proximity_weight.astype(float)
    tip_margin = float(np.average(sorted_margin[start:stop], weights=weights))

    tau_value = max(float(tau), _EPS)
    return _sigmoid(-tip_margin / tau_value)


def _normalized_electoral_weights(district_pop: np.ndarray) -> np.ndarray:
    pop = np.asarray(district_pop, dtype=float)
    total_pop = float(np.sum(pop))
    num_districts = pop.size
    if num_districts == 0:
        return pop
    if total_pop <= _EPS:
        return np.full(num_districts, 1.0 / float(num_districts), dtype=float)
    return pop / total_pop


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_term = float(np.exp(-value))
        return float(1.0 / (1.0 + exp_term))
    exp_term = float(np.exp(value))
    return float(exp_term / (1.0 + exp_term))


def _fault_tolerance_loss(district_mean_swing: np.ndarray, cfg: FaultToleranceConfig) -> float:
    if (
        not cfg.enabled
        or district_mean_swing.size == 0
        or cfg.trials < 2
        or cfg.district_noise_std <= 0.0
    ):
        return 0.0

    rng = np.random.default_rng(cfg.random_seed)
    noise = rng.normal(0.0, cfg.district_noise_std, size=(cfg.trials, district_mean_swing.size))
    noisy_swing = district_mean_swing[None, :] + noise

    baseline_winner = district_mean_swing > 0.0
    trial_winner = noisy_swing > 0.0

    district_flip_rate = np.mean(trial_winner != baseline_winner[None, :], axis=0)
    mean_flip_rate = float(np.mean(district_flip_rate))

    blue_seat_share = np.mean(trial_winner, axis=1)
    alpha = float(np.clip(cfg.ci_alpha, 1e-6, 0.999))
    q_low, q_high = np.quantile(blue_seat_share, [alpha / 2.0, 1.0 - alpha / 2.0])
    seat_share_ci_width = float(q_high - q_low)

    # Higher CI width and higher winner-flip rate both indicate brittle districts.
    combined = 0.5 * mean_flip_rate + 0.5 * seat_share_ci_width
    return float(np.clip(combined, 0.0, 1.0))


def _district_cleanliness_loss(
    labels: np.ndarray,
    pops: np.ndarray,
    num_districts: int,
) -> float:
    valid = labels >= 0
    if not np.any(valid):
        return 1.0

    centers, active_districts = _district_population_centers(
        labels=labels,
        pops=pops,
        num_districts=num_districts,
    )
    if not np.any(active_districts):
        return 1.0

    row_coords, col_coords = np.indices(labels.shape, dtype=float)
    flat_labels = labels[valid].ravel()
    flat_rows = row_coords[valid].ravel()[:, None]
    flat_cols = col_coords[valid].ravel()[:, None]

    active_centers = centers[active_districts]
    active_ids = np.flatnonzero(active_districts)
    dist_sq = (flat_rows - active_centers[:, 0]) ** 2 + (flat_cols - active_centers[:, 1]) ** 2
    voronoi_labels = active_ids[np.argmin(dist_sq, axis=1)]

    mismatch_rate = np.mean(voronoi_labels != flat_labels)
    return float(mismatch_rate)


def _district_population_centers(
    labels: np.ndarray,
    pops: np.ndarray,
    num_districts: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = labels >= 0
    centers = np.full((num_districts, 2), np.nan, dtype=float)
    active_districts = np.zeros(num_districts, dtype=bool)
    if not np.any(valid):
        return centers, active_districts

    row_coords, col_coords = np.indices(labels.shape, dtype=float)
    flat_labels = labels[valid].ravel()
    flat_rows = row_coords[valid].ravel()
    flat_cols = col_coords[valid].ravel()
    flat_pops = pops[valid].ravel()

    district_area = np.bincount(flat_labels, minlength=num_districts).astype(float)
    district_pop = np.bincount(flat_labels, weights=flat_pops, minlength=num_districts).astype(float)
    weighted_row_sum = np.bincount(
        flat_labels,
        weights=flat_rows * flat_pops,
        minlength=num_districts,
    ).astype(float)
    weighted_col_sum = np.bincount(
        flat_labels,
        weights=flat_cols * flat_pops,
        minlength=num_districts,
    ).astype(float)
    geom_row_sum = np.bincount(flat_labels, weights=flat_rows, minlength=num_districts).astype(float)
    geom_col_sum = np.bincount(flat_labels, weights=flat_cols, minlength=num_districts).astype(float)

    active_districts = district_area > 0.0
    pop_weighted = district_pop > _EPS
    centers[pop_weighted, 0] = weighted_row_sum[pop_weighted] / district_pop[pop_weighted]
    centers[pop_weighted, 1] = weighted_col_sum[pop_weighted] / district_pop[pop_weighted]

    zero_pop = active_districts & ~pop_weighted
    centers[zero_pop, 0] = geom_row_sum[zero_pop] / district_area[zero_pop]
    centers[zero_pop, 1] = geom_col_sum[zero_pop] / district_area[zero_pop]

    return centers, active_districts


def _district_balance_loss(district_pop: np.ndarray, metric: BalanceMetric) -> float:
    num_districts = district_pop.size
    if num_districts == 0:
        return 1.0

    total_pop = float(np.sum(district_pop))
    if total_pop <= _EPS:
        return 1.0

    ideal = total_pop / float(num_districts)
    rel_dev = np.abs(district_pop - ideal) / (ideal + _EPS)

    if metric == "kurtosis":
        signed_rel_dev = (district_pop - ideal) / (ideal + _EPS)
        raw = float(np.mean(signed_rel_dev**4))
        max_raw = (((num_districts - 1.0) ** 4) + (num_districts - 1.0)) / float(num_districts)
    elif metric == "mae":
        raw = float(np.mean(rel_dev))
        max_raw = 2.0 * (num_districts - 1.0) / float(num_districts)
    elif metric == "rmse":
        raw = float(np.sqrt(np.mean(rel_dev**2)))
        max_raw = float(np.sqrt(max(num_districts - 1, 1)))
    elif metric == "max":
        raw = float(np.max(rel_dev))
        max_raw = float(max(num_districts - 1, 1))
    else:
        raise ValueError(f"Unsupported balance metric: {metric}")

    return float(np.clip(raw / (max_raw + _EPS), 0.0, 1.0))
