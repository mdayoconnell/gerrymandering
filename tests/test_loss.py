from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.loss import (
    LossConfig,
    _district_balance_loss,
    _district_cleanliness_loss,
    _election_outcome_loss,
    _tipping_band_loss,
)


def _loss(
    margins,
    district_pop=None,
    preferred_party: str = "blue",
    metric: str = "tipping_band",
    progress_tau: float = 0.10,
    tipping_band_radius: int = 1,
) -> float:
    margin_array = np.asarray(margins, dtype=float)
    if district_pop is None:
        district_pop = np.ones_like(margin_array, dtype=float)
    return _election_outcome_loss(
        district_pop=np.asarray(district_pop, dtype=float),
        district_mean_swing=margin_array,
        preferred_party=preferred_party,
        metric=metric,
        progress_tau=progress_tau,
        tipping_band_radius=tipping_band_radius,
    )


class TippingBandLossTests(unittest.TestCase):
    def test_loss_config_defaults_to_tipping_band(self) -> None:
        cfg = LossConfig()
        self.assertEqual(cfg.election_metric, "tipping_band")
        self.assertAlmostEqual(cfg.progress_tau, 0.10)
        self.assertEqual(cfg.tipping_band_radius, 1)
        self.assertEqual(cfg.balance_metric, "kurtosis")

    def test_preferred_party_symmetry(self) -> None:
        margins = np.array([-0.45, -0.05, 0.10, 0.35, 0.90], dtype=float)
        blue_loss = _loss(margins, preferred_party="blue")
        red_loss = _loss(-margins, preferred_party="red")
        self.assertAlmostEqual(blue_loss, red_loss, places=12)

    def test_monotonic_progress_without_seat_flip(self) -> None:
        base = np.array([0.90, 0.20, -0.30, -0.40, -0.80], dtype=float)
        improved = np.array([0.90, 0.20, -0.05, -0.40, -0.80], dtype=float)
        self.assertEqual(int(np.sum(base > 0.0)), int(np.sum(improved > 0.0)))
        self.assertLess(_loss(improved), _loss(base))

    def test_majority_crossing_lowers_loss(self) -> None:
        just_below = np.array([0.90, 0.16, -0.03, -0.19, -0.80], dtype=float)
        just_above = np.array([0.90, 0.16, 0.03, -0.19, -0.80], dtype=float)
        self.assertLess(_loss(just_above), _loss(just_below))

    def test_pivotal_improvement_matters_more_than_safe_seat(self) -> None:
        base = np.array([0.70, 0.20, -0.05, -0.40, -0.80], dtype=float)
        safe_improved = np.array([0.95, 0.20, -0.05, -0.40, -0.80], dtype=float)
        pivotal_improved = np.array([0.70, 0.20, 0.15, -0.40, -0.80], dtype=float)

        base_loss = _loss(base)
        safe_shift = abs(_loss(safe_improved) - base_loss)
        pivotal_shift = abs(_loss(pivotal_improved) - base_loss)

        self.assertLess(safe_shift, pivotal_shift)

    def test_edge_handling_for_one_and_two_districts(self) -> None:
        one_district_loss = _tipping_band_loss(
            district_pop=np.array([1.0], dtype=float),
            signed_margin=np.array([0.25], dtype=float),
            tau=0.10,
            band_radius=1,
        )
        expected_one = 1.0 / (1.0 + math.exp(0.25 / 0.10))
        self.assertAlmostEqual(one_district_loss, expected_one, places=12)

        two_district_loss = _tipping_band_loss(
            district_pop=np.array([0.30, 0.70], dtype=float),
            signed_margin=np.array([0.30, -0.20], dtype=float),
            tau=0.10,
            band_radius=1,
        )
        expected_tip_margin = ((0.30 * 0.30) + (2.0 * -0.20 * 0.70)) / (0.30 + (2.0 * 0.70))
        expected_two = 1.0 / (1.0 + math.exp(expected_tip_margin / 0.10))
        self.assertAlmostEqual(two_district_loss, expected_two, places=12)

    def test_large_losing_district_prevents_false_zero_loss(self) -> None:
        margins = np.array([0.80, -0.60, -0.50, -0.40, -0.30], dtype=float)
        district_pop = np.array([0.63, 0.12, 0.11, 0.08, 0.06], dtype=float)
        red_loss = _loss(margins, district_pop=district_pop, preferred_party="red")
        self.assertGreater(red_loss, 0.95)

    def test_legacy_metrics_unchanged(self) -> None:
        margins = np.array([0.80, 0.40, -0.20, -0.60], dtype=float)
        expected_seat_share = 1.0 - (2.0 / 4.0)
        expected_mean_margin = 0.5 * (1.0 - float(np.mean(margins)))
        expected_hybrid = (0.8 * expected_seat_share) + (0.2 * expected_mean_margin)

        self.assertAlmostEqual(_loss(margins, metric="seat_share"), expected_seat_share, places=12)
        self.assertAlmostEqual(_loss(margins, metric="mean_margin"), expected_mean_margin, places=12)
        self.assertAlmostEqual(_loss(margins, metric="hybrid"), expected_hybrid, places=12)


class DistrictCleanlinessLossTests(unittest.TestCase):
    def test_voronoi_aligned_districts_have_zero_loss(self) -> None:
        labels = np.array([[0, 0, 1, 1]], dtype=np.int16)
        pops = np.ones_like(labels, dtype=float)

        loss = _district_cleanliness_loss(labels=labels, pops=pops, num_districts=2)

        self.assertAlmostEqual(loss, 0.0, places=12)

    def test_pixels_outside_voronoi_region_are_counted(self) -> None:
        labels = np.array([[0, 0, 0, 1, 0]], dtype=np.int16)
        pops = np.ones_like(labels, dtype=float)

        loss = _district_cleanliness_loss(labels=labels, pops=pops, num_districts=2)

        self.assertAlmostEqual(loss, 1.0 / 5.0, places=12)

    def test_population_weighted_centers_shift_voronoi_boundary(self) -> None:
        labels = np.array([[0, 1, 1, 1, 0]], dtype=np.int16)
        pops = np.array([[1.0, 1.0, 1.0, 1.0, 100.0]], dtype=float)

        loss = _district_cleanliness_loss(labels=labels, pops=pops, num_districts=2)

        self.assertAlmostEqual(loss, 2.0 / 5.0, places=12)


class DistrictBalanceLossTests(unittest.TestCase):
    def test_kurtosis_balance_is_zero_for_equal_populations(self) -> None:
        district_pop = np.array([5.0, 5.0, 5.0, 5.0], dtype=float)

        loss = _district_balance_loss(district_pop, metric="kurtosis")

        self.assertAlmostEqual(loss, 0.0, places=12)

    def test_kurtosis_balance_penalizes_outliers_more_strongly(self) -> None:
        mild = np.array([1.0, 1.0, 1.0, 2.0], dtype=float)
        severe = np.array([1.0, 1.0, 1.0, 10.0], dtype=float)

        mild_loss = _district_balance_loss(mild, metric="kurtosis")
        severe_loss = _district_balance_loss(severe, metric="kurtosis")

        self.assertLess(mild_loss, severe_loss)

    def test_kurtosis_balance_normalizes_worst_case_to_one(self) -> None:
        district_pop = np.array([0.0, 0.0, 0.0, 4.0], dtype=float)

        loss = _district_balance_loss(district_pop, metric="kurtosis")

        self.assertAlmostEqual(loss, 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
