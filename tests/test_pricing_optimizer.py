"""Tests for pricing_optimizer.py — per-blockface rate recommendations.

Recommendations are RELATIVE adjustments to whatever rate is posted, because
Seattle publishes no per-meter posted rate (the `paidparkingrate` column is
populated on 0 of 25.2M rows). These tests pin that contract down.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pricing_optimizer as po


# ── Rate adjustment direction ────────────────────────────────────────────────

def test_no_adjustment_inside_target_band():
    """70-85% is the SMC target — leave those blockfaces alone."""
    assert po.rate_adjustment(0.75) == 0.0
    assert po.rate_adjustment(po.TARGET_LOW) == 0.0
    assert po.rate_adjustment(po.TARGET_HIGH) == 0.0


def test_over_target_raises_rate():
    assert po.rate_adjustment(0.95) > 0


def test_under_target_lowers_rate():
    assert po.rate_adjustment(0.40) < 0


def test_adjustment_scales_with_distance_from_target():
    """Further from the band should mean a larger move."""
    assert po.rate_adjustment(0.30) < po.rate_adjustment(0.60) < 0
    assert po.rate_adjustment(0.99) > po.rate_adjustment(0.88) > 0


def test_adjustment_respects_max_move():
    """No single recommendation may exceed the configured cap."""
    assert abs(po.rate_adjustment(0.0)) <= po.MAX_ADJUSTMENT
    assert abs(po.rate_adjustment(1.0)) <= po.MAX_ADJUSTMENT


def test_adjustment_lands_on_quarter_increments():
    """Cities post rates in $0.25 steps."""
    for occ in [0.0, 0.15, 0.35, 0.55, 0.9, 0.95, 1.0]:
        adj = po.rate_adjustment(occ)
        assert round(adj / po.RATE_STEP, 6) == round(adj / po.RATE_STEP)


def test_adjustment_handles_nan():
    assert po.rate_adjustment(float("nan")) == 0.0


# ── Action classification ────────────────────────────────────────────────────

def test_classify_matches_target_band():
    assert po.classify(0.95) == "increase"
    assert po.classify(0.40) == "decrease"
    assert po.classify(0.78) == "hold"
    assert po.classify(float("nan")) == "unknown"


def test_classify_agrees_with_rate_adjustment():
    """A 'hold' must never carry a nonzero adjustment, and vice versa."""
    for occ in np.arange(0.0, 1.01, 0.05):
        if po.classify(occ) == "hold":
            assert po.rate_adjustment(occ) == 0.0
        else:
            assert po.rate_adjustment(occ) != 0.0


# ── Demand response ──────────────────────────────────────────────────────────

def test_raising_rate_reduces_occupancy():
    r = po.demand_response(rate_delta=1.00, spaces=100, occupancy=0.90)
    assert r["projected_occupancy"] < 0.90
    assert r["occupancy_change_pts"] < 0


def test_lowering_rate_increases_occupancy():
    r = po.demand_response(rate_delta=-1.00, spaces=100, occupancy=0.40)
    assert r["projected_occupancy"] > 0.40
    assert r["occupancy_change_pts"] > 0


def test_no_rate_change_leaves_occupancy_flat():
    r = po.demand_response(rate_delta=0.0, spaces=100, occupancy=0.55)
    assert r["projected_occupancy"] == pytest.approx(0.55)
    assert r["revenue_delta_per_posted_dollar"] == pytest.approx(0.0)


def test_projected_occupancy_stays_bounded():
    """Occupancy is a fraction — it can never exceed 1 or drop below 0."""
    assert po.demand_response(-8.0, 100, 0.95)["projected_occupancy"] <= 1.0
    assert po.demand_response(8.0, 100, 0.05)["projected_occupancy"] >= 0.0


def test_revenue_is_expressed_per_posted_dollar():
    """The contract: revenue is reported per $1.00 of posted rate, never as an
    absolute dollar figure, since the posted rate is unknown."""
    r = po.demand_response(rate_delta=0.50, spaces=100, occupancy=0.90)
    assert "revenue_delta_per_posted_dollar" in r
    assert not any("revenue_delta" == k for k in r if k.endswith("_delta"))


# ── Output contract ──────────────────────────────────────────────────────────

def test_recommendations_carry_relative_not_absolute_rates():
    """Guard against a regression back to fabricated absolute rates."""
    recent = pd.DataFrame({
        "blockfacename": ["A ST BETWEEN 1ST AND 2ND"] * 2,
        "paidparkingarea": ["Belltown"] * 2,
        "hour_of_day": [10, 11],
        "avg_occupancy_rate": [0.95, 0.40],
        "total_spaces": [10.0, 10.0],
    })
    registry = pd.DataFrame({
        "blockfacename": ["A ST BETWEEN 1ST AND 2ND"],
        "spaces": [12.0],
    })
    recs = po.build_recommendations(recent, registry)

    assert "rate_adjustment" in recs.columns
    assert "recommended_rate" not in recs.columns, "absolute rates are fabricated"
    assert "current_rate" not in recs.columns, "absolute rates are fabricated"
    # Registry capacity must win over the feed's understated count
    assert recs["spaces"].iloc[0] == 12.0
    assert recs.loc[recs.hour_of_day == 10, "action"].iloc[0] == "increase"
    assert recs.loc[recs.hour_of_day == 11, "action"].iloc[0] == "decrease"


def test_bounds_are_the_legal_smc_values():
    assert po.RATE_MIN == 0.50
    assert po.RATE_MAX == 8.00
    assert po.TARGET_LOW == 0.70
    assert po.TARGET_HIGH == 0.85
