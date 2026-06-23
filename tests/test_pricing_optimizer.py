"""Tests for pricing_optimizer.py — rate logic under SMC 11.16.121."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pricing_optimizer as po


def test_recommend_rate_hold():
    """Occupancy in 70-85% band → no change."""
    assert po.recommend_rate(0.75, 2.50) == 2.50
    assert po.recommend_rate(0.70, 2.00) == 2.00
    assert po.recommend_rate(0.85, 2.00) == 2.00


def test_recommend_rate_increase():
    """Occupancy > 85% → rate increases."""
    rec = po.recommend_rate(0.95, 2.50)
    assert rec > 2.50


def test_recommend_rate_decrease():
    """Occupancy < 70% → rate decreases."""
    rec = po.recommend_rate(0.40, 2.50)
    assert rec < 2.50


def test_recommend_rate_bounds():
    """Rate must never go below RATE_MIN or above RATE_MAX."""
    assert po.recommend_rate(0.0, 0.50) >= po.RATE_MIN
    assert po.recommend_rate(1.0, 8.00) <= po.RATE_MAX


def test_revenue_impact_increase():
    """Higher rate with elastic demand → net revenue check."""
    result = po.revenue_impact(current_rate=2.00, recommended_rate=2.50,
                               spaces=100, hours=1.0)
    assert "revenue_delta" in result
    assert "projected_revenue" in result
    # With 5% elasticity per $0.25 increase, $0.50 increase → 10% demand drop
    # Revenue = 2.50 × 90 vs 2.00 × 100 = 225 vs 200 → positive
    assert result["revenue_delta"] > 0


def test_revenue_impact_decrease():
    """Lower rate with elastic demand — verify structure."""
    result = po.revenue_impact(current_rate=2.00, recommended_rate=1.50, spaces=100)
    assert "current_revenue" in result
    assert "projected_revenue" in result
