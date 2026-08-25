"""Tests for aggregate_features.py — the core feature engineering step.

The unit of analysis is the BLOCKFACE. Region labels come from Seattle's own
`paidparkingarea` field via the meter registry, so there is no longer any
geocoding/keyword-matching logic to test.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import aggregate_features as af


def make_parking_df():
    """Two blockfaces in the same hour, so aggregation must keep them separate."""
    return pd.DataFrame({
        "occupancy_date": pd.to_datetime(["2024-03-15", "2024-03-15"]),
        "occupancy_hour": [10, 10],
        "hour": pd.to_datetime(["2024-03-15 10:00", "2024-03-15 10:00"]),
        "blockfacename": ["PIKE ST BETWEEN 1ST AVE AND 2ND AVE",
                          "5TH AVE BETWEEN PIKE ST AND PINE ST"],
        "avg_occupied": [3.5, 8.0],
        "peak_occupied": [5.0, 10.0],
        "avg_spaces": [5.0, 10.0],
    })


# ── Aggregation keeps blockface granularity ──────────────────────────────────

def test_aggregate_parking_preserves_blockface_granularity():
    """The whole platform depends on NOT collapsing blockfaces together."""
    agg = af.aggregate_parking(make_parking_df())
    assert len(agg) == 2, "aggregation must not merge distinct blockfaces"
    assert set(agg["blockfacename"]) == {
        "PIKE ST BETWEEN 1ST AVE AND 2ND AVE",
        "5TH AVE BETWEEN PIKE ST AND PINE ST",
    }


def test_aggregate_parking_columns():
    agg = af.aggregate_parking(make_parking_df())
    for col in ["hour", "blockfacename", "avg_occupancy_rate",
                "peak_occupancy_rate", "total_spaces"]:
        assert col in agg.columns


def test_aggregate_parking_no_leakage():
    """turnover_proxy leaks the target and must never appear."""
    agg = af.aggregate_parking(make_parking_df())
    assert "turnover_proxy" not in agg.columns


def test_aggregate_parking_rate_is_bounded():
    agg = af.aggregate_parking(make_parking_df())
    assert agg["avg_occupancy_rate"].between(0, 1).all()
    assert agg["peak_occupancy_rate"].between(0, 1).all()


def test_aggregate_parking_rate_math():
    agg = af.aggregate_parking(make_parking_df()).set_index("blockfacename")
    # 3.5 occupied / 5 spaces = 0.70
    assert agg.loc["PIKE ST BETWEEN 1ST AVE AND 2ND AVE", "avg_occupancy_rate"] == pytest.approx(0.70)
    # 8.0 occupied / 10 spaces = 0.80
    assert agg.loc["5TH AVE BETWEEN PIKE ST AND PINE ST", "avg_occupancy_rate"] == pytest.approx(0.80)


def test_aggregate_parking_empty():
    assert af.aggregate_parking(pd.DataFrame()).empty


# ── Weather station assignment (replaces the old polygon geocoding) ──────────

def test_nearest_station_picks_closest():
    """A point sitting on Ballard should resolve to the Ballard station."""
    lat = pd.Series([47.6686, 47.5868])
    lon = pd.Series([-122.3867, -122.3331])
    got = af.nearest_station(lat, lon)
    assert got.iloc[0] == "Ballard"
    assert got.iloc[1] == "International District"


def test_nearest_station_returns_known_stations():
    lat = pd.Series([47.61, 47.65, 47.60])
    lon = pd.Series([-122.33, -122.35, -122.32])
    got = af.nearest_station(lat, lon)
    assert set(got).issubset(set(af.WEATHER_STATIONS))


# ── Time features ────────────────────────────────────────────────────────────

def test_add_time_features_peak_hours():
    df = pd.DataFrame({"hour": pd.to_datetime(
        ["2024-03-15 11:00", "2024-03-15 18:00", "2024-03-15 03:00"])})
    result = af.add_time_features(df)
    assert result["is_peak_am"].tolist() == [True, False, False]
    assert result["is_peak_pm"].tolist() == [False, True, False]


def test_add_time_features_cyclical_bounds():
    df = pd.DataFrame({"hour": pd.date_range("2024-01-01", periods=48, freq="h")})
    result = af.add_time_features(df)
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
        assert result[col].between(-1, 1).all()


def test_add_time_features_weekend():
    # 2024-03-16 is a Saturday, 2024-03-18 a Monday
    df = pd.DataFrame({"hour": pd.to_datetime(["2024-03-16 10:00", "2024-03-18 10:00"])})
    result = af.add_time_features(df)
    assert result["is_weekend"].tolist() == [True, False]


# ── Event flags ──────────────────────────────────────────────────────────────

def test_build_event_flags_no_events():
    base = af.aggregate_parking(make_parking_df())
    base["paidparkingarea"] = "Pioneer Square"
    empty = pd.DataFrame()
    result = af.build_event_flags(base, empty, empty, empty, empty)

    assert not result["is_event_day"].any()
    assert not result["has_city_event"].any()
    assert not result["has_road_closure"].any()
    assert not result["is_holiday"].any()


def test_build_event_flags_marks_matching_area_only():
    """An event at a Pioneer Square venue must not flag a Belltown blockface."""
    base = af.aggregate_parking(make_parking_df())
    base["paidparkingarea"] = ["Pioneer Square", "Belltown"]

    events = pd.DataFrame({
        "date": [pd.Timestamp("2024-03-15").date()],
        "paidparkingarea": ["Pioneer Square"],
        "event_name": ["Sounders match"],
        "venue": ["Lumen Field"],
        "genre": ["Sports"],
    })
    empty = pd.DataFrame()
    result = af.build_event_flags(base, events, empty, empty, empty)
    flags = dict(zip(result["paidparkingarea"], result["is_event_day"]))

    assert flags["Pioneer Square"] is np.True_ or flags["Pioneer Square"] == True
    assert flags["Belltown"] == False


def test_venue_area_map_uses_official_areas():
    """Venues must map to real Seattle paid parking areas, not invented ones."""
    assert af.VENUE_AREA["Climate Pledge Arena"] == "Uptown"
    assert af.VENUE_AREA["Lumen Field"] == "Pioneer Square"
