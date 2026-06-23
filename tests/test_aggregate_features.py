"""Tests for aggregate_features.py — the core feature engineering step."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import aggregate_features as af


def make_parking_df():
    return pd.DataFrame({
        "occupancy_date": pd.to_datetime(["2024-03-15", "2024-03-15"]),
        "occupancy_hour": [10, 10],
        "blockfacename": ["PIKE ST BETWEEN 1ST AVE AND 2ND AVE",
                          "DOWNTOWN 5TH AVE BETWEEN PIKE AND PINE"],
        "avg_occupied": [3.5, 8.0],
        "peak_occupied": [5.0, 10.0],
        "avg_spaces": [5.0, 10.0],
        "occupancy_rate": [0.70, 0.80],
        "peak_occupancy_rate": [1.0, 1.0],
    })


def test_map_blockface_to_region_downtown():
    series = pd.Series(["DOWNTOWN 5TH AVE", "PIKE ST"])
    result = af.map_blockface_to_region(series)
    assert result.iloc[0] == "Downtown Seattle"


def test_map_blockface_to_region_capitol_hill():
    series = pd.Series(["CAPITOL HILL BROADWAY"])
    result = af.map_blockface_to_region(series)
    assert result.iloc[0] == "Capitol Hill"


def test_aggregate_parking_no_leakage():
    """turnover_proxy must not appear in aggregate_parking output."""
    parking = make_parking_df()
    parking["hour"] = parking["occupancy_date"] + pd.to_timedelta(parking["occupancy_hour"], unit="h")
    parking["region"] = af.map_blockface_to_region(parking["blockfacename"])
    agg = af.aggregate_parking(parking)
    assert "turnover_proxy" not in agg.columns, "turnover_proxy causes data leakage — must not exist"


def test_aggregate_parking_columns():
    parking = make_parking_df()
    parking["hour"] = parking["occupancy_date"] + pd.to_timedelta(parking["occupancy_hour"], unit="h")
    parking["region"] = af.map_blockface_to_region(parking["blockfacename"])
    agg = af.aggregate_parking(parking)
    required = {"hour", "region", "avg_occupancy_rate", "total_spaces", "total_occupied"}
    assert required.issubset(set(agg.columns))


def test_add_time_features_peak_hours():
    """Verify updated peak hours reflect actual Seattle meter patterns (not 7-9am)."""
    df = pd.DataFrame({"hour": pd.to_datetime(["2024-06-15 10:00", "2024-06-15 08:00",
                                                 "2024-06-15 17:00", "2024-06-15 07:00"])})
    result = af.add_time_features(df)
    assert result.loc[0, "is_peak_am"] == True,  "10am should be peak AM"
    assert result.loc[1, "is_peak_am"] == False, "8am should NOT be peak AM (old assumption)"
    assert result.loc[2, "is_peak_pm"] == True,  "5pm should be peak PM"
    assert result.loc[3, "is_peak_pm"] == False, "7am should NOT be peak PM"


def test_add_time_features_cyclical_bounds():
    """sin/cos encoding must stay in [-1, 1]."""
    df = pd.DataFrame({"hour": pd.date_range("2024-01-01", periods=24, freq="h")})
    result = af.add_time_features(df)
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
        assert result[col].between(-1, 1).all(), f"{col} out of [-1,1] bounds"


def test_build_event_flags_no_events():
    """With no events/permits/holidays/closures, flags should all be False/0."""
    base = pd.DataFrame({
        "hour": pd.to_datetime(["2024-06-15 10:00", "2024-06-15 11:00"]),
        "region": ["Downtown Seattle", "Ballard"],
    })
    result = af.build_event_flags(base, pd.DataFrame(), pd.DataFrame(),
                                   pd.DataFrame(), pd.DataFrame())
    assert (result["is_event_day"] == False).all()
    assert (result["has_city_event"] == False).all()
    assert (result["is_holiday"] == False).all()
    assert (result["has_road_closure"] == False).all()
