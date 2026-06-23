"""
Revenue analysis — estimates actual and potential revenue from parking occupancy data.

Three scenarios:
  current   — actual rates × actual occupancy (what the city earns today)
  target    — actual rates × 80% occupancy (revenue if utilization were at target)
  optimized — recommended rates × 80% occupancy (full pricing + utilization upside)

Uplift is always positive: the city is always leaving money on the table relative
to achieving target utilization with optimal rates.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

FEATURES_FILE = DATA_DIR / "features.csv"
PRICING_FILE = DATA_DIR / "pricing_recommendations.csv"
REVENUE_SUMMARY_FILE = DATA_DIR / "revenue_summary.csv"
REVENUE_DETAIL_FILE = DATA_DIR / "revenue_by_region_hour.csv"

METER_START = 8
METER_END = 20
HOUR_DURATION = 1.0
TARGET_OCC = 0.80  # midpoint of 70–85% target band

BASE_RATES = {
    "Downtown Seattle":    2.50,
    "Capitol Hill":        2.00,
    "South Lake Union":    2.00,
    "Ballard":             1.50,
    "Industrial District": 1.50,
}


def revenue(spaces: float, occ: float, rate: float) -> float:
    return rate * spaces * occ * HOUR_DURATION


def run():
    if not FEATURES_FILE.exists():
        print("features.csv not found.")
        return

    df = pd.read_csv(FEATURES_FILE, parse_dates=["hour"])
    df["hour_of_day"] = df["hour"].dt.hour
    df["date"] = df["hour"].dt.date

    df = df[df["hour_of_day"].between(METER_START, METER_END - 1)].copy()

    df["base_rate"] = df["region"].map(BASE_RATES).fillna(2.00)

    # Merge recommended rates
    if PRICING_FILE.exists():
        recs = pd.read_csv(PRICING_FILE)[["region", "hour_of_day", "recommended_rate"]]
        df = df.merge(recs, on=["region", "hour_of_day"], how="left")
        df["recommended_rate"] = df["recommended_rate"].fillna(df["base_rate"])
    else:
        df["recommended_rate"] = df["base_rate"]

    # Three scenarios — all use the same space counts
    df["current_revenue"] = df.apply(
        lambda r: revenue(r["total_spaces"], r["avg_occupancy_rate"], r["base_rate"]), axis=1
    )
    # Target: what current rates would earn if occupancy hit 80% (utilization opportunity)
    df["target_revenue"] = df.apply(
        lambda r: revenue(r["total_spaces"], TARGET_OCC, r["base_rate"]), axis=1
    )
    # Optimized: apply rate INCREASES only (rate decreases serve traffic flow, not revenue)
    # In under-target zones, keep current rate; in over-target zones, apply recommended (higher) rate
    df["revenue_rate"] = df[["base_rate", "recommended_rate"]].max(axis=1)
    df["optimized_revenue"] = df.apply(
        lambda r: revenue(r["total_spaces"], TARGET_OCC, r["revenue_rate"]), axis=1
    )

    df["revenue_uplift"] = df["optimized_revenue"] - df["current_revenue"]
    df["utilization_gap"] = df["target_revenue"] - df["current_revenue"]

    # Daily summary
    daily = df.groupby("date").agg(
        current_revenue=("current_revenue", "sum"),
        target_revenue=("target_revenue", "sum"),
        optimized_revenue=("optimized_revenue", "sum"),
        revenue_uplift=("revenue_uplift", "sum"),
        utilization_gap=("utilization_gap", "sum"),
        avg_occupancy=("avg_occupancy_rate", "mean"),
        total_spaces=("total_spaces", "sum"),
    ).reset_index()
    daily["uplift_pct"] = (
        daily["revenue_uplift"] / daily["current_revenue"].replace(0, np.nan) * 100
    ).round(1)

    # Regional × hourly breakdown
    region_hour = df.groupby(["region", "hour_of_day"]).agg(
        avg_occupancy=("avg_occupancy_rate", "mean"),
        current_revenue_per_hour=("current_revenue", "mean"),
        target_revenue_per_hour=("target_revenue", "mean"),
        optimized_revenue_per_hour=("optimized_revenue", "mean"),
        avg_spaces=("total_spaces", "mean"),
        current_rate=("base_rate", "mean"),
        recommended_rate=("recommended_rate", "mean"),
    ).reset_index()
    region_hour["revenue_uplift"] = (
        region_hour["optimized_revenue_per_hour"] - region_hour["current_revenue_per_hour"]
    )

    daily.to_csv(REVENUE_SUMMARY_FILE, index=False)
    region_hour.to_csv(REVENUE_DETAIL_FILE, index=False)

    total_current = daily["current_revenue"].sum()
    total_target = daily["target_revenue"].sum()
    total_optimized = daily["optimized_revenue"].sum()
    print(f"Revenue analysis: {len(daily)} days")
    print(f"  Current revenue:   ${total_current:,.2f}  (actual rates × actual occupancy)")
    print(f"  Target revenue:    ${total_target:,.2f}  (actual rates × 80% occupancy)")
    print(f"  Optimized revenue: ${total_optimized:,.2f}  (optimal rates × 80% occupancy)")
    print(f"  Full uplift:       ${total_optimized - total_current:+,.2f} "
          f"({(total_optimized / max(total_current, 0.01) - 1) * 100:+.1f}%)")
    print(f"Saved → {REVENUE_SUMMARY_FILE}, {REVENUE_DETAIL_FILE}")

    return daily, region_hour


if __name__ == "__main__":
    run()
