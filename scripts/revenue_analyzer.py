"""
Revenue analysis — estimates actual and potential revenue from parking occupancy data.
Outputs: revenue_summary.csv and revenue_by_region_hour.csv
Used by the Streamlit dashboard's Revenue Intelligence page.
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

# Meter operating hours per region (approximation — meters go off at 8pm in most Seattle areas)
METER_START = 8   # 8am
METER_END = 20    # 8pm

# Minutes per reading × readings per hour = 1.0 full hour per slot
HOUR_DURATION = 1.0

BASE_RATES = {
    "Downtown Seattle": 2.50,
    "Capitol Hill": 2.00,
    "South Lake Union": 2.00,
    "Ballard": 1.50,
    "Industrial District": 1.50,
}


def estimate_revenue(spaces: float, occupancy_rate: float,
                     rate: float, duration_hours: float = 1.0) -> float:
    """Revenue = rate × occupied_spaces × hours."""
    occupied = spaces * occupancy_rate
    return rate * occupied * duration_hours


def run():
    if not FEATURES_FILE.exists():
        print("features.csv not found.")
        return

    df = pd.read_csv(FEATURES_FILE, parse_dates=["hour"])
    df["hour_of_day"] = df["hour"].dt.hour
    df["date"] = df["hour"].dt.date

    # Only meter hours
    df = df[df["hour_of_day"].between(METER_START, METER_END - 1)].copy()

    df["base_rate"] = df["region"].map(BASE_RATES).fillna(2.00)
    df["current_revenue"] = df.apply(
        lambda r: estimate_revenue(r["total_spaces"], r["avg_occupancy_rate"], r["base_rate"]),
        axis=1,
    )

    # Merge recommended rates if available
    if PRICING_FILE.exists():
        recs = pd.read_csv(PRICING_FILE)[["region", "hour_of_day", "recommended_rate"]]
        df = df.merge(recs, on=["region", "hour_of_day"], how="left")
        df["recommended_rate"] = df["recommended_rate"].fillna(df["base_rate"])
    else:
        df["recommended_rate"] = df["base_rate"]

    # Apply 5% demand elasticity: -5% demand per $0.25 increase
    rate_delta = df["recommended_rate"] - df["base_rate"]
    demand_adj = 1 + (-0.05 / 0.25) * rate_delta
    df["adj_occupancy"] = (df["avg_occupancy_rate"] * demand_adj).clip(0, 1)
    df["optimized_revenue"] = df.apply(
        lambda r: estimate_revenue(r["total_spaces"], r["adj_occupancy"], r["recommended_rate"]),
        axis=1,
    )

    df["revenue_uplift"] = df["optimized_revenue"] - df["current_revenue"]

    # Daily summary
    daily = df.groupby("date").agg(
        current_revenue=("current_revenue", "sum"),
        optimized_revenue=("optimized_revenue", "sum"),
        revenue_uplift=("revenue_uplift", "sum"),
        avg_occupancy=("avg_occupancy_rate", "mean"),
        total_spaces=("total_spaces", "sum"),
    ).reset_index()
    daily["uplift_pct"] = (daily["revenue_uplift"] / daily["current_revenue"].replace(0, np.nan) * 100).round(1)

    # Regional × hourly breakdown
    region_hour = df.groupby(["region", "hour_of_day"]).agg(
        avg_occupancy=("avg_occupancy_rate", "mean"),
        current_revenue_per_hour=("current_revenue", "mean"),
        optimized_revenue_per_hour=("optimized_revenue", "mean"),
        avg_spaces=("total_spaces", "mean"),
        current_rate=("base_rate", "mean"),
        recommended_rate=("recommended_rate", "mean"),
    ).reset_index()
    region_hour["revenue_uplift"] = region_hour["optimized_revenue_per_hour"] - region_hour["current_revenue_per_hour"]

    daily.to_csv(REVENUE_SUMMARY_FILE, index=False)
    region_hour.to_csv(REVENUE_DETAIL_FILE, index=False)

    total_current = daily["current_revenue"].sum()
    total_optimized = daily["optimized_revenue"].sum()
    print(f"Revenue analysis: {len(daily)} days")
    print(f"  Total current revenue:   ${total_current:,.2f}")
    print(f"  Total optimized revenue: ${total_optimized:,.2f}")
    print(f"  Potential uplift:        ${total_optimized - total_current:+,.2f} ({(total_optimized/max(total_current,0.01)-1)*100:+.1f}%)")
    print(f"Saved → {REVENUE_SUMMARY_FILE}")
    print(f"Saved → {REVENUE_DETAIL_FILE}")

    return daily, region_hour


if __name__ == "__main__":
    run()
