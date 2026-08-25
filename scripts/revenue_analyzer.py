"""
Utilization and revenue-opportunity analysis, per official parking area.

WHY THERE ARE NO DOLLAR TOTALS HERE
    Seattle publishes no per-meter posted rate (the `paidparkingrate` column in
    the occupancy feed is populated on 0 of 25.2M rows). Any absolute dollar
    figure would therefore rest on invented rates. This module instead reports
    the quantity that IS measured — occupied space-hours — plus revenue
    expressed per $1.00 of posted rate, which the city multiplies by the real
    posted rate to get dollars.

    space-hours          one paid space occupied for one hour
    revenue_per_dollar   revenue generated per $1.00/hr of posted rate
                         (numerically equal to occupied space-hours)

Outputs
    data/revenue_summary.csv        daily citywide utilization
    data/revenue_by_area_hour.csv   area x hour-of-day breakdown
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

FEATURES_FILE = DATA_DIR / "features.parquet"
REGISTRY_FILE = DATA_DIR / "meter_registry.csv"
REVENUE_SUMMARY_FILE = DATA_DIR / "revenue_summary.csv"
REVENUE_DETAIL_FILE = DATA_DIR / "revenue_by_area_hour.csv"

# Paid parking hours (the feed only carries paid periods anyway)
METER_START = 8
METER_END = 20

# City utilization target — midpoint of the SMC 11.16.121 70-85% band
TARGET_OCC = 0.80


def run():
    if not FEATURES_FILE.exists():
        print(f"{FEATURES_FILE.name} not found — run aggregate_features.py first.")
        return
    if not REGISTRY_FILE.exists():
        print("meter_registry.csv not found — run fetch_meter_registry.py first.")
        return

    registry = pd.read_csv(REGISTRY_FILE)[["blockfacename", "spaces"]]

    df = pd.read_parquet(
        FEATURES_FILE,
        columns=["hour", "blockfacename", "paidparkingarea",
                 "avg_occupancy_rate", "hour_of_day"],
    )
    df = df[df["hour_of_day"].between(METER_START, METER_END - 1)].copy()
    df["date"] = df["hour"].dt.date

    # Authoritative capacity from the registry (the feed averages rather than
    # sums a blockface's meter keys, understating capacity by ~38%)
    df = df.merge(registry, on="blockfacename", how="inner")

    # The measurable quantities
    df["occupied_space_hours"] = df["avg_occupancy_rate"] * df["spaces"]
    df["target_space_hours"] = TARGET_OCC * df["spaces"]
    df["unsold_space_hours"] = (df["target_space_hours"] - df["occupied_space_hours"]).clip(lower=0)

    daily = df.groupby("date").agg(
        occupied_space_hours=("occupied_space_hours", "sum"),
        target_space_hours=("target_space_hours", "sum"),
        unsold_space_hours=("unsold_space_hours", "sum"),
        capacity_space_hours=("spaces", "sum"),
        avg_occupancy=("avg_occupancy_rate", "mean"),
        n_blockfaces=("blockfacename", "nunique"),
    ).reset_index()

    daily["utilization_pct"] = (
        daily["occupied_space_hours"] / daily["capacity_space_hours"].replace(0, np.nan) * 100
    ).round(1)
    daily["gap_to_target_pct"] = (
        daily["unsold_space_hours"] / daily["target_space_hours"].replace(0, np.nan) * 100
    ).round(1)
    # Revenue per $1.00/hr of posted rate == occupied space-hours
    daily["revenue_per_posted_dollar"] = daily["occupied_space_hours"].round(1)

    area_hour = df.groupby(["paidparkingarea", "hour_of_day"]).agg(
        avg_occupancy=("avg_occupancy_rate", "mean"),
        occupied_space_hours=("occupied_space_hours", "mean"),
        target_space_hours=("target_space_hours", "mean"),
        unsold_space_hours=("unsold_space_hours", "mean"),
        spaces=("spaces", "mean"),
        n_blockfaces=("blockfacename", "nunique"),
    ).reset_index()
    area_hour["revenue_per_posted_dollar"] = area_hour["occupied_space_hours"].round(2)

    for c in ["avg_occupancy"]:
        daily[c] = daily[c].round(4)
        area_hour[c] = area_hour[c].round(4)

    daily.to_csv(REVENUE_SUMMARY_FILE, index=False)
    area_hour.to_csv(REVENUE_DETAIL_FILE, index=False)

    occ = daily["occupied_space_hours"].sum()
    tgt = daily["target_space_hours"].sum()
    unsold = daily["unsold_space_hours"].sum()

    print(f"Utilization analysis: {len(daily):,} days, "
          f"{df['blockfacename'].nunique():,} blockfaces, "
          f"{df['paidparkingarea'].nunique()} areas")
    print(f"  Occupied      : {occ:>14,.0f} space-hours "
          f"({daily['utilization_pct'].mean():.1f}% of capacity)")
    print(f"  At 80% target : {tgt:>14,.0f} space-hours")
    print(f"  Unsold vs tgt : {unsold:>14,.0f} space-hours "
          f"({unsold / max(tgt, 1) * 100:.1f}% short)")
    print()
    print(f"  Revenue is reported per $1.00/hr of posted rate, since Seattle")
    print(f"  publishes no per-meter rates. Multiply by the real posted rate:")
    print(f"    e.g. at $2.00/hr → ${occ * 2:,.0f} earned, "
          f"${unsold * 2:,.0f} foregone vs target")
    print(f"Saved → {REVENUE_SUMMARY_FILE.name}, {REVENUE_DETAIL_FILE.name}")

    return daily, area_hour


if __name__ == "__main__":
    run()
