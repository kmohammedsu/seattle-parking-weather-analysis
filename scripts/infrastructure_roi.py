"""
Infrastructure ROI Calculator for Seattle Parking.

Given a region's current occupancy patterns, calculates whether investing in
additional parking capacity (garage construction or surface lot) is financially
justified, and at what demand level it breaks even.

Uses actual revenue data from revenue_analyzer.py outputs.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

REVENUE_DETAIL_FILE = DATA_DIR / "revenue_by_region_hour.csv"
FEATURES_FILE = DATA_DIR / "features.csv"
OUTPUT_FILE = DATA_DIR / "infrastructure_roi.csv"

# Construction cost estimates (Seattle 2024 benchmarks, USD)
COST_ESTIMATES = {
    "surface_lot_per_space": 5_000,        # pave, stripe, lighting
    "structured_garage_per_space": 45_000, # above-ground precast
    "underground_garage_per_space": 90_000,# underground in urban core
}

# Operating costs per space per year
OPERATING_COST_PER_SPACE_YEAR = 800  # maintenance, security, management

# Financing
BOND_RATE = 0.045     # 4.5% municipal bond rate
BOND_TERM_YEARS = 20  # standard municipal infrastructure bond

METER_HOURS_PER_DAY = 12   # 8am-8pm typical Seattle metered hours
DAYS_PER_YEAR = 365

# Target occupancy range (SMC 11.16.121)
TARGET_OCCUPANCY = 0.80  # midpoint of 70-85% target band


def annual_debt_service(cost_per_space: float, n_spaces: int) -> float:
    """Annual payment on a municipal bond financing the infrastructure."""
    P = cost_per_space * n_spaces
    r = BOND_RATE / 12  # monthly rate
    n = BOND_TERM_YEARS * 12
    # Standard amortization formula
    monthly = P * r * (1 + r) ** n / ((1 + r) ** n - 1)
    return monthly * 12


def breakeven_analysis(region: str, avg_occupancy: float, avg_spaces: float,
                       current_rate: float, n_new_spaces: int,
                       infra_type: str = "structured_garage") -> dict:
    """Calculate ROI for adding n_new_spaces in a given region."""
    cost_per_space = COST_ESTIMATES[f"{infra_type}_per_space"]
    total_cost = cost_per_space * n_new_spaces

    # Annual debt service on construction
    debt_service = annual_debt_service(cost_per_space, n_new_spaces)

    # Annual operating cost
    annual_ops = OPERATING_COST_PER_SPACE_YEAR * n_new_spaces

    total_annual_cost = debt_service + annual_ops

    # Revenue from new spaces at target occupancy
    # Assume new spaces fill to target occupancy over time
    annual_revenue = (
        current_rate
        * n_new_spaces
        * TARGET_OCCUPANCY
        * METER_HOURS_PER_DAY
        * DAYS_PER_YEAR
    )

    net_annual = annual_revenue - total_annual_cost
    roi_pct = net_annual / total_cost * 100

    # Breakeven occupancy (what occupancy do we need to cover all costs?)
    breakeven_occ = total_annual_cost / (
        current_rate * n_new_spaces * METER_HOURS_PER_DAY * DAYS_PER_YEAR
    )

    # Simple payback period at target occupancy
    payback_years = total_cost / max(net_annual, 0.01) if net_annual > 0 else float("inf")

    return {
        "region": region,
        "infra_type": infra_type,
        "n_new_spaces": n_new_spaces,
        "current_occupancy": round(avg_occupancy, 3),
        "current_spaces": round(avg_spaces, 0),
        "current_rate": current_rate,
        "total_construction_cost": round(total_cost, 0),
        "annual_debt_service": round(debt_service, 0),
        "annual_operating_cost": round(annual_ops, 0),
        "annual_revenue_at_target": round(annual_revenue, 0),
        "net_annual_income": round(net_annual, 0),
        "roi_percent": round(roi_pct, 2),
        "breakeven_occupancy": round(breakeven_occ, 3),
        "viable": breakeven_occ < avg_occupancy,
        "payback_years": round(payback_years, 1) if payback_years != float("inf") else None,
    }


def run():
    if not REVENUE_DETAIL_FILE.exists():
        print("revenue_by_region_hour.csv not found. Run revenue_analyzer.py first.")
        return

    detail = pd.read_csv(REVENUE_DETAIL_FILE)

    # Get per-region averages (across meter hours only: 8am-8pm)
    meter_hours = detail[detail["hour_of_day"].between(8, 19)]
    region_stats = meter_hours.groupby("region").agg(
        avg_occupancy=("avg_occupancy", "mean"),
        avg_spaces=("avg_spaces", "mean"),
        current_rate=("current_rate", "mean"),
    ).reset_index()

    results = []
    for _, row in region_stats.iterrows():
        for infra_type in ["surface_lot", "structured_garage"]:
            for n_spaces in [50, 100, 250]:
                result = breakeven_analysis(
                    region=row["region"],
                    avg_occupancy=row["avg_occupancy"],
                    avg_spaces=row["avg_spaces"],
                    current_rate=row["current_rate"],
                    n_new_spaces=n_spaces,
                    infra_type=infra_type,
                )
                results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Infrastructure ROI analysis: {len(df)} scenarios")
    viable = df[df["viable"]]
    print(f"Viable scenarios (breakeven < current occupancy): {len(viable)}")
    if not viable.empty:
        print("\nTop viable investments by ROI%:")
        top = viable.nlargest(5, "roi_percent")[
            ["region", "infra_type", "n_new_spaces", "roi_percent", "payback_years"]
        ]
        print(top.to_string(index=False))
    print(f"\nSaved → {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    run()
