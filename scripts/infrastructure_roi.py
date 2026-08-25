"""
Infrastructure ROI calculator for Seattle paid parking, per official area.

Given an area's observed occupancy, evaluates whether adding capacity (surface
lot / structured garage / underground) is financially justified.

RATE HANDLING
    Seattle publishes no per-meter posted rate, so this module does not assume
    one. Instead it inverts the question and solves for the BREAKEVEN RATE —
    the hourly rate required to cover debt service plus operating costs at the
    80% target occupancy. The city compares that figure against what it
    actually charges (or could charge) in that area.

    A project is flagged viable when two conditions hold:
      1. observed demand already exceeds the target occupancy the model assumes
      2. the breakeven rate is within the SMC 11.16.121 bound of $8.00/hr

Outputs
    data/infrastructure_roi.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

REVENUE_DETAIL_FILE = DATA_DIR / "revenue_by_area_hour.csv"
OUTPUT_FILE = DATA_DIR / "infrastructure_roi.csv"

# Construction cost estimates (Seattle 2024 benchmarks, USD)
COST_ESTIMATES = {
    "surface_lot_per_space": 5_000,          # pave, stripe, lighting
    "structured_garage_per_space": 45_000,   # above-ground precast
    "underground_per_space": 90_000,         # underground in urban core
}

OPERATING_COST_PER_SPACE_YEAR = 800   # maintenance, security, management

BOND_RATE = 0.045      # 4.5% municipal bond
BOND_TERM_YEARS = 20

METER_HOURS_PER_DAY = 12   # 8am-8pm metered
DAYS_PER_YEAR = 313        # ~365 less Sundays (free) and major holidays

TARGET_OCCUPANCY = 0.80    # midpoint of the SMC 70-85% band
RATE_CEILING = 8.00        # SMC 11.16.121 maximum

SCENARIO_SPACES = [50, 100, 250]


def annual_debt_service(cost_per_space: float, n_spaces: int) -> float:
    """Annual payment on a municipal bond financing the build."""
    P = cost_per_space * n_spaces
    r = BOND_RATE / 12
    n = BOND_TERM_YEARS * 12
    monthly = P * r * (1 + r) ** n / ((1 + r) ** n - 1)
    return monthly * 12


def breakeven_analysis(area: str, avg_occupancy: float, current_spaces: float,
                       n_new_spaces: int, infra_type: str) -> dict:
    cost_per_space = COST_ESTIMATES[f"{infra_type}_per_space"]
    total_cost = cost_per_space * n_new_spaces

    debt = annual_debt_service(cost_per_space, n_new_spaces)
    ops = OPERATING_COST_PER_SPACE_YEAR * n_new_spaces
    total_annual_cost = debt + ops

    # Paid space-hours the new capacity would sell at target occupancy
    annual_space_hours = (
        n_new_spaces * TARGET_OCCUPANCY * METER_HOURS_PER_DAY * DAYS_PER_YEAR
    )

    # The rate required to cover all annual costs
    breakeven_rate = total_annual_cost / max(annual_space_hours, 1)

    # Revenue per $1.00/hr of posted rate, so the city can scale it themselves
    revenue_per_posted_dollar = annual_space_hours

    demand_supports = avg_occupancy >= TARGET_OCCUPANCY
    rate_feasible = breakeven_rate <= RATE_CEILING

    return {
        "paidparkingarea": area,
        "infra_type": infra_type,
        "n_new_spaces": n_new_spaces,
        "current_occupancy": round(avg_occupancy, 3),
        "current_spaces": round(current_spaces, 0),
        "total_construction_cost": round(total_cost, 0),
        "annual_debt_service": round(debt, 0),
        "annual_operating_cost": round(ops, 0),
        "total_annual_cost": round(total_annual_cost, 0),
        "annual_space_hours_at_target": round(annual_space_hours, 0),
        "revenue_per_posted_dollar": round(revenue_per_posted_dollar, 0),
        "breakeven_rate_per_hour": round(breakeven_rate, 2),
        "rate_within_legal_cap": bool(rate_feasible),
        "demand_supports_expansion": bool(demand_supports),
        "viable": bool(demand_supports and rate_feasible),
    }


def run():
    if not REVENUE_DETAIL_FILE.exists():
        print(f"{REVENUE_DETAIL_FILE.name} not found — run revenue_analyzer.py first.")
        return

    detail = pd.read_csv(REVENUE_DETAIL_FILE)
    meter_hours = detail[detail["hour_of_day"].between(8, 19)]

    area_stats = meter_hours.groupby("paidparkingarea").agg(
        avg_occupancy=("avg_occupancy", "mean"),
        avg_spaces=("spaces", "mean"),
        n_blockfaces=("n_blockfaces", "max"),
    ).reset_index()

    rows = []
    for _, row in area_stats.iterrows():
        for infra_type in ["surface_lot", "structured_garage", "underground"]:
            for n_spaces in SCENARIO_SPACES:
                rows.append(breakeven_analysis(
                    area=row["paidparkingarea"],
                    avg_occupancy=row["avg_occupancy"],
                    current_spaces=row["avg_spaces"],
                    n_new_spaces=n_spaces,
                    infra_type=infra_type,
                ))

    df = pd.DataFrame(rows).sort_values(
        ["viable", "breakeven_rate_per_hour"], ascending=[False, True]
    )
    df.to_csv(OUTPUT_FILE, index=False)

    viable = df[df["viable"]]
    print(f"Infrastructure ROI: {len(df)} scenarios across "
          f"{df['paidparkingarea'].nunique()} areas")
    print(f"  Viable (demand supports expansion AND breakeven rate under "
          f"${RATE_CEILING:.2f}): {len(viable)}")

    if viable.empty:
        best = df.nsmallest(5, "breakeven_rate_per_hour")
        print("\n  No area currently has demand at or above the 80% target, so no")
        print("  expansion is justified on utilization grounds. Lowest breakeven")
        print("  rates (cheapest to justify if demand rises):")
    else:
        best = viable.nsmallest(5, "breakeven_rate_per_hour")
        print("\n  Lowest breakeven rate (easiest to justify):")

    cols = ["paidparkingarea", "infra_type", "n_new_spaces",
            "current_occupancy", "breakeven_rate_per_hour"]
    print(best[cols].to_string(index=False))
    print(f"\nSaved → {OUTPUT_FILE.name}")
    return df


if __name__ == "__main__":
    run()
