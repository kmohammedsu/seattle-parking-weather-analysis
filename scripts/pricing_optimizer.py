"""
Per-blockface pricing recommendations under Seattle Municipal Code 11.16.121.
Performance-Based Parking Pricing — the city may adjust meter rates by demand.

Target: 70–85% occupancy per blockface (standard SFMTA/Seattle benchmark).
    below 70%  → reduce rate to attract parkers
    above 85%  → raise rate to free up spaces
    within     → hold

RATE BASELINE
    Seattle does not publish per-meter rates anywhere in its open data. The
    `paidparkingrate` column exists in the occupancy feed but is populated on
    0 of 25.2M rows, and no separate rate dataset exists. Rather than invent a
    baseline, this script outputs a RELATIVE ADJUSTMENT — "+$0.50/hr vs the
    currently posted rate" — which the city applies to whatever rate is
    actually posted at that blockface. Nothing here is fabricated.

Outputs
    data/meter_recommendations.csv   blockface x hour recommendations (committed)
    data/meter_summary.csv           one row per blockface, for the map (committed)
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

FEATURES_FILE = DATA_DIR / "features.parquet"
REGISTRY_FILE = DATA_DIR / "meter_registry.csv"
MODEL_FILE = MODELS_DIR / "parking_demand_lgbm.pkl"
FEATURE_LIST_FILE = MODELS_DIR / "feature_columns.json"

RECS_FILE = DATA_DIR / "meter_recommendations.csv"
SUMMARY_FILE = DATA_DIR / "meter_summary.csv"

# SMC 11.16.121 rate bounds (USD/hour) — constrain the *magnitude* of change
RATE_MIN = 0.50
RATE_MAX = 8.00
RATE_STEP = 0.25
MAX_ADJUSTMENT = 2.00   # largest single recommended move, in either direction

TARGET_LOW = 0.70
TARGET_HIGH = 0.85

# Demand response: roughly -5% demand per +$0.25/hr (conservative, SFpark-style)
ELASTICITY_PER_DOLLAR = -0.05 / 0.25

# How recent a window to base recommendations on
RECENT_DAYS = 90


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_FILE}. Run train_model.py first.")
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


def load_feature_cols():
    if not FEATURE_LIST_FILE.exists():
        raise FileNotFoundError("feature_columns.json missing. Run train_model.py first.")
    return json.loads(FEATURE_LIST_FILE.read_text())


def rate_adjustment(occupancy: float) -> float:
    """Recommended change to the posted rate, in dollars/hour.

    Returns a delta (not an absolute rate) because no per-meter posted rate
    exists in Seattle's open data — see module docstring.
    """
    if np.isnan(occupancy):
        return 0.0

    if occupancy < TARGET_LOW:
        steps = max(1, int((TARGET_LOW - occupancy) / 0.05))
        delta = -steps * RATE_STEP
    elif occupancy > TARGET_HIGH:
        steps = max(1, int((occupancy - TARGET_HIGH) / 0.05))
        delta = steps * RATE_STEP
    else:
        return 0.0

    return round(float(np.clip(delta, -MAX_ADJUSTMENT, MAX_ADJUSTMENT)), 2)


def classify(occupancy: float) -> str:
    if np.isnan(occupancy):
        return "unknown"
    if occupancy > TARGET_HIGH:
        return "increase"
    if occupancy < TARGET_LOW:
        return "decrease"
    return "hold"


def demand_response(rate_delta: float, spaces: float, occupancy: float) -> dict:
    """Projected occupancy and per-hour revenue change *per dollar of posted
    rate*, since the posted rate itself is unknown.

    `revenue_delta_per_posted_dollar` is the change in hourly revenue for each
    $1.00 of currently posted rate — the city multiplies by the real rate.
    """
    occ_change = ELASTICITY_PER_DOLLAR * rate_delta
    projected_occ = float(np.clip(occupancy * (1 + occ_change), 0, 1))

    occupied_now = occupancy * spaces
    occupied_new = projected_occ * spaces

    # Revenue is rate x occupied spaces. Expressed per $1 of posted rate:
    #   now = 1.00 * occupied_now ; after = (1.00 + delta) * occupied_new
    revenue_now = occupied_now
    revenue_after = (1.0 + rate_delta) * occupied_new

    return {
        "projected_occupancy": round(projected_occ, 3),
        "occupancy_change_pts": round((projected_occ - occupancy) * 100, 1),
        "revenue_delta_per_posted_dollar": round(revenue_after - revenue_now, 3),
    }


def load_recent() -> pd.DataFrame:
    df = pd.read_parquet(
        FEATURES_FILE,
        columns=["hour", "blockfacename", "paidparkingarea", "paidparkingsubarea",
                 "avg_occupancy_rate", "total_spaces", "hour_of_day",
                 "lat", "lon", "time_limit", "n_meters"],
    )
    cutoff = df["hour"].max() - pd.Timedelta(days=RECENT_DAYS)
    recent = df[df["hour"] >= cutoff]
    print(f"  Recent window: {cutoff.date()} → {df['hour'].max().date()} "
          f"({len(recent):,} rows)")
    return recent


def build_recommendations(recent: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """One recommendation per blockface x hour-of-day.

    Capacity comes from the registry, not the occupancy feed. The historical
    feed aggregated `avg(parkingspacecount)` across each blockface's meter
    keys rather than summing them, understating true capacity by ~38%. The
    occupancy *rate* is unaffected (it is an average over the same keys), but
    any absolute space count must come from the registry.
    """
    grouped = (
        recent.groupby(["blockfacename", "paidparkingarea", "hour_of_day"], observed=True)
        .agg(
            avg_occupancy=("avg_occupancy_rate", "mean"),
            spaces_observed=("total_spaces", "mean"),
            n_observations=("avg_occupancy_rate", "size"),
        )
        .reset_index()
        .merge(registry[["blockfacename", "spaces"]], on="blockfacename", how="left")
    )
    grouped["spaces"] = grouped["spaces"].fillna(grouped["spaces_observed"])

    grouped["rate_adjustment"] = grouped["avg_occupancy"].apply(rate_adjustment)
    grouped["action"] = grouped["avg_occupancy"].apply(classify)

    response = grouped.apply(
        lambda r: demand_response(r["rate_adjustment"], r["spaces"], r["avg_occupancy"]),
        axis=1, result_type="expand",
    )
    grouped = pd.concat([grouped, response], axis=1)

    grouped["avg_occupancy"] = grouped["avg_occupancy"].round(3)
    grouped["spaces"] = grouped["spaces"].round(1)
    return grouped.sort_values(["paidparkingarea", "blockfacename", "hour_of_day"])


def build_summary(recs: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """One row per blockface — powers the map and the meter drill-down."""
    summary = (
        recs.groupby(["blockfacename", "paidparkingarea"], observed=True)
        .agg(
            avg_occupancy=("avg_occupancy", "mean"),
            peak_hour_occupancy=("avg_occupancy", "max"),
            spaces=("spaces", "mean"),
            hours_over_target=("action", lambda s: int((s == "increase").sum())),
            hours_under_target=("action", lambda s: int((s == "decrease").sum())),
            hours_on_target=("action", lambda s: int((s == "hold").sum())),
            mean_adjustment=("rate_adjustment", "mean"),
            max_adjustment=("rate_adjustment", "max"),
            min_adjustment=("rate_adjustment", "min"),
        )
        .reset_index()
    )

    # Peak hour per blockface
    peak = recs.loc[recs.groupby("blockfacename", observed=True)["avg_occupancy"].idxmax()]
    summary = summary.merge(
        peak[["blockfacename", "hour_of_day"]].rename(columns={"hour_of_day": "peak_hour"}),
        on="blockfacename", how="left",
    )

    summary["primary_action"] = summary["avg_occupancy"].apply(classify)
    summary = summary.merge(
        registry[["blockfacename", "paidparkingsubarea", "lat", "lon",
                  "time_limit", "n_meters"]],
        on="blockfacename", how="left",
    )

    for c in ["avg_occupancy", "peak_hour_occupancy", "mean_adjustment"]:
        summary[c] = summary[c].round(3)
    summary["spaces"] = summary["spaces"].round(1)
    return summary.sort_values(["paidparkingarea", "blockfacename"])


def run():
    if not FEATURES_FILE.exists():
        print(f"{FEATURES_FILE.name} not found — run aggregate_features.py first.")
        return
    if not REGISTRY_FILE.exists():
        print("meter_registry.csv not found — run fetch_meter_registry.py first.")
        return

    registry = pd.read_csv(REGISTRY_FILE)
    recent = load_recent()
    if recent.empty:
        print("No recent data to price.")
        return

    recs = build_recommendations(recent, registry)
    summary = build_summary(recs, registry)

    recs.to_csv(RECS_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    counts = summary["primary_action"].value_counts()
    print(f"\nRecommendations: {len(recs):,} blockface-hours across "
          f"{summary['blockfacename'].nunique():,} blockfaces")
    print(f"  Raise rate : {counts.get('increase', 0):>4} blockfaces (over 85% occupancy)")
    print(f"  Lower rate : {counts.get('decrease', 0):>4} blockfaces (under 70%)")
    print(f"  Hold       : {counts.get('hold', 0):>4} blockfaces (in 70-85% target band)")
    print(f"\n  Rate changes are RELATIVE to each blockface's posted rate")
    print(f"  (Seattle publishes no per-meter rate data — see module docstring)")
    print(f"\nSaved → {RECS_FILE.name}, {SUMMARY_FILE.name}")
    return recs


if __name__ == "__main__":
    run()
