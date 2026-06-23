"""
Dynamic pricing recommendations under Seattle Municipal Code 11.16.121.
Performance-Based Parking Pricing — city can adjust meter rates based on demand.

Target: 70–85% occupancy per block (standard SFMTA/Seattle benchmark).
Below 70% → reduce rate to stimulate demand.
Above 85% → increase rate to free up spaces.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUT_FILE = DATA_DIR / "pricing_recommendations.csv"

FEATURES_FILE = DATA_DIR / "features.csv"
MODEL_FILE = MODELS_DIR / "parking_demand_lgbm.pkl"
FEATURE_LIST_FILE = MODELS_DIR / "feature_columns.json"

# Seattle SMC 11.16.121 rate bounds (USD/hour)
RATE_MIN = 0.50
RATE_MAX = 8.00
RATE_STEP = 0.25

# Occupancy targets
TARGET_LOW = 0.70   # below this → reduce price
TARGET_HIGH = 0.85  # above this → increase price

# Current base rates by area (approximate — city publishes these)
BASE_RATES = {
    "Downtown Seattle": 2.50,
    "Capitol Hill": 2.00,
    "South Lake Union": 2.00,
    "Ballard": 1.50,
    "Industrial District": 1.50,
}


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_FILE}. Run train_model.py first.")
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


def load_feature_cols():
    if not FEATURE_LIST_FILE.exists():
        raise FileNotFoundError("feature_columns.json missing. Run train_model.py first.")
    return json.loads(FEATURE_LIST_FILE.read_text())


def recommend_rate(occupancy: float, base_rate: float) -> float:
    """Step rate up/down to push occupancy toward 70-85% target band."""
    if occupancy < TARGET_LOW:
        # How many steps below target?
        gap = TARGET_LOW - occupancy
        steps = max(1, int(gap / 0.05))
        new_rate = base_rate - steps * RATE_STEP
    elif occupancy > TARGET_HIGH:
        gap = occupancy - TARGET_HIGH
        steps = max(1, int(gap / 0.05))
        new_rate = base_rate + steps * RATE_STEP
    else:
        return round(base_rate, 2)

    return round(float(np.clip(new_rate, RATE_MIN, RATE_MAX)), 2)


def revenue_impact(current_rate: float, recommended_rate: float,
                   spaces: float, hours: float = 1.0) -> dict:
    """Estimate revenue delta under recommended rate vs current."""
    # Assume 5% demand elasticity per $0.25 change (conservative)
    rate_delta = recommended_rate - current_rate
    elasticity = -0.05 / 0.25  # demand % change per $0.25
    demand_change = elasticity * rate_delta  # as fraction

    current_revenue = current_rate * spaces * hours
    new_revenue = recommended_rate * spaces * (1 + demand_change) * hours
    return {
        "current_revenue": round(current_revenue, 2),
        "projected_revenue": round(new_revenue, 2),
        "revenue_delta": round(new_revenue - current_revenue, 2),
        "revenue_delta_pct": round((new_revenue - current_revenue) / max(current_revenue, 0.01) * 100, 1),
    }


def run():
    if not FEATURES_FILE.exists():
        print("features.csv not found — run aggregate_features.py first.")
        return

    features = pd.read_csv(FEATURES_FILE, parse_dates=["hour"])

    # Work with the most recent 30 days
    cutoff = features["hour"].max() - pd.Timedelta(days=30)
    recent = features[features["hour"] >= cutoff].copy()

    if recent.empty:
        print("No recent features to generate recommendations from.")
        return

    # Load model for predictions
    try:
        model = load_model()
        feature_cols = load_feature_cols()
        available = [c for c in feature_cols if c in recent.columns]
        for col in available:
            if recent[col].dtype == bool:
                recent[col] = recent[col].astype(int)
        recent["predicted_occupancy"] = model.predict(recent[available].fillna(0))
    except FileNotFoundError as e:
        print(f"  Warning: {e}")
        print("  Using actual occupancy rates for recommendations (no model predictions).")
        recent["predicted_occupancy"] = recent["avg_occupancy_rate"]

    # Generate recommendations per region × hour bucket
    recs = []
    for (region, hour_of_day), grp in recent.groupby(["region", "hour_of_day"]):
        avg_predicted = grp["predicted_occupancy"].mean()
        avg_spaces = grp["total_spaces"].mean()
        base_rate = BASE_RATES.get(region, 2.00)
        rec_rate = recommend_rate(avg_predicted, base_rate)
        impact = revenue_impact(base_rate, rec_rate, avg_spaces)

        recs.append({
            "region": region,
            "hour_of_day": hour_of_day,
            "avg_predicted_occupancy": round(avg_predicted, 3),
            "avg_spaces": round(avg_spaces, 0),
            "current_rate": base_rate,
            "recommended_rate": rec_rate,
            "rate_change": round(rec_rate - base_rate, 2),
            "action": "increase" if rec_rate > base_rate else ("decrease" if rec_rate < base_rate else "hold"),
            **impact,
        })

    df = pd.DataFrame(recs).sort_values(["region", "hour_of_day"])
    df.to_csv(OUTPUT_FILE, index=False)

    total_delta = df["revenue_delta"].sum()
    print(f"Generated {len(df)} pricing recommendations")
    print(f"Estimated hourly revenue delta (all regions): ${total_delta:+.2f}")
    print(f"Saved → {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    run()
