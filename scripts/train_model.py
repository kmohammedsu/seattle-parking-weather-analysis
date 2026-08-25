"""
Train LightGBM to predict occupancy per BLOCKFACE per hour.

The model learns a demand curve for each individual blockface rather than for a
broad neighbourhood, which is what makes per-meter pricing possible. Blockface
identity and the official parking area are passed as native LightGBM
categoricals so the model can learn per-location baselines directly.

Evaluation uses a TEMPORAL split (train on the past, test on the most recent
slice). A random shuffle would leak future information into training and inflate
R² — badly so with blockface as a categorical, since the model could recover a
blockface's level from temporally adjacent rows.

Saves model to models/ and performance metrics to models/performance/.
"""
import pickle
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
PERF_DIR = MODELS_DIR / "performance"

FEATURES_FILE = DATA_DIR / "features.parquet"
MODEL_FILE = MODELS_DIR / "parking_demand_lgbm.pkl"
PERF_FILE = PERF_DIR / "final_model_performance.csv"
FEATURE_LIST_FILE = MODELS_DIR / "feature_columns.json"
CATEGORIES_FILE = MODELS_DIR / "feature_categories.json"

# Location identity — the features that make this per-meter rather than per-region
CATEGORICAL_COLS = ["blockfacename", "paidparkingarea"]

# Leakage-free features — excludes any direct function of the target
FEATURE_COLS = [
    # location identity + physical attributes
    "blockfacename",
    "paidparkingarea",
    "total_spaces",
    "n_meters",
    "time_limit",
    "lat",
    "lon",
    # weather
    "temperature",
    "precipitation",
    "wind_speed",
    "elevation",
    # demand drivers
    "is_event_day",
    "has_city_event",
    "max_attendance",
    "has_road_closure",
    "is_holiday",
    # time
    "hour_of_day",
    "day_of_week",
    "month",
    "year",
    "is_weekend",
    "is_peak_am",
    "is_peak_pm",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

TARGET = "avg_occupancy_rate"
TEST_FRACTION = 0.2   # most recent 20% of the window, chronologically

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 255,          # raised: ~960 blockfaces need more capacity than 5 regions
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "max_cat_threshold": 64,
    "cat_smooth": 20.0,
    "verbose": -1,
}

WEATHER_COLS = {"temperature", "precipitation", "wind_speed", "elevation"}


def load_features(rolling_months=12) -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_FILE)
    if rolling_months is not None:
        cutoff = df["hour"].max() - pd.DateOffset(months=rolling_months)
        df = df[df["hour"] >= cutoff].copy()
        print(f"  Rolling window: {cutoff.date()} → {df['hour'].max().date()} ({len(df):,} rows)")
    else:
        print(f"  Full history: {df['hour'].min().date()} → {df['hour'].max().date()} ({len(df):,} rows)")
    return df


def prepare(df: pd.DataFrame) -> tuple:
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available)
    if missing:
        print(f"  Warning: missing features (will be skipped): {missing}")

    for col in available:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # Historical rows may predate weather coverage; median-fill keeps them usable
    for col in available:
        if col in WEATHER_COLS and df[col].isna().any():
            median = df[col].median()
            fill = median if pd.notna(median) else 0.0
            df[col] = df[col].fillna(fill)
            print(f"  Filled {col} NaNs with median ({fill:.2f})")

    cats = [c for c in CATEGORICAL_COLS if c in available]
    for col in cats:
        df[col] = df[col].astype("category")

    df = df.dropna(subset=[TARGET])
    # Keep chronological order so the split below is a true time split
    df = df.sort_values("hour")

    X = df[available]
    y = df[TARGET]
    return X, y, available, cats


def temporal_split(X, y, test_fraction=TEST_FRACTION):
    """Train on the earlier portion, test on the most recent portion."""
    cut = int(len(X) * (1 - test_fraction))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def train(X_train, y_train, X_val, y_val, cats, existing_model=None):
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cats)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data,
                           categorical_feature=cats)

    return lgb.train(
        LGBM_PARAMS,
        train_data,
        num_boost_round=600,
        valid_sets=[train_data, val_data],
        init_model=existing_model,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "n_test": len(y_test),
    }


def run(rolling_months=12):
    MODELS_DIR.mkdir(exist_ok=True)
    PERF_DIR.mkdir(exist_ok=True)

    if not FEATURES_FILE.exists():
        print(f"{FEATURES_FILE.name} not found — run aggregate_features.py first.")
        return

    print("Loading features...")
    df = load_features(rolling_months=rolling_months)

    if len(df) < 100:
        print(f"  Only {len(df)} rows — need more data. Run backfill_parking.py first.")
        return

    X, y, used_cols, cats = prepare(df)
    n_bf = X["blockfacename"].nunique() if "blockfacename" in X else 0
    print(f"  Training on {len(X):,} samples | {len(used_cols)} features | "
          f"{n_bf:,} blockfaces")

    X_train, X_test, y_train, y_test = temporal_split(X, y)
    print(f"  Temporal split: {len(X_train):,} train / {len(X_test):,} test "
          f"(most recent {int(TEST_FRACTION*100)}%)")

    # Warm-start only if the feature set AND category sets still match
    existing_model = None
    if MODEL_FILE.exists() and FEATURE_LIST_FILE.exists():
        saved_cols = json.loads(FEATURE_LIST_FILE.read_text())
        if saved_cols == used_cols:
            with open(MODEL_FILE, "rb") as f:
                existing_model = pickle.load(f)
            print("  Warm-starting from existing model")
        else:
            print("  Feature set changed — training from scratch")

    print("Training LightGBM...")
    model = train(X_train, y_train, X_test, y_test, cats, existing_model)
    print(f"  Best iteration: {model.best_iteration}")

    metrics = evaluate(model, X_test, y_test)
    metrics.update({
        "n_train": len(X_train),
        "n_blockfaces": int(n_bf),
        "best_iteration": model.best_iteration,
        "trained_at": datetime.utcnow().isoformat(),
        "n_features": len(used_cols),
        "split": "temporal",
    })

    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE : {metrics['mae']:.4f}")
    print(f"  R²  : {metrics['r2']:.4f}   (temporal split — not comparable to "
          f"the old shuffled-split figure)")

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    FEATURE_LIST_FILE.write_text(json.dumps(used_cols, indent=2))

    # Persist category levels so prediction-time encoding matches training
    CATEGORIES_FILE.write_text(json.dumps(
        {c: sorted(map(str, X[c].cat.categories)) for c in cats}, indent=2
    ))

    pd.DataFrame([metrics]).to_csv(PERF_FILE, index=False)
    print(f"  Model saved → {MODEL_FILE}")
    print(f"  Metrics saved → {PERF_FILE}")
    return metrics


if __name__ == "__main__":
    run()
