"""
Train LightGBM on the feature matrix with a rolling 12-month window.
Saves model to models/ and performance metrics to models/performance/.
Called by run_pipeline.py after aggregate_features.py completes.
"""
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
PERF_DIR = MODELS_DIR / "performance"

FEATURES_FILE = DATA_DIR / "features.csv"
MODEL_FILE = MODELS_DIR / "parking_demand_lgbm.pkl"
PERF_FILE = PERF_DIR / "final_model_performance.csv"
FEATURE_LIST_FILE = MODELS_DIR / "feature_columns.json"

# Leakage-free features — excludes turnover_proxy (same info as target)
FEATURE_COLS = [
    "total_spaces",
    "num_blockfaces",
    "temperature",
    "precipitation",
    "wind_speed",
    "elevation",
    "is_event_day",
    "has_city_event",
    "max_attendance",
    "has_road_closure",
    "is_holiday",
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

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
}


def load_features(rolling_months=12) -> pd.DataFrame:
    df = pd.read_csv(FEATURES_FILE, parse_dates=["hour"])
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

    df = df.dropna(subset=[TARGET] + available)
    X = df[available]
    y = df[TARGET]
    return X, y, available


def train(X_train, y_train, X_val, y_val, existing_model=None):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    return lgb.train(
        LGBM_PARAMS,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        init_model=existing_model,  # warm start if retraining
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2, "n_test": len(y_test)}


def run(rolling_months=12):
    MODELS_DIR.mkdir(exist_ok=True)
    PERF_DIR.mkdir(exist_ok=True)

    if not FEATURES_FILE.exists():
        print("features.csv not found — run aggregate_features.py first.")
        return

    print("Loading features...")
    df = load_features(rolling_months=rolling_months)

    if len(df) < 100:
        print(f"  Only {len(df)} rows — need more data. Run backfill_parking.py first.")
        return

    X, y, used_cols = prepare(df)
    print(f"  Training on {len(X):,} samples with {len(used_cols)} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Warm-start from existing model if available and feature set matches
    existing_model = None
    if MODEL_FILE.exists() and FEATURE_LIST_FILE.exists():
        saved_cols = json.loads(FEATURE_LIST_FILE.read_text())
        if saved_cols == used_cols:
            with open(MODEL_FILE, "rb") as f:
                existing_model = pickle.load(f)
            print("  Warm-starting from existing model")

    print("Training LightGBM...")
    model = train(X_train, y_train, X_test, y_test, existing_model)
    print(f"  Best iteration: {model.best_iteration}")

    metrics = evaluate(model, X_test, y_test)
    metrics.update({
        "n_train": len(X_train),
        "best_iteration": model.best_iteration,
        "trained_at": datetime.utcnow().isoformat(),
        "n_features": len(used_cols),
    })

    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE : {metrics['mae']:.4f}")
    print(f"  R²  : {metrics['r2']:.4f}")

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    FEATURE_LIST_FILE.write_text(json.dumps(used_cols, indent=2))

    pd.DataFrame([metrics]).to_csv(PERF_FILE, index=False)
    print(f"  Model saved → {MODEL_FILE}")
    print(f"  Metrics saved → {PERF_FILE}")

    return metrics


if __name__ == "__main__":
    run()
