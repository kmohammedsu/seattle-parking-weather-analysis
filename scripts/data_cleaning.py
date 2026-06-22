import pandas as pd
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "Annual_Parking_Study_Data_20250220.csv"
OUTPUT_FILE = ROOT / "data" / "Cleaned_Annual_Parking_Study_Data_20250220.csv"

NUMERIC_COLS = ["Parking_Spaces", "Total_Vehicle_Count", "Dp_Count", "Rpz_Count"]
CATEGORICAL_COLS = ["Study_Area", "Sub_Area"]
SPARSE_COLS = ["TG_Car2Go", "BMW_DN", "Lime", "Idling", "Field Notes", "Peak Hour_SDOT"]


def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, low_memory=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Parse datetime
    df["Date Time"] = pd.to_datetime(df["Date Time"], errors="coerce", format="%m/%d/%Y %H:%M")
    df = df.dropna(subset=["Date Time"])

    # Drop sparse columns not used in analysis
    df = df.drop(columns=[c for c in SPARSE_COLS if c in df.columns])

    # Numeric coercion
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing numerics with median
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(df[NUMERIC_COLS].median())

    # Remove negative vehicle counts
    df = df[df["Total_Vehicle_Count"] >= 0]

    # Normalize categorical columns — strip whitespace, title case, fill unknown
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title().replace("Nan", "Unknown")

    # Normalize Study_Area casing inconsistencies (e.g. "Ballard locks summer" variants)
    if "Study_Area" in df.columns:
        df["Study_Area"] = df["Study_Area"].str.strip().str.title()

    df = df.set_index("Date Time")
    return df


def run():
    print(f"Loading {RAW_FILE.name}...")
    df = load_raw(RAW_FILE)
    print(f"  Raw shape: {df.shape}")

    df = clean(df)
    print(f"  Clean shape: {df.shape}")

    df.to_csv(OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    run()
