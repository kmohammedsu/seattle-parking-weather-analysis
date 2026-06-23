"""
Backfill historical paid parking occupancy data (2012–2024).

The live fetch script only pulls the last 30 days. This script fills in
the full history so the model has years of patterns to learn from.
Run once: python scripts/backfill_parking.py
Each year is fetched as one Socrata call (~150K rows after aggregation).
Progress is saved so the script can be interrupted and resumed.
"""
import requests
import pandas as pd
import json
import time
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "live_parking_occupancy.csv"
PROGRESS_FILE = DATA_DIR / "backfill_progress.json"

TIMEOUT = 120
BASE_URL = "https://data.seattle.gov/resource/{dataset_id}.json"

# Each year is a separate Socrata dataset on Seattle Open Data.
# 2012-2017 are archived as file-only (not queryable via API) — start from 2018.
YEAR_DATASETS = {
    2018: "6yaw-2m8q",
    2019: "qktt-2bsy",
    2020: "wtpb-jp8d",
    2021: "jb6y-98nr",
    2022: "bwk6-iycu",
    2023: "3uar-q5py",
    2024: "snbb-v8b9",
    2025: "7c2e-uany",
}

# Each year-dataset only contains that year's data — no WHERE clause needed
SOQL_QUERY = """
SELECT
  date_trunc_ymd(occupancydatetime) AS occupancy_date,
  date_extract_hh(occupancydatetime) AS occupancy_hour,
  blockfacename,
  avg(paidoccupancy) AS avg_occupied,
  max(paidoccupancy) AS peak_occupied,
  avg(parkingspacecount) AS avg_spaces,
  count(*) AS num_readings
GROUP BY occupancy_date, occupancy_hour, blockfacename
ORDER BY occupancy_date ASC, occupancy_hour ASC
LIMIT 500000
"""


def load_progress() -> set:
    if PROGRESS_FILE.exists():
        return set(json.loads(PROGRESS_FILE.read_text()).get("completed_years", []))
    return set()


def save_progress(completed: set):
    PROGRESS_FILE.write_text(json.dumps({"completed_years": sorted(completed)}, indent=2))


def fetch_year(year: int) -> pd.DataFrame:
    dataset_id = YEAR_DATASETS[year]
    url = BASE_URL.format(dataset_id=dataset_id)

    print(f"  Querying {year} (dataset {dataset_id})...")
    try:
        resp = requests.get(url, params={"$query": SOQL_QUERY}, timeout=TIMEOUT)
        resp.raise_for_status()
        records = resp.json()
    except requests.RequestException as e:
        print(f"  FAILED {year}: {e}")
        return pd.DataFrame()

    if not records:
        print(f"  {year}: 0 records returned")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print(f"  {year}: {len(df):,} hourly block summaries")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["occupancy_date"] = pd.to_datetime(df["occupancy_date"], errors="coerce")
    df["occupancy_hour"] = pd.to_numeric(df["occupancy_hour"], errors="coerce")
    for col in ["avg_occupied", "peak_occupied", "avg_spaces", "num_readings"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["avg_spaces"] = df["avg_spaces"].replace(0, pd.NA)
    df["occupancy_rate"] = (df["avg_occupied"] / df["avg_spaces"]).clip(0, 1)
    df["peak_occupancy_rate"] = (df["peak_occupied"] / df["avg_spaces"]).clip(0, 1)
    return df.dropna(subset=["occupancy_date"])


def run():
    DATA_DIR.mkdir(exist_ok=True)
    completed = load_progress()
    remaining = [y for y in YEAR_DATASETS if y not in completed]

    if not remaining:
        print("Backfill already complete.")
        return

    print(f"Backfill: {len(remaining)} years to fetch — {remaining}")
    print("This will take several minutes. Progress is saved per year.\n")

    all_new = []

    for year in remaining:
        df = fetch_year(year)
        if not df.empty:
            df = clean(df)
            all_new.append(df)
            completed.add(year)
            save_progress(completed)
        else:
            print(f"  Skipping {year} (no data)")
            completed.add(year)
            save_progress(completed)
        time.sleep(1)  # be polite to Socrata

    if not all_new:
        print("No data fetched.")
        return

    new_data = pd.concat(all_new, ignore_index=True)
    print(f"\nFetched {len(new_data):,} total new rows across {len(remaining)} years")

    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE, parse_dates=["occupancy_date"])
        print(f"Existing file: {len(existing):,} rows")
        combined = pd.concat([existing, new_data]).drop_duplicates(
            subset=["occupancy_date", "occupancy_hour", "blockfacename"]
        ).sort_values(["occupancy_date", "occupancy_hour", "blockfacename"])
    else:
        combined = new_data.sort_values(["occupancy_date", "occupancy_hour", "blockfacename"])

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved: {len(combined):,} total rows → {OUTPUT_FILE}")
    print("\nNext: run aggregate_features.py to rebuild the feature matrix.")


if __name__ == "__main__":
    run()
