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

TIMEOUT = 300  # Socrata GROUP BY on monthly data takes ~2.5 min
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

# Fetch one month at a time — a full year has ~1.86M grouped rows which exceeds
# reasonable limits. Each month has ~155K rows (well within 500K limit).
SOQL_MONTH_QUERY = """
SELECT
  date_trunc_ymd(occupancydatetime) AS occupancy_date,
  date_extract_hh(occupancydatetime) AS occupancy_hour,
  blockfacename,
  avg(paidoccupancy) AS avg_occupied,
  max(paidoccupancy) AS peak_occupied,
  avg(parkingspacecount) AS avg_spaces,
  count(*) AS num_readings
WHERE occupancydatetime >= '{since}' AND occupancydatetime < '{until}'
GROUP BY occupancy_date, occupancy_hour, blockfacename
ORDER BY occupancy_date ASC, occupancy_hour ASC
LIMIT 500000
"""


def load_progress() -> set:
    """Returns set of completed 'YYYY-MM' month strings."""
    if PROGRESS_FILE.exists():
        data = json.loads(PROGRESS_FILE.read_text())
        # Support old format (list of years as ints)
        completed = data.get("completed_months", data.get("completed_years", []))
        return set(str(x) for x in completed)
    return set()


def save_progress(completed: set):
    PROGRESS_FILE.write_text(json.dumps({"completed_months": sorted(completed)}, indent=2))


def fetch_month(year: int, month: int) -> pd.DataFrame:
    from calendar import monthrange
    dataset_id = YEAR_DATASETS[year]
    url = BASE_URL.format(dataset_id=dataset_id)
    last_day = monthrange(year, month)[1]
    since = f"{year}-{month:02d}-01T00:00:00"
    # Next month start for exclusive upper bound
    if month == 12:
        until = f"{year + 1}-01-01T00:00:00"
    else:
        until = f"{year}-{month + 1:02d}-01T00:00:00"

    query = SOQL_MONTH_QUERY.format(since=since, until=until)
    try:
        resp = requests.get(url, params={"$query": query}, timeout=TIMEOUT)
        resp.raise_for_status()
        records = resp.json()
    except requests.RequestException as e:
        print(f"    FAILED {year}-{month:02d}: {e}")
        return pd.DataFrame()

    return pd.DataFrame(records) if records else pd.DataFrame()


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

    # Build list of all year-month pairs to fetch
    today = date.today()
    all_months = [
        (y, m)
        for y in YEAR_DATASETS
        for m in range(1, 13)
        if date(y, m, 1) <= today
    ]
    remaining = [(y, m) for y, m in all_months if f"{y}-{m:02d}" not in completed]

    if not remaining:
        print("Backfill already complete.")
        return

    years_remaining = sorted(set(y for y, _ in remaining))
    print(f"Backfill: {len(remaining)} months across years {years_remaining}")
    print("Saves after every month — safe to interrupt.\n")

    for year, month in remaining:
        key = f"{year}-{month:02d}"
        print(f"  Fetching {key}...")
        df = fetch_month(year, month)

        if not df.empty:
            df = clean(df)
            if OUTPUT_FILE.exists():
                existing = pd.read_csv(OUTPUT_FILE, parse_dates=["occupancy_date"])
                df = pd.concat([existing, df]).drop_duplicates(
                    subset=["occupancy_date", "occupancy_hour", "blockfacename"]
                ).sort_values(["occupancy_date", "occupancy_hour", "blockfacename"])
            df.to_csv(OUTPUT_FILE, index=False)
            print(f"    {key}: {len(df):,} total rows saved")
            completed.add(key)
            save_progress(completed)
        else:
            # Only skip permanently if it's a month that clearly has no data
            # (not a timeout — timeouts print FAILED and return empty too)
            print(f"    {key}: no data (skipping)")
        time.sleep(0.5)  # be polite to Socrata

    total = pd.read_csv(OUTPUT_FILE).shape[0] if OUTPUT_FILE.exists() else 0
    print(f"\nBackfill complete. {total:,} total rows → {OUTPUT_FILE.name}")
    print("Next: run aggregate_features.py to rebuild the feature matrix.")


if __name__ == "__main__":
    run()
