import requests
import pandas as pd
import json
import os
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STATE_FILE = ROOT / "last_run.json"
OUTPUT_FILE = DATA_DIR / "live_parking_occupancy.csv"

# Socrata API — Paid Parking Occupancy (Last 30 Days)
SOCRATA_URL = "https://data.seattle.gov/resource/rke9-rsvs.json"
TIMEOUT = 60

# Aggregate on the server — fetch summaries not raw rows
# This turns 15M rows into ~thousands of hourly summaries
SOQL_QUERY = """
SELECT
  date_trunc_ymd(occupancydatetime) AS occupancy_date,
  date_extract_hh(occupancydatetime) AS occupancy_hour,
  blockfacename,
  avg(paidoccupancy) AS avg_occupied,
  max(paidoccupancy) AS peak_occupied,
  avg(parkingspacecount) AS avg_spaces,
  count(*) AS num_readings
WHERE occupancydatetime >= '{since}' AND occupancydatetime <= '{until}'
GROUP BY occupancy_date, occupancy_hour, blockfacename
ORDER BY occupancy_date ASC, occupancy_hour ASC
LIMIT 500000
"""


def get_date_range() -> tuple[str, str]:
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    last = state.get("last_parking_date", "")

    if last:
        since = (date.fromisoformat(last[:10]) + timedelta(days=1)).isoformat()
    else:
        since = (date.today() - timedelta(days=30)).isoformat()

    until = date.today().isoformat()
    return since, until


def update_state(latest_date: str):
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    state["last_parking_date"] = latest_date
    STATE_FILE.write_text(json.dumps(state, indent=2))


def fetch_aggregated(since: str, until: str) -> pd.DataFrame:
    query = SOQL_QUERY.format(since=since, until=until)
    params = {"$query": query}

    print(f"  Querying Socrata (server-side aggregation)...")
    try:
        resp = requests.get(SOCRATA_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        records = resp.json()
    except requests.RequestException as e:
        print(f"  Fetch failed: {e}")
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


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
    os.makedirs(DATA_DIR, exist_ok=True)

    since, until = get_date_range()
    if since > until:
        print("Parking data already up to date.")
        return

    print(f"Fetching parking data {since} → {until} (server-side aggregated)...")
    df = fetch_aggregated(since, until)

    if df.empty:
        print("No new parking data found.")
        return

    df = clean(df)
    print(f"  Fetched {len(df):,} hourly block summaries")

    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE, parse_dates=["occupancy_date"])
        df = pd.concat([existing, df]).drop_duplicates(
            subset=["occupancy_date", "occupancy_hour", "blockfacename"]
        )

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE} ({len(df):,} total rows)")

    latest = df["occupancy_date"].max()
    if pd.notna(latest):
        update_state(str(latest)[:10])


if __name__ == "__main__":
    run()
