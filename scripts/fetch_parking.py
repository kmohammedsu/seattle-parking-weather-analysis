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
PAGE_SIZE = 50000


def get_last_fetch_date() -> str:
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        return state.get("last_parking_date", "")
    return ""


def update_state(latest_date: str):
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    state["last_parking_date"] = latest_date
    STATE_FILE.write_text(json.dumps(state, indent=2))


def fetch_page(offset: int, since: str) -> list:
    params = {
        "$limit": PAGE_SIZE,
        "$offset": offset,
        "$order": "occupancydatetime ASC",
    }
    if since:
        params["$where"] = f"occupancydatetime > '{since}'"

    try:
        resp = requests.get(SOCRATA_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  Page fetch failed (offset={offset}): {e}")
        return []


def fetch_all(since: str) -> pd.DataFrame:
    records = []
    offset = 0
    while True:
        print(f"  Fetching page offset={offset}...")
        page = fetch_page(offset, since)
        if not page:
            break
        records.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df["occupancydatetime"] = pd.to_datetime(df["occupancydatetime"], errors="coerce")
    df = df.dropna(subset=["occupancydatetime"])

    numeric_cols = ["paidoccupancy", "parkingspacecount"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["occupancy_rate"] = (
        df["paidoccupancy"] / df["parkingspacecount"].replace(0, pd.NA)
    ).clip(0, 1)

    keep = ["occupancydatetime", "blockfacename", "sideofstreet",
            "paidoccupancy", "parkingspacecount", "occupancy_rate"]
    df = df[[c for c in keep if c in df.columns]]
    return df


def run():
    os.makedirs(DATA_DIR, exist_ok=True)

    since = get_last_fetch_date()
    if since:
        print(f"Fetching parking data since {since}...")
    else:
        print("Fetching parking data (full last-30-days window)...")

    df = fetch_all(since)
    if df.empty:
        print("No new parking data found.")
        return

    df = clean(df)
    print(f"  Fetched {len(df):,} records")

    # Append to existing file
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE, parse_dates=["occupancydatetime"])
        df = pd.concat([existing, df]).drop_duplicates(
            subset=["occupancydatetime", "blockfacename", "sideofstreet"]
        )

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE} ({len(df):,} total records)")

    latest = df["occupancydatetime"].max()
    if pd.notna(latest):
        update_state(str(latest))


if __name__ == "__main__":
    run()
