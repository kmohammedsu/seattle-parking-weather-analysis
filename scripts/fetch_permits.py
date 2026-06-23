import requests
import pandas as pd
import json
import os
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STATE_FILE = ROOT / "last_run.json"
OUTPUT_FILE = DATA_DIR / "seattle_event_permits.csv"

# Seattle Special Event Permits — Socrata
SOCRATA_URL = "https://data.seattle.gov/resource/dm95-f8w5.json"
TIMEOUT = 30
PAGE_SIZE = 10000


def get_last_fetch_date() -> str:
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        return state.get("last_permits_date", "")
    return ""


def update_state(latest_date: str):
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    state["last_permits_date"] = latest_date
    STATE_FILE.write_text(json.dumps(state, indent=2))


def fetch_permits(since: str) -> pd.DataFrame:
    params = {
        "$limit": PAGE_SIZE,
        "$order": "event_start_date ASC",
    }
    if since:
        since_iso = str(since).replace(" ", "T")
        params["$where"] = f"event_start_date > '{since_iso}'"

    try:
        resp = requests.get(SOCRATA_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        records = resp.json()
    except requests.RequestException as e:
        print(f"  Permits fetch failed: {e}")
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    for col in ["startdate", "enddate"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ["event_start_date", "event_end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    keep = ["year_month_app", "name_of_event", "permit_type", "permit_status",
            "event_start_date", "event_end_date", "event_location_neighborhood",
            "organization", "attendance"]
    df = df[[c for c in keep if c in df.columns]]

    if "attendance" in df.columns:
        df = df.copy()
        df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")

    return df.dropna(subset=["event_start_date"])


def run():
    os.makedirs(DATA_DIR, exist_ok=True)

    since = get_last_fetch_date()
    print(f"Fetching event permits{' since ' + since if since else ''}...")

    df = fetch_permits(since)
    if df.empty:
        print("No new permits found.")
        return

    df = clean(df)
    print(f"  Fetched {len(df):,} permits")

    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE, parse_dates=["event_start_date", "event_end_date"])
        df = pd.concat([existing, df]).drop_duplicates(subset=["year_month_app", "name_of_event"])

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

    latest = df["event_start_date"].max()
    if pd.notna(latest):
        update_state(str(latest))


if __name__ == "__main__":
    run()
