import requests
import pandas as pd
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STATE_FILE = ROOT / "last_run.json"
OUTPUT_FILE = DATA_DIR / "seattle_road_closures.csv"

# Seattle Street Closures — Socrata
SOCRATA_URL = "https://data.seattle.gov/resource/ium9-iqtc.json"
TIMEOUT = 30
PAGE_SIZE = 10000


def get_last_fetch_date() -> str:
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        return state.get("last_closures_date", "")
    return ""


def update_state(latest_date: str):
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    state["last_closures_date"] = latest_date
    STATE_FILE.write_text(json.dumps(state, indent=2))


def fetch_closures(since: str) -> pd.DataFrame:
    params = {
        "$limit": PAGE_SIZE,
        "$order": "start_date ASC",
    }
    if since:
        since_iso = str(since).replace(" ", "T")
        params["$where"] = f"start_date > '{since_iso}'"

    try:
        resp = requests.get(SOCRATA_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        records = resp.json()
    except requests.RequestException as e:
        print(f"  Road closures fetch failed: {e}")
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    for col in ["start_date", "end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    keep = ["permit_number", "permit_type", "project_name", "start_date",
            "end_date", "street_on", "street_from", "street_to"]
    df = df[[c for c in keep if c in df.columns]]
    return df.dropna(subset=["start_date"])


def run():
    os.makedirs(DATA_DIR, exist_ok=True)

    since = get_last_fetch_date()
    print(f"Fetching road closures{' since ' + since if since else ''}...")

    df = fetch_closures(since)
    if df.empty:
        print("No new closures found.")
        return

    df = clean(df)
    print(f"  Fetched {len(df):,} closure records")

    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE, parse_dates=["start_date", "end_date"])
        df = pd.concat([existing, df]).drop_duplicates(subset=["permit_number"])

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

    latest = df["start_date"].max()
    if pd.notna(latest):
        update_state(str(latest))


if __name__ == "__main__":
    run()
