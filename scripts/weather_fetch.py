import requests
import json
import os
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STATE_FILE = ROOT / "last_run.json"
OUTPUT_FILE = DATA_DIR / "seattle_weather_by_region.json"

SEATTLE_REGIONS = {
    "Downtown Seattle":   {"latitude": 47.6062, "longitude": -122.3321},
    "South Lake Union":   {"latitude": 47.6235, "longitude": -122.3381},
    "Capitol Hill":       {"latitude": 47.6219, "longitude": -122.3194},
    "Ballard":            {"latitude": 47.6686, "longitude": -122.3867},
    "Industrial District":{"latitude": 47.5868, "longitude": -122.3331},
}

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEOUT = 30


def get_date_range() -> tuple[str, str]:
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        start = state.get("last_weather_date", "2014-04-08")
        # start from the day after last successful fetch
        start = (date.fromisoformat(start) + timedelta(days=1)).isoformat()
    else:
        start = "2014-04-08"

    end = (date.today() - timedelta(days=5)).isoformat()  # archive has ~5 day lag
    return start, end


def fetch_region(region: str, coords: dict, start: str, end: str) -> dict | None:
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "start_date": start,
        "end_date": end,
        "hourly": ["temperature_2m", "precipitation", "windspeed_10m"],
        "timezone": "America/Los_Angeles",
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"  Failed for {region}: {e}")
        return None


def run():
    os.makedirs(DATA_DIR, exist_ok=True)

    start, end = get_date_range()
    if start > end:
        print("Weather data already up to date.")
        return

    print(f"Fetching weather {start} → {end}")

    # Load existing data to append to
    existing = {}
    if OUTPUT_FILE.exists():
        existing = json.loads(OUTPUT_FILE.read_text())

    for region, coords in SEATTLE_REGIONS.items():
        print(f"  {region}...")
        data = fetch_region(region, coords, start, end)
        if data:
            existing[region] = data
            print(f"  OK")

    OUTPUT_FILE.write_text(json.dumps(existing, indent=2))
    print(f"Saved to {OUTPUT_FILE}")

    # Update state
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    state["last_weather_date"] = end
    STATE_FILE.write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    run()
