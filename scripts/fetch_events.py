import requests
import pandas as pd
import json
import os
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "seattle_events.csv"

# Ticketmaster Discovery API — free tier, 5000 calls/day
# Set TICKETMASTER_API_KEY in environment or .env
TM_URL = "https://app.ticketmaster.com/discovery/v2/events.json"
TIMEOUT = 30

SEATTLE_VENUES = {
    "Lumen Field":          "KovZpZAEdntA",   # Seahawks / Sounders / concerts
    "T-Mobile Park":        "KovZpZAEevAA",   # Mariners
    "Climate Pledge Arena": "KovZ917Ahkk",    # Kraken / Storm / concerts
}


def get_api_key() -> str | None:
    key = os.environ.get("TICKETMASTER_API_KEY", "")
    if not key:
        print("  TICKETMASTER_API_KEY not set — skipping events fetch")
    return key or None


def fetch_venue_events(venue_id: str, venue_name: str, api_key: str) -> list:
    start = date.today().isoformat() + "T00:00:00Z"
    end = (date.today() + timedelta(days=60)).isoformat() + "T23:59:59Z"

    params = {
        "apikey": api_key,
        "venueId": venue_id,
        "startDateTime": start,
        "endDateTime": end,
        "size": 200,
    }

    try:
        resp = requests.get(TM_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        events = data.get("_embedded", {}).get("events", [])
        return events
    except requests.RequestException as e:
        print(f"  Failed for {venue_name}: {e}")
        return []


def parse_events(events: list, venue_name: str) -> list:
    parsed = []
    for e in events:
        start_info = e.get("dates", {}).get("start", {})
        parsed.append({
            "event_id": e.get("id"),
            "event_name": e.get("name"),
            "venue": venue_name,
            "event_date": start_info.get("localDate"),
            "event_time": start_info.get("localTime"),
            "event_type": e.get("classifications", [{}])[0].get("segment", {}).get("name"),
            "genre": e.get("classifications", [{}])[0].get("genre", {}).get("name"),
        })
    return parsed


def run():
    os.makedirs(DATA_DIR, exist_ok=True)

    api_key = get_api_key()
    if not api_key:
        return

    all_events = []
    for venue_name, venue_id in SEATTLE_VENUES.items():
        print(f"  Fetching events for {venue_name}...")
        events = fetch_venue_events(venue_id, venue_name, api_key)
        parsed = parse_events(events, venue_name)
        all_events.extend(parsed)
        print(f"    {len(parsed)} events found")

    if not all_events:
        print("No events fetched.")
        return

    df = pd.DataFrame(all_events)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"])
    df = df.sort_values("event_date")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} events to {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    run()
