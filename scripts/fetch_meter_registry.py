"""
Build the meter registry — the static attribute table for every paid parking
blockface in Seattle.

This replaces the old geocoding subsystem (shapely polygons + blockface_coords
+ a 5-region keyword fallback that silently misfiled ~17% of meters into
"Downtown Seattle"). Seattle's own feed carries the authoritative values at
100% coverage:

    sourceelementkey   stable unique meter id
    paidparkingarea    Seattle's 23 official paid parking areas
    location           real lat/lon per blockface
    parkingspacecount  authoritative capacity

The registry is small (~1,100 rows) and static, so it is committed to the repo
and joined onto the hourly occupancy data during aggregation.
"""
import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "meter_registry.csv"

SOCRATA_URL = "https://data.seattle.gov/resource/rke9-rsvs.json"
TIMEOUT = 120
PAGE_SIZE = 2000

# Look back far enough to catch every active meter, but not so far that
# decommissioned blockfaces pollute the registry.
LOOKBACK_DAYS = 45

SOQL = """
SELECT
  sourceelementkey, blockfacename, paidparkingarea, paidparkingsubarea, location,
  max(parkingspacecount) AS spaces,
  max(parkingtimelimitcategory) AS time_limit,
  count(*) AS readings
WHERE occupancydatetime >= '{since}'
GROUP BY sourceelementkey, blockfacename, paidparkingarea, paidparkingsubarea, location
ORDER BY sourceelementkey
LIMIT {limit} OFFSET {offset}
"""


def fetch_all(since: str) -> pd.DataFrame:
    """Page through the grouped query until the API stops returning rows."""
    frames, offset = [], 0
    while True:
        query = SOQL.format(since=since, limit=PAGE_SIZE, offset=offset)
        try:
            resp = requests.get(SOCRATA_URL, params={"$query": query}, timeout=TIMEOUT)
            resp.raise_for_status()
            records = resp.json()
        except requests.RequestException as e:
            print(f"  Fetch failed at offset {offset}: {e}")
            break

        if not records:
            break

        frames.append(pd.DataFrame(records))
        print(f"  Fetched {len(records)} meters (offset {offset})")
        offset += PAGE_SIZE

        if len(records) < PAGE_SIZE:
            break

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def extract_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Pull lat/lon out of the Socrata GeoJSON point column."""
    def coords(loc):
        if isinstance(loc, dict):
            c = loc.get("coordinates") or []
            if len(c) == 2:
                return pd.Series({"lon": c[0], "lat": c[1]})
        return pd.Series({"lon": pd.NA, "lat": pd.NA})

    return df["location"].apply(coords)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = pd.concat([df, extract_coords(df)], axis=1).drop(columns=["location"])

    df["sourceelementkey"] = pd.to_numeric(df["sourceelementkey"], errors="coerce")
    for col in ["spaces", "time_limit", "readings", "lat", "lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["sourceelementkey", "paidparkingarea", "blockfacename"])
    df["sourceelementkey"] = df["sourceelementkey"].astype(int)
    df["paidparkingsubarea"] = df["paidparkingsubarea"].fillna("")

    # Roll meter keys up to the blockface.
    #
    # Seattle issues a sourceelementkey per street segment/side (1,512 of them),
    # but prices by blockface (965) — and, critically, our historical occupancy
    # data only carries blockfacename, since the Socrata feed is "Last 30 Days"
    # and meter-key history beyond that does not exist. Blockface is therefore
    # both the correct pricing unit and the only one with full history.
    #
    # Blockface -> paidparkingarea is a verified 1:1 mapping, so the area label
    # survives the rollup unambiguously.
    reg = (
        df.groupby("blockfacename")
        .agg(
            paidparkingarea=("paidparkingarea", "first"),
            paidparkingsubarea=("paidparkingsubarea", "first"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            spaces=("spaces", "sum"),        # total capacity across both sides
            time_limit=("time_limit", "max"),
            n_meters=("sourceelementkey", "nunique"),
        )
        .reset_index()
        .sort_values("blockfacename")
        .reset_index(drop=True)
    )
    return reg


def run():
    os.makedirs(DATA_DIR, exist_ok=True)

    since = (pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=LOOKBACK_DAYS)).date().isoformat()
    print(f"Building meter registry (meters active since {since})...")

    df = clean(fetch_all(since))

    if df.empty:
        print("  No meters returned — keeping existing registry.")
        return

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nRegistry: {len(df):,} blockfaces across {df['paidparkingarea'].nunique()} official areas")
    print(f"  Total paid spaces: {int(df['spaces'].sum()):,}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
