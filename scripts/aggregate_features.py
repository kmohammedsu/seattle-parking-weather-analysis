"""
Aggregate raw data sources into the modelling feature table.

The unit of analysis is the BLOCKFACE — the level Seattle actually prices at,
and the finest level with full history (the Socrata occupancy feed is
"Last 30 Days", so per-meter-key history beyond 30 days does not exist).

Region labels come from Seattle's own `paidparkingarea` field via the meter
registry, replacing the previous hand-drawn 5-polygon geocoding that silently
misfiled roughly a sixth of all blockfaces into "Downtown Seattle".

Outputs
    data/features.parquet          meter-level, full history  (gitignored, for training)
    data/area_daily.csv            area x date rollup          (committed, dashboard trends)
    data/area_hour_profile.csv     area x hour-of-day profile  (committed, dashboard heatmap)
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

PARKING_FILE   = DATA_DIR / "live_parking_occupancy.csv"
WEATHER_FILE   = DATA_DIR / "processed_weather_data.csv"
EVENTS_FILE    = DATA_DIR / "seattle_events.csv"
PERMITS_FILE   = DATA_DIR / "seattle_event_permits.csv"
CLOSURES_FILE  = DATA_DIR / "seattle_road_closures.csv"
HOLIDAYS_FILE  = DATA_DIR / "seattle_holidays.csv"
REGISTRY_FILE  = DATA_DIR / "meter_registry.csv"

OUTPUT_FILE       = DATA_DIR / "features.parquet"
AREA_DAILY_FILE   = DATA_DIR / "area_daily.csv"
AREA_PROFILE_FILE = DATA_DIR / "area_hour_profile.csv"

# Official Seattle paid parking areas for the major event venues.
VENUE_AREA = {
    "Lumen Field":          "Pioneer Square",
    "T-Mobile Park":        "Pioneer Square",
    "Climate Pledge Arena": "Uptown",
}

# Weather is fetched at five points across the city; each blockface inherits
# the nearest one. Seattle's weather varies little over these distances, but
# elevation does, and elevation is a real feature.
WEATHER_STATIONS = {
    "Downtown Seattle":       (47.6062, -122.3321),
    "South Lake Union":       (47.6235, -122.3381),
    "Capitol Hill":           (47.6219, -122.3194),
    "Ballard":                (47.6686, -122.3867),
    "International District": (47.5868, -122.3331),
}


def load_registry() -> pd.DataFrame:
    """Blockface -> official area, coordinates, capacity, time limit."""
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(
            f"{REGISTRY_FILE} missing. Run scripts/fetch_meter_registry.py first."
        )
    reg = pd.read_csv(REGISTRY_FILE)
    reg["paidparkingsubarea"] = reg["paidparkingsubarea"].fillna("")
    reg["weather_station"] = nearest_station(reg["lat"], reg["lon"])
    return reg


def nearest_station(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """Assign each point to the closest weather station (equirectangular is
    plenty accurate at city scale)."""
    names = list(WEATHER_STATIONS)
    coords = np.array([WEATHER_STATIONS[n] for n in names])
    lat_r, lon_r = np.radians(lat.to_numpy()), np.radians(lon.to_numpy())
    s_lat, s_lon = np.radians(coords[:, 0]), np.radians(coords[:, 1])

    dx = (lon_r[:, None] - s_lon[None, :]) * np.cos((lat_r[:, None] + s_lat[None, :]) / 2)
    dy = lat_r[:, None] - s_lat[None, :]
    idx = np.argmin(dx ** 2 + dy ** 2, axis=1)
    return pd.Series([names[i] for i in idx], index=lat.index)


def load_parking() -> pd.DataFrame:
    if not PARKING_FILE.exists():
        print("  No live parking data yet — skipping")
        return pd.DataFrame()

    df = pd.read_csv(
        PARKING_FILE,
        usecols=["occupancy_date", "occupancy_hour", "blockfacename",
                 "avg_occupied", "peak_occupied", "avg_spaces"],
        parse_dates=["occupancy_date"],
        dtype={"blockfacename": "category"},
    )
    df["hour"] = df["occupancy_date"] + pd.to_timedelta(df["occupancy_hour"], unit="h")
    return df


def aggregate_parking(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per blockface-hour. This is the granularity the
    whole platform is built on — do not aggregate above it here."""
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby(["hour", "blockfacename"], observed=True).agg(
        avg_occupied=("avg_occupied", "mean"),
        peak_occupied=("peak_occupied", "max"),
        total_spaces=("avg_spaces", "sum"),
    ).reset_index()

    agg["avg_occupancy_rate"] = (agg["avg_occupied"] / agg["total_spaces"]).clip(0, 1)
    agg["peak_occupancy_rate"] = (agg["peak_occupied"] / agg["total_spaces"]).clip(0, 1)
    return agg


def load_weather() -> pd.DataFrame:
    if not WEATHER_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(WEATHER_FILE, parse_dates=["timestamp"])
    df["hour"] = df["timestamp"].dt.floor("h")
    df = df.rename(columns={"region": "weather_station"})
    return df[["hour", "weather_station", "temperature", "precipitation",
               "wind_speed", "elevation"]]


def load_events() -> pd.DataFrame:
    if not EVENTS_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(EVENTS_FILE, parse_dates=["event_date"])
    df["date"] = df["event_date"].dt.date
    df["paidparkingarea"] = df["venue"].map(VENUE_AREA)
    df = df.dropna(subset=["paidparkingarea"])
    return df[["date", "paidparkingarea", "event_name", "venue", "genre"]]


def load_permits() -> pd.DataFrame:
    if not PERMITS_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(PERMITS_FILE, parse_dates=["event_start_date", "event_end_date"])
    df["date"] = df["event_start_date"].dt.date
    return df[["date", "name_of_event", "event_location_neighborhood", "attendance"]]


def load_closures() -> pd.DataFrame:
    if not CLOSURES_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(CLOSURES_FILE, parse_dates=["start_date", "end_date"])
    return df[["start_date", "end_date", "street_on", "permit_type"]]


def load_holidays() -> pd.DataFrame:
    if not HOLIDAYS_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(HOLIDAYS_FILE, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    return df[["date", "holiday_name", "holiday_type"]]


def build_event_flags(base: pd.DataFrame, events: pd.DataFrame,
                      permits: pd.DataFrame, holidays: pd.DataFrame,
                      closures: pd.DataFrame) -> pd.DataFrame:
    base["date"] = base["hour"].dt.date

    # Sports / concert events — scoped to the venue's official parking area
    if not events.empty:
        event_days = events.groupby(["date", "paidparkingarea"]).agg(
            is_event_day=("event_name", lambda x: True),
            event_count=("event_name", "count"),
            event_genres=("genre", lambda x: "|".join(x.dropna().unique())),
        ).reset_index()
        base = base.merge(event_days, on=["date", "paidparkingarea"], how="left")
        # The merge yields True/NaN, so presence is the signal. notna() avoids
        # the object-dtype fillna downcast warning.
        base["is_event_day"] = base["is_event_day"].notna()
    else:
        base["is_event_day"] = False
        base["event_count"] = 0
        base["event_genres"] = ""

    # City event permits (citywide signal)
    if not permits.empty:
        permit_days = permits.groupby("date").agg(
            has_city_event=("name_of_event", lambda x: True),
            max_attendance=("attendance", "max"),
        ).reset_index()
        base = base.merge(permit_days, on="date", how="left")
        base["has_city_event"] = base["has_city_event"].notna()
    else:
        base["has_city_event"] = False
        base["max_attendance"] = 0

    if not holidays.empty:
        base = base.merge(holidays, on="date", how="left")
    else:
        base["holiday_name"] = ""
        base["holiday_type"] = ""

    if not closures.empty:
        closure_dates = set()
        starts = closures["start_date"].dt.date
        ends = closures["end_date"].dt.date
        for d in base["date"].unique():
            if ((starts <= d) & (ends >= d)).any():
                closure_dates.add(d)
        base["has_road_closure"] = base["date"].isin(closure_dates)
    else:
        base["has_road_closure"] = False
    # Both flags are already proper booleans by this point (set in their
    # respective branches above), so just normalise dtype.
    base["is_event_day"] = base["is_event_day"].astype(bool)
    base["has_city_event"] = base["has_city_event"].astype(bool)
    base["event_count"] = base["event_count"].fillna(0)
    base["is_holiday"] = base["holiday_name"].notna() & (base["holiday_name"] != "")
    base["max_attendance"] = base["max_attendance"].fillna(0)
    return base


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek   # 0=Mon
    df["month"] = df["hour"].dt.month
    df["year"] = df["hour"].dt.year
    df["is_weekend"] = df["day_of_week"] >= 5
    # Real Seattle meter peaks are midday and late afternoon
    df["is_peak_am"] = df["hour_of_day"].between(10, 13)
    df["is_peak_pm"] = df["hour_of_day"].between(17, 19)
    # Cyclical encoding so the model understands hour 23 = hour 0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def write_dashboard_rollups(df: pd.DataFrame):
    """Small committed artifacts so the dashboard never loads the full
    meter-level table (which is far too large for Streamlit Cloud)."""
    daily = (
        df.groupby(["paidparkingarea", "date"], observed=True)
        .agg(
            avg_occupancy_rate=("avg_occupancy_rate", "mean"),
            peak_occupancy_rate=("peak_occupancy_rate", "max"),
            total_spaces=("total_spaces", "sum"),
            n_blockfaces=("blockfacename", "nunique"),
        )
        .reset_index()
        .sort_values(["date", "paidparkingarea"])
    )
    daily.to_csv(AREA_DAILY_FILE, index=False)

    profile = (
        df.groupby(["paidparkingarea", "hour_of_day", "day_of_week"], observed=True)
        .agg(avg_occupancy_rate=("avg_occupancy_rate", "mean"))
        .reset_index()
    )
    profile.to_csv(AREA_PROFILE_FILE, index=False)

    print(f"  Area daily rollup : {len(daily):,} rows -> {AREA_DAILY_FILE.name}")
    print(f"  Area hour profile : {len(profile):,} rows -> {AREA_PROFILE_FILE.name}")


def run():
    print("Loading data sources...")
    registry = load_registry()
    parking  = load_parking()
    weather  = load_weather()
    events   = load_events()
    permits  = load_permits()
    closures = load_closures()
    holidays = load_holidays()

    if parking.empty:
        print("No parking data — cannot build features.")
        return pd.DataFrame()

    print(f"  Registry: {len(registry):,} blockfaces / "
          f"{registry['paidparkingarea'].nunique()} official areas")
    print(f"  Parking:  {len(parking):,} records")
    print(f"  Weather:  {len(weather):,} records")
    print(f"  Events:   {len(events):,} records")
    print(f"  Permits:  {len(permits):,} records")
    print(f"  Holidays: {len(holidays):,} records")
    print(f"  Closures: {len(closures):,} records")

    print("Aggregating parking to hourly per blockface...")
    base = aggregate_parking(parking)
    del parking

    print("Joining meter registry (official areas, coordinates, capacity)...")
    before = len(base)
    base["blockfacename"] = base["blockfacename"].astype(str)
    base = base.merge(
        registry[["blockfacename", "paidparkingarea", "paidparkingsubarea",
                  "lat", "lon", "time_limit", "n_meters", "weather_station"]],
        on="blockfacename", how="inner",
    )
    dropped = before - len(base)
    print(f"  Matched {len(base):,} rows "
          f"({dropped:,} dropped — blockfaces no longer active)")

    print("Joining weather...")
    if not weather.empty:
        base = base.merge(weather, on=["hour", "weather_station"], how="left")

    print("Adding event flags...")
    base = build_event_flags(base, events, permits, holidays, closures)

    print("Adding time features...")
    base = add_time_features(base)

    base = base.sort_values(["hour", "blockfacename"]).reset_index(drop=True)
    base.to_parquet(OUTPUT_FILE, index=False, compression="snappy")

    size_mb = OUTPUT_FILE.stat().st_size / 1e6
    print(f"\nFeatures: {len(base):,} rows x {len(base.columns)} cols "
          f"-> {OUTPUT_FILE.name} ({size_mb:.0f} MB)")
    print(f"  Blockfaces: {base['blockfacename'].nunique():,} | "
          f"Areas: {base['paidparkingarea'].nunique()}")

    print("Writing dashboard rollups...")
    write_dashboard_rollups(base)
    return base


if __name__ == "__main__":
    run()
