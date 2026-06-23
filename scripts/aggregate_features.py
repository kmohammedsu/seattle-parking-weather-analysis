import pandas as pd
import numpy as np
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

PARKING_FILE  = DATA_DIR / "live_parking_occupancy.csv"
WEATHER_FILE  = DATA_DIR / "processed_weather_data.csv"
EVENTS_FILE   = DATA_DIR / "seattle_events.csv"
PERMITS_FILE  = DATA_DIR / "seattle_event_permits.csv"
CLOSURES_FILE = DATA_DIR / "seattle_road_closures.csv"
HOLIDAYS_FILE = DATA_DIR / "seattle_holidays.csv"
OUTPUT_FILE   = DATA_DIR / "features.csv"

BLOCKFACE_REGION = {
    "DOWNTOWN":       "Downtown Seattle",
    "CAPITOL HILL":   "Capitol Hill",
    "SOUTH LAKE":     "South Lake Union",
    "BALLARD":        "Ballard",
    "SODO":           "Industrial District",
    "PIONEER":        "Downtown Seattle",
    "FIRST HILL":     "Capitol Hill",
    "UPTOWN":         "Downtown Seattle",
}

VENUE_REGION = {
    "Lumen Field":          "Industrial District",
    "T-Mobile Park":        "Industrial District",
    "Climate Pledge Arena": "Downtown Seattle",
}


def map_blockface_to_region(blockface: pd.Series) -> pd.Series:
    upper = blockface.str.upper().fillna("")
    region = pd.Series("Downtown Seattle", index=blockface.index)
    for keyword, reg in BLOCKFACE_REGION.items():
        region = region.where(~upper.str.contains(keyword), reg)
    return region


def load_parking() -> pd.DataFrame:
    if not PARKING_FILE.exists():
        print("  No live parking data yet — skipping")
        return pd.DataFrame()
    df = pd.read_csv(PARKING_FILE, parse_dates=["occupancy_date"])
    # Reconstruct hourly timestamp from date + hour columns
    df["hour"] = df["occupancy_date"] + pd.to_timedelta(df["occupancy_hour"], unit="h")
    df["region"] = map_blockface_to_region(df["blockfacename"])
    return df


def aggregate_parking(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby(["hour", "region"]).agg(
        avg_occupancy_rate=("occupancy_rate", "mean"),
        peak_occupancy_rate=("peak_occupancy_rate", "max"),
        total_spaces=("avg_spaces", "sum"),
        total_occupied=("avg_occupied", "sum"),
        num_blockfaces=("blockfacename", "nunique"),
    ).reset_index()
    agg["turnover_proxy"] = agg["total_occupied"] / agg["total_spaces"].replace(0, np.nan)
    return agg


def load_weather() -> pd.DataFrame:
    if not WEATHER_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(WEATHER_FILE, parse_dates=["timestamp"])
    df["hour"] = df["timestamp"].dt.floor("h")
    df = df.rename(columns={"region": "region"})
    return df[["hour", "region", "temperature", "precipitation", "wind_speed", "elevation"]]


def load_events() -> pd.DataFrame:
    if not EVENTS_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(EVENTS_FILE, parse_dates=["event_date"])
    df["date"] = df["event_date"].dt.date
    df["region"] = df["venue"].map(VENUE_REGION).fillna("Downtown Seattle")
    return df[["date", "region", "event_name", "venue", "genre"]]


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

    # Sports / concert events
    if not events.empty:
        event_days = events.groupby(["date", "region"]).agg(
            is_event_day=("event_name", lambda x: True),
            event_count=("event_name", "count"),
            event_genres=("genre", lambda x: "|".join(x.dropna().unique())),
        ).reset_index()
        base = base.merge(event_days, on=["date", "region"], how="left")
    else:
        base["is_event_day"] = False
        base["event_count"] = 0
        base["event_genres"] = ""

    # City event permits
    if not permits.empty:
        permit_days = permits.groupby("date").agg(
            has_city_event=("name_of_event", lambda x: True),
            max_attendance=("attendance", "max"),
        ).reset_index()
        base = base.merge(permit_days, on="date", how="left")
    else:
        base["has_city_event"] = False
        base["max_attendance"] = 0

    # Holidays
    if not holidays.empty:
        base = base.merge(holidays, on="date", how="left")
    else:
        base["holiday_name"] = ""
        base["holiday_type"] = ""

    # Road closures — flag if any closure active on this date
    if not closures.empty:
        def has_closure(dt):
            return ((closures["start_date"].dt.date <= dt) &
                    (closures["end_date"].dt.date >= dt)).any()
        closure_dates = set(
            d for d in base["date"].unique() if has_closure(d)
        )
        base["has_road_closure"] = base["date"].isin(closure_dates)
    else:
        base["has_road_closure"] = False

    # Fill booleans
    base["is_event_day"] = base["is_event_day"].fillna(False).infer_objects(copy=False)
    base["has_city_event"] = base["has_city_event"].fillna(False).infer_objects(copy=False)
    base["is_holiday"] = base["holiday_name"].notna() & (base["holiday_name"] != "")
    base["max_attendance"] = base["max_attendance"].fillna(0)

    return base


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek   # 0=Mon
    df["month"] = df["hour"].dt.month
    df["year"] = df["hour"].dt.year
    df["is_weekend"] = df["day_of_week"] >= 5
    df["is_peak_am"] = df["hour_of_day"].between(7, 9)
    df["is_peak_pm"] = df["hour_of_day"].between(16, 19)
    # Cyclical encoding so model understands hour 23 ≈ hour 0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def run():
    print("Loading data sources...")
    parking  = load_parking()
    weather  = load_weather()
    events   = load_events()
    permits  = load_permits()
    holidays = load_holidays()
    closures = load_closures()

    if parking.empty:
        print("No parking data to aggregate. Run fetch_parking.py first.")
        return None

    print(f"  Parking:  {len(parking):,} records")
    print(f"  Weather:  {len(weather):,} records")
    print(f"  Events:   {len(events):,} records")
    print(f"  Permits:  {len(permits):,} records")
    print(f"  Holidays: {len(holidays):,} records")
    print(f"  Closures: {len(closures):,} records")

    print("Aggregating parking to hourly per region...")
    base = aggregate_parking(parking)

    print("Joining weather...")
    if not weather.empty:
        base = base.merge(weather, on=["hour", "region"], how="left")

    print("Adding event flags...")
    base = build_event_flags(base, events, permits, holidays, closures)

    print("Adding time features...")
    base = add_time_features(base)

    # Append to existing features file
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE, parse_dates=["hour"])
        combined = pd.concat([existing, base]).drop_duplicates(
            subset=["hour", "region"]
        ).sort_values(["hour", "region"])
    else:
        combined = base.sort_values(["hour", "region"])

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFeatures saved: {len(combined):,} rows × {len(combined.columns)} cols → {OUTPUT_FILE}")
    return combined


if __name__ == "__main__":
    run()
