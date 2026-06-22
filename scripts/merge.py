import pandas as pd
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARKING_FILE = ROOT / "data" / "Cleaned_Annual_Parking_Study_Data_20250220.csv"
WEATHER_FILE = ROOT / "data" / "processed_weather_data.csv"
OUTPUT_FILE = ROOT / "data" / "merged_parking_weather_data.csv"
VIZ_DIR = ROOT / "merged_visualizations"

REGION_MAP = {
    "South Lake Union": "South Lake Union",
    "Downtown": "Downtown Seattle",
    "Capitol Hill": "Capitol Hill",
    "Ballard": "Ballard",
    "Sodo": "Industrial District",
    "Industrial District": "Industrial District",
    "Pioneer Square": "Downtown Seattle",
    "First Hill": "Capitol Hill",
    "Uptown": "Downtown Seattle",
    "12Th Avenue": "Capitol Hill",
    "Cherry Hill": "Capitol Hill",
}


def load_parking(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date Time"], index_col="Date Time")
    df.index = df.index.floor("h")
    # Map Study_Area to weather region
    df["weather_region"] = df["Study_Area"].map(REGION_MAP).fillna("Downtown Seattle")
    return df


def load_weather(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.rename(columns={"timestamp": "Date Time"})
    df["Date Time"] = df["Date Time"].dt.floor("h")
    df = df.rename(columns={"region": "weather_region"})
    return df


def merge(parking: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    parking = parking.reset_index()
    merged = pd.merge(
        parking,
        weather,
        on=["Date Time", "weather_region"],
        how="inner"
    )
    return merged


def run():
    print("Loading parking data...")
    parking = load_parking(PARKING_FILE)
    print(f"  Parking shape: {parking.shape}")

    print("Loading weather data...")
    weather = load_weather(WEATHER_FILE)
    print(f"  Weather shape: {weather.shape}")

    print("Merging on Date Time + region...")
    merged = merge(parking, weather)
    print(f"  Merged shape: {merged.shape}")

    os.makedirs(VIZ_DIR, exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")
    return merged


if __name__ == "__main__":
    run()
