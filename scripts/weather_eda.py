import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_JSON = ROOT / "data" / "seattle_weather_by_region.json"
OUTPUT_CSV = ROOT / "data" / "processed_weather_data.csv"
VIZ_DIR    = ROOT / "visualizations" / "weather"


def process_weather_data(weather_data: dict) -> pd.DataFrame:
    data_list = []
    for region, region_data in weather_data.items():
        hourly     = region_data.get("hourly", {})
        timestamps = hourly.get("time", [])
        temps      = hourly.get("temperature_2m", [])
        precip     = hourly.get("precipitation", [])
        wind_kmh   = hourly.get("windspeed_10m", [])
        elevation  = region_data.get("elevation", None)

        for ts, t, p, w in zip(timestamps, temps, precip, wind_kmh):
            data_list.append({
                "region":        region,
                "timestamp":     pd.to_datetime(ts),
                "temperature":   t,
                "precipitation": p,
                "wind_speed":    w / 3.6,  # km/h → m/s
                "elevation":     elevation,
            })
    return pd.DataFrame(data_list)


def run():
    os.makedirs(VIZ_DIR, exist_ok=True)

    print(f"Processing weather data from {INPUT_JSON.name}...")
    with open(INPUT_JSON) as f:
        weather_data = json.load(f)

    df = process_weather_data(weather_data)
    df = df.set_index("timestamp")

    print(f"  {len(df):,} hourly records | {df.index.min()} → {df.index.max()}")
    print(df.describe().round(2))

    df.to_csv(OUTPUT_CSV)
    print(f"Saved to {OUTPUT_CSV}")

    df_reset = df.reset_index()

    plt.figure(figsize=(12, 6))
    for region in df_reset["region"].unique():
        sub = df_reset[df_reset["region"] == region].resample("ME", on="timestamp")["temperature"].mean()
        plt.plot(sub.index, sub.values, label=region, alpha=0.8)
    plt.xlabel("Time"); plt.ylabel("Temperature (°C)")
    plt.title("Monthly Average Temperature by Region")
    plt.legend(title="Region")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "weather_temperature_trends.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="region", y="precipitation", data=df_reset)
    plt.xticks(rotation=45); plt.tight_layout()
    plt.title("Precipitation Distribution by Region")
    plt.savefig(VIZ_DIR / "weather_precipitation_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.histplot(df["wind_speed"], bins=40, kde=True)
    plt.xlabel("Wind Speed (m/s)"); plt.title("Wind Speed Distribution")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "weather_wind_speed_distribution.png")
    plt.close()

    print(f"Visualizations saved in {VIZ_DIR}")


if __name__ == "__main__":
    run()
