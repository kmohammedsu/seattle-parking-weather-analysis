import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
input_json_path = "data/seattle_weather_by_region.json"
output_folder = "visualizations/weather"
processed_weather_file = "data/processed_weather_data.csv"

# Ensure visualization directory exists
os.makedirs(output_folder, exist_ok=True)

# Load the saved JSON file
with open(input_json_path, "r") as json_file:
    weather_data = json.load(json_file)

# Convert JSON data into a structured DataFrame
def process_weather_data(weather_data):
    data_list = []
    for region, region_data in weather_data.items():
        hourly_data = region_data.get("hourly", {})
        timestamps = hourly_data.get("time", [])
        temperatures = hourly_data.get("temperature_2m", [])
        precipitation = hourly_data.get("precipitation", [])
        wind_speed_kmh = hourly_data.get("windspeed_10m", [])
        elevation = region_data.get("elevation", None)  # Capture elevation data

        for i in range(len(timestamps)):
            data_list.append({
                "region": region,
                "timestamp": pd.to_datetime(timestamps[i]),
                "temperature": temperatures[i],
                "precipitation": precipitation[i],
                "wind_speed": wind_speed_kmh[i] / 3.6,  # Convert km/h to m/s
                "elevation": elevation  # Store elevation per region
            })
    return pd.DataFrame(data_list)

weather_df = process_weather_data(weather_data)
weather_df.set_index("timestamp", inplace=True)

# Summary statistics
print("\nSummary Statistics for Weather Data:")
print(weather_df.describe())

# Save processed weather data
weather_df.to_csv(processed_weather_file)
print(f"Processed weather data saved at: {processed_weather_file}")

# Temperature trends over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=weather_df, x=weather_df.index, y="temperature", hue="region", alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Trends Over Time by Region")
plt.legend(title="Region")
plt.savefig(os.path.join(output_folder, "weather_temperature_trends.png"))
plt.close()

# Precipitation impact
plt.figure(figsize=(12, 6))
sns.boxplot(x="region", y="precipitation", data=weather_df)
plt.xlabel("Region")
plt.ylabel("Precipitation (mm)")
plt.title("Precipitation Distribution by Region")
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_folder, "weather_precipitation_distribution.png"))
plt.close()

# Wind Speed Analysis (Converted to m/s)
plt.figure(figsize=(12, 6))
sns.histplot(weather_df["wind_speed"], bins=30, kde=True)
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Frequency")
plt.title("Wind Speed Distribution Across All Regions (m/s)")
plt.savefig(os.path.join(output_folder, "weather_wind_speed_distribution.png"))
plt.close()

# Elevation Analysis
plt.figure(figsize=(8, 6))
sns.boxplot(x="region", y="elevation", data=weather_df)
plt.xlabel("Region")
plt.ylabel("Elevation (m)")
plt.title("Elevation Levels by Region")
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_folder, "weather_elevation_by_region.png"))
plt.close()

print(f"All visualizations saved in: {output_folder}")
