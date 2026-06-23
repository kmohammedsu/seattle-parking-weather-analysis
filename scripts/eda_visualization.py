import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Cleaned Data
cleaned_file_path = r"data/Cleaned_Annual_Parking_Study_Data_20250220.csv"
cleaned_df = pd.read_csv(cleaned_file_path, parse_dates=["Date Time"], index_col="Date Time")

# Convert Numeric Columns Properly (Fix Non-Numeric Values)
numeric_cols = ["Parking_Spaces", "Total_Vehicle_Count", "Dp_Count", "Rpz_Count"]
for col in numeric_cols:
    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")  # Coerce non-numeric values to NaN

# Drop Non-Numeric Columns for Aggregation
numeric_df = cleaned_df[numeric_cols]  # Selecting only numeric columns for resampling

# Create a directory for saving visualizations
visualization_dir = "visualizations/parking"
os.makedirs(visualization_dir, exist_ok=True)

# Summary Statistics
eda_summary = numeric_df.describe()
print("EDA Summary Statistics:")
print(eda_summary)

# Save Summary Statistics
eda_summary.to_csv(os.path.join(visualization_dir, "eda_summary.csv"))

# ---- 1. Time-Series Plot for Parking Occupancy Trends ----
plt.figure(figsize=(12, 6))
plt.plot(numeric_df.resample('D').mean().index, numeric_df.resample('D').mean()["Total_Vehicle_Count"],
         label="Daily Avg Vehicle Count", color="blue", alpha=0.6)
plt.xlabel("Time")
plt.ylabel("Vehicle Count")
plt.title("Parking Occupancy Over Time (Daily Average)")
plt.legend()
plt.xticks(rotation=45)
plt.savefig(os.path.join(visualization_dir, "time_series_plot.png"))
plt.close()

# ---- 2. Correlation Heatmap ----
plt.figure(figsize=(8, 5))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Parking Data")
plt.savefig(os.path.join(visualization_dir, "correlation_heatmap.png"))
plt.close()

# ---- 3. Scatter Plot: Parking Spaces vs. Total Vehicle Count ----
plt.figure(figsize=(8, 5))
sns.scatterplot(x=numeric_df["Parking_Spaces"], y=numeric_df["Total_Vehicle_Count"], alpha=0.6, color="red")
plt.xlabel("Parking Spaces")
plt.ylabel("Total Vehicle Count")
plt.title("Parking Spaces vs. Total Vehicle Count")
plt.savefig(os.path.join(visualization_dir, "scatter_plot.png"))
plt.close()

# ---- 4. Weekday vs. Weekend Analysis ----
cleaned_df["Weekday"] = cleaned_df.index.dayofweek  # 0 = Monday, 6 = Sunday
cleaned_df["Weekend"] = cleaned_df["Weekday"].apply(lambda x: "Weekend" if x >= 5 else "Weekday")

plt.figure(figsize=(8, 5))
sns.boxplot(x=cleaned_df["Weekend"], y=cleaned_df["Total_Vehicle_Count"])
plt.xlabel("Day Type")
plt.ylabel("Total Vehicle Count")
plt.title("Weekday vs. Weekend Parking Trends")
plt.savefig(os.path.join(visualization_dir, "weekday_weekend_analysis.png"))
plt.close()

# ---- 5. Time-of-Day Analysis (Rush Hours) ----
cleaned_df["Hour"] = cleaned_df.index.hour
plt.figure(figsize=(10, 5))
sns.lineplot(x=cleaned_df["Hour"], y=cleaned_df["Total_Vehicle_Count"], estimator="mean", marker="o", color="blue")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Vehicle Count")
plt.title("Parking Demand by Hour (Rush Hour Analysis)")
plt.xticks(range(0, 24))
plt.savefig(os.path.join(visualization_dir, "time_of_day_analysis.png"))
plt.close()

print(f"EDA visualizations saved in: {visualization_dir}")
