# **Seattle Parking and Weather Analysis**

## **Project Overview**
Urban parking in Seattle remains a persistent challenge, with drivers facing unpredictable availability during peak hours. This unpredictability contributes to increased congestion, wasted time, and elevated emissions as vehicles search for available spots. Moreover, external factors—especially weather conditions such as rain or extreme temperatures—can further influence parking occupancy and usage patterns. 

This project integrates real-time weather data with parking data to analyze patterns, provide insights, and help city planners and drivers make informed decisions.

---

## **Team Members**
- **Mohammed Khaja Moinuddin**
- **Nikhil Ghugare**

---

## **Problem Statement**
The lack of an integrated system correlating real-time parking availability with weather conditions leads to inefficiencies in urban mobility. Despite various municipal initiatives, there is no system providing real-time parking occupancy insights based on weather forecasts. This project aims to **analyze and model the impact of weather on parking demand** to improve urban planning and reduce congestion.

---

## **Data Sources**

### **1. Seattle Annual Parking Study Data (CSV)**
- **Source:** [Seattle Open Data Portal](https://data.seattle.gov/Transportation/Annual-Parking-Study-Data/7jzm-ucez/about_data)
- **Format:** CSV
- **Description:** 
  - Historical parking occupancy data across various neighborhoods in Seattle.
  - Provides insights into parking availability trends based on location and time.
  - Used to **identify correlations between weather and parking demand**.
- **Preprocessing Steps:**
  - Handling missing values.
  - Converting timestamps to standard datetime format.
  - Normalizing numerical fields (occupancy rates, demand trends).

### **2. Open-Meteo Weather API (JSON)**
- **Source:** [Open-Meteo API](https://open-meteo.com/)
- **Format:** JSON
- **Description:** 
  - Provides historical weather data (temperature, precipitation, wind speed) for Seattle.
  - Helps **analyze how weather affects parking patterns**.
- **Preprocessing Steps:**
  - Extracting relevant weather parameters.
  - Synchronizing timestamps with parking occupancy records.

---

## **Literature Survey**
1. **A Short-Term Parking Demand Prediction Framework Integrating Overall and Internal Information**  
   - **DOI:** [10.3390/su15097096](https://doi.org/10.3390/su15097096)  
   - **Summary:** This study proposes a **machine learning framework** for predicting short-term parking demand by incorporating external factors (weather, traffic) and internal parking facility data.

2. **Impact of Weather on Urban Transit Ridership**  
   - **DOI:** [10.1016/j.tra.2014.09.008](https://doi.org/10.1016/j.tra.2014.09.008)  
   - **Summary:** The study finds that rain, snow, and extreme temperatures **decrease transit ridership**, leading more people to **switch to personal vehicles**, which increases parking demand.

---

## **Project Structure**

```
Project Work/
│── data/                          # Stores raw & processed datasets
│── logs/                          # Stores logs for processing
│── merged_visualizations/         # Stores merged data visualizations
│── models/                        # Stores trained models
│── notebooks/                     # Jupyter notebooks for analysis
│── reports/                       # Final reports and presentations
│── scripts/                       # All data processing scripts
│── visualizations/                # EDA visualizations
│── .gitignore                     # Files to be ignored by Git
│── README.md                      # Project documentation
│── requirements.txt               # Required Python libraries
│── setup_project.py               # Setup script for the project
```

---

## **Flow of Work**
### **1. Extract**
- **Parking Data:** Downloaded from Seattle Open Data Portal.
- **Weather Data:** Fetched using Open-Meteo API.

### **2. Clean**
- **CSV Data:** Standardized date formats, handled missing values, and filtered required columns.
- **Weather Data:** Extracted relevant fields (temperature, humidity, wind speed) and normalized units.

### **3. Process**
- **Parking Data:** Stored in structured CSV format for analysis.
- **Weather Data:** Stored in JSON format and linked to parking occupancy records.

### **4. Merge & Analyze**
- Merged parking and weather datasets on timestamps.
- Conducted **Exploratory Data Analysis (EDA)** to identify trends.

---

## **Tools & Technologies**
| **Category**          | **Tools/Technologies** |
|----------------------|----------------------|
| Programming         | Python |
| Data Processing    | Pandas, NumPy |
| API Integration    | Requests, JSON |
| Data Storage       | CSV, JSON |
| Visualization      | Matplotlib, Seaborn |
| Machine Learning (Planned) | Scikit-Learn, TensorFlow |

---

## **Key Findings**
- **Weekday vs. Weekend Trends:** Parking occupancy is **higher on weekdays**, with reduced demand on weekends.  
- **Weather Influence:** Rainy and snowy conditions **increase parking demand**, likely due to decreased public transit usage.  
- **Time-of-Day Variation:** Peak demand occurs between **8 AM – 10 AM (morning rush)** and **5 PM – 7 PM (evening rush)**.  

---

## **Next Steps**
- Develop **predictive models** to forecast parking occupancy based on weather conditions.
- Integrate a **real-time dashboard** using **Streamlit** to visualize parking trends.

---

## **Setup Instructions**
### **1. Clone Repository**
```bash
git clone https://github.com/kmohammedsu/seattle-parking-weather-analysis.git
cd seattle-parking-weather-analysis
```

### **2. Create Virtual Environment & Install Dependencies**
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows
pip install -r requirements.txt
```

### **3. Quick Setup with setup_project.py**
If you're setting up the project in a new folder, you can use the provided setup script to automatically create all necessary files and folder structure:

```bash
python setup_project.py
```

This script will create all required directories and placeholder files, allowing you to quickly get started with the project. Note that large data files will need to be downloaded or generated by running the scripts in order.

### **4. Run Data Processing Scripts in Order**
For proper execution of the project, run the scripts in the following order:

```bash
# 1. Clean the parking data
python scripts/data_cleaning.py

# 2. Generate initial visualizations for parking data
python scripts/eda_visualization.py

# 3. Check locations in the dataset
python scripts/locations_check.py

# 4. Check timeframe of the dataset
python scripts/timeframe_check.py

# 5. Fetch weather data from API
python scripts/weather_fetch.py

# 6. Generate visualizations for weather data
python scripts/weather_eda.py

# 7. Merge parking and weather datasets
python scripts/merge.py
```

This sequence ensures that each script has the necessary data from previous steps.

---

## **References**
- **Seattle Open Data Portal** – [Parking Data](https://data.seattle.gov/Transportation/Annual-Parking-Study-Data/7jzm-ucez/about_data)
- **Open-Meteo API** – [Weather Data](https://open-meteo.com/)
- Wang, X., Li, Y., & Zhang, J. (2023). *Sustainability, 15(9), 7096.* [DOI: 10.3390/su15097096](https://doi.org/10.3390/su15097096)
- Guo, Z., Wilson, N. H., & Rahbee, A. (2014). *Transportation Research Part A: Policy and Practice, 64, 154-164.* [DOI: 10.1016/j.tra.2014.09.008](https://doi.org/10.1016/j.tra.2014.09.008)
