# Seattle Parking Revenue Intelligence Platform

A city decision-support tool for the City of Seattle to optimize parking infrastructure and meter pricing under **Seattle Municipal Code 11.16.121** (Performance-Based Parking Pricing).

Live data pipeline · LightGBM demand forecasting · Dynamic pricing recommendations · Streamlit dashboard

---

## What It Does

- **Ingests live data daily** — paid parking occupancy (Socrata), weather (Open-Meteo), sports events (Ticketmaster), city permits, road closures, and holidays
- **Trains a LightGBM model** on rolling 12-month historical occupancy data to predict demand
- **Recommends meter rate adjustments** to hit the city's 70–85% occupancy target
- **Projects revenue impact** of dynamic pricing with demand elasticity modeling
- **Displays everything** in a 5-page Streamlit dashboard (Overview, Revenue, Forecast, Pricing, Geo Map)

---

## Architecture

```
GitHub Actions (daily, 8 AM UTC)
    │
    ├── fetch_parking.py       → Socrata live occupancy (last 30 days, server-side aggregated)
    ├── weather_fetch.py       → Open-Meteo historical + current weather
    ├── fetch_events.py        → Ticketmaster sports/concerts (Lumen Field, T-Mobile Park, Climate Pledge)
    ├── fetch_permits.py       → Seattle Special Event Permits (Socrata dm95-f8w5)
    ├── fetch_closures.py      → Seattle Road Closures (Socrata ium9-iqtc)
    ├── holidays_features.py   → WA state holidays (python holidays library)
    ├── aggregate_features.py  → blockface → region aggregation, feature engineering
    ├── train_model.py         → LightGBM rolling 12-month training (leakage-free)
    ├── pricing_optimizer.py   → dynamic pricing recommendations (SMC 11.16.121)
    └── revenue_analyzer.py    → P&L analysis, committed to repo as CSV
         │
         ▼
    data/features.csv          (committed, grows daily)
    data/pricing_recommendations.csv
    data/revenue_summary.csv
    models/parking_demand_lgbm.pkl (not committed — re-trained daily)
```

Zero-cost infrastructure: GitHub Actions free tier (2000 min/month) + Streamlit Community Cloud.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/kmohammedsu/seattle-parking-weather-analysis.git
cd seattle-parking-weather-analysis
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set API key

```bash
export TICKETMASTER_API_KEY=your_key_here
```

For GitHub Actions: add `TICKETMASTER_API_KEY` as a repository secret.

### 3. Historical backfill (one-time, ~4 hours)

The daily pipeline only fetches the last 30 days. To train on years of history:

```bash
python scripts/backfill_parking.py   # fetches 2018–2025 from Socrata
python scripts/aggregate_features.py # rebuilds features.csv from full history
```

Progress saves after each month — safe to interrupt and resume.

### 4. Run the full pipeline

```bash
python run_pipeline.py
```

### 5. Launch the dashboard

```bash
streamlit run streamlit_app.py
```

---

## Key Data Insights (from EDA)

| Finding | Detail |
|---------|--------|
| **Real peak hours** | 11am–1pm and 5pm–7pm (NOT 7–9am AM rush — street meters, not garages) |
| **Busiest day** | Saturday (~33% avg occupancy, meters free Sunday) |
| **Weather impact** | Temperature and precipitation have <5% correlation with occupancy |
| **Event impact** | Needs historical event data overlapping with parking dates to quantify |
| **Top demand areas** | Downtown Seattle, Capitol Hill, South Lake Union |

---

## Model

- **Algorithm**: LightGBM (supports incremental `init_model` warm-start)
- **Target**: `avg_occupancy_rate` (0–1) per region per hour
- **Features**: time cyclical encoding, weather, event flags, space counts (no leakage)
- **Training window**: rolling 12 months (stays current without growing compute)
- **Current R²**: 0.958 (leakage-free, trained on 30 days — improves significantly with backfill)

Leakage note: `turnover_proxy` and `peak_occupancy_rate` are excluded — they are derived from the same raw readings as the target.

---

## Dynamic Pricing (SMC 11.16.121)

Seattle's Performance-Based Parking Pricing Program (established 2010) authorizes the city to adjust meter rates based on observed demand. Rate bounds: **$0.50–$8.00/hour**. Changes require City Council approval.

This tool accelerates the existing quarterly review process by:
1. Predicting demand by block area and hour
2. Flagging zones consistently above 85% (raise rate) or below 70% (lower rate)
3. Estimating revenue impact using 5% demand elasticity per $0.25 change

---

## Project Structure

```
├── dashboard/
│   └── app.py                  # Streamlit 5-page dashboard
├── data/                       # features.csv, pricing CSVs (gitignored: raw CSVs)
├── models/                     # model artifacts, performance metrics
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model.ipynb
├── scripts/
│   ├── backfill_parking.py     # one-time historical data fetch (2018–2025)
│   ├── fetch_parking.py        # daily live occupancy (Socrata)
│   ├── weather_fetch.py        # Open-Meteo weather
│   ├── fetch_events.py         # Ticketmaster events
│   ├── fetch_permits.py        # city event permits
│   ├── fetch_closures.py       # road closures
│   ├── holidays_features.py    # WA state holidays
│   ├── aggregate_features.py   # feature engineering
│   ├── train_model.py          # LightGBM training
│   ├── pricing_optimizer.py    # dynamic pricing logic
│   └── revenue_analyzer.py     # P&L analysis
├── tests/                      # pytest unit tests
├── .github/workflows/
│   ├── daily_pipeline.yml      # scheduled data + model refresh
│   └── tests.yml               # CI on every push
├── run_pipeline.py             # orchestrator (10 steps)
├── streamlit_app.py            # Streamlit Cloud entry point
├── Dockerfile                  # container deployment
└── requirements.txt
```

---

## Data Sources

| Source | Dataset | Endpoint |
|--------|---------|----------|
| Seattle Paid Parking | Occupancy (2018–present) | Socrata per-year datasets |
| Seattle Paid Parking | Live (last 30 days) | `rke9-rsvs` |
| Weather | Historical + forecast | Open-Meteo API |
| Sports/Concerts | Venue events | Ticketmaster Discovery API |
| City Events | Special Event Permits | Socrata `dm95-f8w5` |
| Road Closures | Street Closure Permits | Socrata `ium9-iqtc` |
| Holidays | WA state holidays | `holidays` Python library |

---

## References

- Wang, X. et al. (2023). *Sustainability, 15(9), 7096.* [DOI: 10.3390/su15097096](https://doi.org/10.3390/su15097096)
- Guo, Z. et al. (2014). *Transportation Research Part A, 64, 154–164.* [DOI: 10.1016/j.tra.2014.09.008](https://doi.org/10.1016/j.tra.2014.09.008)
- Seattle Municipal Code 11.16.121 — Performance-Based Parking Pricing Program
