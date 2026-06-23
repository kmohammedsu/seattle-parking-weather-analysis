"""
Seattle Parking Revenue Intelligence Platform
City of Seattle Decision-Support Dashboard

Pages:
  1. Overview       — live occupancy snapshot across all regions
  2. Revenue        — current vs optimized revenue analysis
  3. Forecast       — 7-day demand predictions by region
  4. Pricing        — dynamic pricing recommendations (SMC 11.16.121)
  5. Geo Map        — blockface-level occupancy heatmap
"""
import pickle
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

st.set_page_config(
    page_title="Seattle Parking Intelligence",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared loaders (cached) ──────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_features():
    f = DATA_DIR / "features.csv"
    if not f.exists():
        return pd.DataFrame()
    return pd.read_csv(f, parse_dates=["hour"])


@st.cache_data(ttl=3600)
def load_revenue_summary():
    f = DATA_DIR / "revenue_summary.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f, parse_dates=["date"])
    return df


@st.cache_data(ttl=3600)
def load_revenue_detail():
    f = DATA_DIR / "revenue_by_region_hour.csv"
    if not f.exists():
        return pd.DataFrame()
    return pd.read_csv(f)


@st.cache_data(ttl=3600)
def load_pricing():
    f = DATA_DIR / "pricing_recommendations.csv"
    if not f.exists():
        return pd.DataFrame()
    return pd.read_csv(f)


@st.cache_resource
def load_model():
    model_file = MODELS_DIR / "parking_demand_lgbm.pkl"
    feat_file = MODELS_DIR / "feature_columns.json"
    if not model_file.exists():
        return None, None
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    feat_cols = json.loads(feat_file.read_text()) if feat_file.exists() else []
    return model, feat_cols


@st.cache_data(ttl=3600)
def load_perf():
    f = MODELS_DIR / "performance" / "final_model_performance.csv"
    if not f.exists():
        return {}
    return pd.read_csv(f).iloc[0].to_dict()


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.markdown("## 🏙️ City of Seattle")
st.sidebar.title("Parking Intelligence")
st.sidebar.caption("Revenue Optimization Platform")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Revenue", "Forecast", "Pricing", "Geo Map"],
    label_visibility="collapsed",
)

features = load_features()
pricing = load_pricing()
rev_summary = load_revenue_summary()
rev_detail = load_revenue_detail()
model, feat_cols = load_model()
perf = load_perf()

last_updated = features["hour"].max().strftime("%b %d %Y %H:%M") if not features.empty else "No data"
st.sidebar.divider()
st.sidebar.caption(f"Data through: **{last_updated}**")
st.sidebar.caption(f"Model R²: **{perf.get('r2', 'N/A'):.3f}**" if perf else "Model: not trained")

# ── Page: Overview ───────────────────────────────────────────────────────────

if page == "Overview":
    st.title("🅿️ Seattle Parking — Live Overview")
    st.caption("Hourly occupancy across all metered regions")

    if features.empty:
        st.warning("No feature data available. Run the pipeline first.")
        st.stop()

    recent = features[features["hour"] >= features["hour"].max() - pd.Timedelta(hours=48)]

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_occ = features["avg_occupancy_rate"].mean()
        st.metric("Avg Occupancy (all time)", f"{avg_occ:.1%}")
    with col2:
        recent_occ = recent["avg_occupancy_rate"].mean()
        st.metric("Avg Occupancy (48h)", f"{recent_occ:.1%}")
    with col3:
        n_regions = features["region"].nunique()
        st.metric("Regions Tracked", n_regions)
    with col4:
        n_days = features["hour"].dt.date.nunique()
        st.metric("Days of Data", f"{n_days:,}")

    st.divider()

    # Occupancy by hour heatmap
    pivot = (
        features
        .groupby(["hour_of_day", "day_of_week"])["avg_occupancy_rate"]
        .mean()
        .unstack(fill_value=0)
    )
    pivot.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = px.imshow(
        pivot,
        labels={"x": "Day of Week", "y": "Hour of Day", "color": "Occupancy"},
        color_continuous_scale="RdYlGn_r",
        zmin=0, zmax=1,
        title="Average Occupancy: Hour of Day × Day of Week",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # By region
    region_occ = (
        features.groupby("region")["avg_occupancy_rate"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig2 = px.bar(
        region_occ, x="region", y="avg_occupancy_rate",
        title="Average Occupancy by Region",
        labels={"avg_occupancy_rate": "Occupancy Rate", "region": "Region"},
        color="avg_occupancy_rate",
        color_continuous_scale="RdYlGn_r",
    )
    fig2.add_hline(y=0.85, line_dash="dash", line_color="red",
                   annotation_text="85% target upper", annotation_position="top right")
    fig2.add_hline(y=0.70, line_dash="dash", line_color="orange",
                   annotation_text="70% target lower", annotation_position="bottom right")
    fig2.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig2, use_container_width=True)

# ── Page: Revenue ────────────────────────────────────────────────────────────

elif page == "Revenue":
    st.title("💰 Revenue Intelligence")
    st.caption("Current vs optimized revenue under dynamic pricing (SMC 11.16.121)")

    if rev_summary.empty:
        st.warning("Revenue data not yet generated. Run `python scripts/revenue_analyzer.py`.")
        st.stop()

    total_current = rev_summary["current_revenue"].sum()
    total_optimized = rev_summary["optimized_revenue"].sum()
    total_uplift = total_optimized - total_current
    uplift_pct = total_uplift / max(total_current, 0.01) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue (Current Rates)", f"${total_current:,.0f}")
    col2.metric("Total Revenue (Optimized Rates)", f"${total_optimized:,.0f}")
    col3.metric("Potential Uplift", f"${total_uplift:+,.0f}", f"{uplift_pct:+.1f}%")

    st.divider()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rev_summary["date"], y=rev_summary["current_revenue"],
                             name="Current Rates", line=dict(color="steelblue")))
    fig.add_trace(go.Scatter(x=rev_summary["date"], y=rev_summary["optimized_revenue"],
                             name="Optimized Rates", line=dict(color="green", dash="dash")))
    fig.update_layout(title="Daily Revenue: Current vs Optimized",
                      yaxis_title="Revenue ($)", xaxis_title="Date", height=350)
    st.plotly_chart(fig, use_container_width=True)

    if not rev_detail.empty:
        pivot = rev_detail.pivot_table(
            index="hour_of_day", columns="region",
            values="revenue_uplift", aggfunc="mean"
        )
        fig2 = px.imshow(
            pivot,
            labels={"x": "Region", "y": "Hour", "color": "Uplift ($/hr)"},
            color_continuous_scale="RdYlGn",
            title="Revenue Uplift Opportunity by Region × Hour",
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Page: Forecast ───────────────────────────────────────────────────────────

elif page == "Forecast":
    st.title("📈 Demand Forecast")
    st.caption("7-day occupancy predictions by region")

    if model is None:
        st.warning("No trained model found. Run `python scripts/train_model.py` first.")
        st.stop()

    if features.empty:
        st.warning("No feature data. Run the pipeline first.")
        st.stop()

    # Build next-7-days prediction frame
    last_hour = features["hour"].max()
    future_hours = pd.date_range(
        start=last_hour + pd.Timedelta(hours=1),
        periods=7 * 24, freq="h"
    )
    regions = features["region"].unique()
    future = pd.MultiIndex.from_product([future_hours, regions], names=["hour", "region"])
    future_df = pd.DataFrame(index=future).reset_index()

    future_df["hour_of_day"] = future_df["hour"].dt.hour
    future_df["day_of_week"] = future_df["hour"].dt.dayofweek
    future_df["month"] = future_df["hour"].dt.month
    future_df["year"] = future_df["hour"].dt.year
    future_df["is_weekend"] = future_df["day_of_week"] >= 5
    future_df["is_peak_am"] = future_df["hour_of_day"].between(10, 13)
    future_df["is_peak_pm"] = future_df["hour_of_day"].between(17, 19)
    future_df["hour_sin"] = np.sin(2 * np.pi * future_df["hour_of_day"] / 24)
    future_df["hour_cos"] = np.cos(2 * np.pi * future_df["hour_of_day"] / 24)
    future_df["dow_sin"] = np.sin(2 * np.pi * future_df["day_of_week"] / 7)
    future_df["dow_cos"] = np.cos(2 * np.pi * future_df["day_of_week"] / 7)
    future_df["month_sin"] = np.sin(2 * np.pi * future_df["month"] / 12)
    future_df["month_cos"] = np.cos(2 * np.pi * future_df["month"] / 12)

    # Fill space counts from historical averages
    space_avgs = features.groupby("region")["total_spaces"].mean()
    future_df["total_spaces"] = future_df["region"].map(space_avgs).fillna(100)
    future_df["num_blockfaces"] = features.groupby("region")["num_blockfaces"].mean().reindex(
        future_df["region"]).values

    # Fill weather and event cols with 0/False (forward-looking — we don't have them yet)
    for col in feat_cols:
        if col not in future_df.columns:
            future_df[col] = 0

    for col in feat_cols:
        if future_df[col].dtype == bool:
            future_df[col] = future_df[col].astype(int)

    available = [c for c in feat_cols if c in future_df.columns]
    future_df["predicted_occupancy"] = model.predict(future_df[available].fillna(0))
    future_df["predicted_occupancy"] = future_df["predicted_occupancy"].clip(0, 1)

    selected_region = st.selectbox("Select Region", sorted(regions))
    region_forecast = future_df[future_df["region"] == selected_region].copy()

    fig = px.line(
        region_forecast, x="hour", y="predicted_occupancy",
        title=f"7-Day Occupancy Forecast — {selected_region}",
        labels={"predicted_occupancy": "Predicted Occupancy", "hour": ""},
        height=400,
    )
    fig.add_hline(y=0.85, line_dash="dash", line_color="red", annotation_text="85% (reduce supply pressure)")
    fig.add_hline(y=0.70, line_dash="dash", line_color="orange", annotation_text="70% (stimulate demand)")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    # Peak hours table
    peak_hours = (
        region_forecast[region_forecast["predicted_occupancy"] > 0.85]
        [["hour", "predicted_occupancy"]]
        .assign(predicted_occupancy=lambda x: x["predicted_occupancy"].map("{:.1%}".format))
        .rename(columns={"hour": "Time", "predicted_occupancy": "Predicted Occupancy"})
    )
    if not peak_hours.empty:
        st.subheader(f"High-demand hours (>85%) — {selected_region}")
        st.dataframe(peak_hours, use_container_width=True, hide_index=True)

# ── Page: Pricing ────────────────────────────────────────────────────────────

elif page == "Pricing":
    st.title("🎯 Dynamic Pricing Recommendations")
    st.caption("Performance-Based Parking Pricing — Seattle Municipal Code 11.16.121")

    st.info(
        "Seattle SMC 11.16.121 authorizes adjusting meter rates to achieve 70–85% occupancy targets. "
        "Rates are bounded between $0.50 and $8.00/hour. Changes require City Council approval.",
        icon="⚖️",
    )

    if pricing.empty:
        st.warning("No pricing recommendations. Run `python scripts/pricing_optimizer.py`.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    increases = (pricing["action"] == "increase").sum()
    decreases = (pricing["action"] == "decrease").sum()
    holds = (pricing["action"] == "hold").sum()
    col1.metric("Rate Increases Recommended", increases, help="Drive away excess demand")
    col2.metric("Rate Decreases Recommended", decreases, help="Stimulate under-utilized zones")
    col3.metric("No Change", holds)

    st.divider()

    region_filter = st.selectbox("Filter by Region", ["All"] + sorted(pricing["region"].unique()))
    disp = pricing if region_filter == "All" else pricing[pricing["region"] == region_filter]

    def color_action(val):
        if val == "increase":
            return "background-color: #fde8e8; color: #c0392b"
        if val == "decrease":
            return "background-color: #e8f4fd; color: #2980b9"
        return ""

    styled = disp[["region", "hour_of_day", "avg_predicted_occupancy",
                    "current_rate", "recommended_rate", "rate_change",
                    "action", "revenue_delta"]].copy()
    styled["avg_predicted_occupancy"] = styled["avg_predicted_occupancy"].map("{:.1%}".format)
    styled["current_rate"] = styled["current_rate"].map("${:.2f}".format)
    styled["recommended_rate"] = styled["recommended_rate"].map("${:.2f}".format)
    styled["rate_change"] = styled["rate_change"].map("${:+.2f}".format)
    styled["revenue_delta"] = styled["revenue_delta"].map("${:+.2f}".format)
    styled.columns = ["Region", "Hour", "Pred. Occupancy", "Current Rate",
                      "Rec. Rate", "Change", "Action", "Rev. Delta/hr"]

    st.dataframe(
        styled.style.applymap(color_action, subset=["Action"]),
        use_container_width=True, hide_index=True,
    )

# ── Page: Geo Map ────────────────────────────────────────────────────────────

elif page == "Geo Map":
    st.title("🗺️ Blockface Occupancy Map")
    st.caption("Geographic distribution of parking demand — last 30 days")

    if features.empty:
        st.warning("No data to display.")
        st.stop()

    # Region centroids (approximate)
    REGION_COORDS = {
        "Downtown Seattle":   (47.6062, -122.3321),
        "Capitol Hill":       (47.6253, -122.3222),
        "South Lake Union":   (47.6278, -122.3367),
        "Ballard":            (47.6677, -122.3833),
        "Industrial District":(47.5873, -122.3294),
    }

    region_occ = (
        features.groupby("region")["avg_occupancy_rate"]
        .mean()
        .reset_index()
    )
    region_occ["lat"] = region_occ["region"].map(lambda r: REGION_COORDS.get(r, (47.61, -122.33))[0])
    region_occ["lon"] = region_occ["region"].map(lambda r: REGION_COORDS.get(r, (47.61, -122.33))[1])
    region_occ["occupancy_pct"] = (region_occ["avg_occupancy_rate"] * 100).round(1)

    fig = px.scatter_mapbox(
        region_occ,
        lat="lat", lon="lon",
        size="occupancy_pct",
        color="avg_occupancy_rate",
        color_continuous_scale="RdYlGn_r",
        hover_name="region",
        hover_data={"occupancy_pct": True, "lat": False, "lon": False},
        zoom=11,
        center={"lat": 47.615, "lon": -122.33},
        mapbox_style="carto-positron",
        title="Average Occupancy by Region",
        height=600,
        size_max=50,
    )
    fig.update_layout(coloraxis_colorbar=dict(tickformat=".0%"))
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Bubble size = occupancy rate. Red = high demand, Green = low demand.")
