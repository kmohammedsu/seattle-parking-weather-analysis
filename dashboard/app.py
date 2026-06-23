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
import folium
from streamlit_folium import st_folium

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
def load_roi():
    f = DATA_DIR / "infrastructure_roi.csv"
    if not f.exists():
        return pd.DataFrame()
    return pd.read_csv(f)


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

    # Try local first, fall back to Hugging Face Hub
    if not model_file.exists():
        try:
            import sys
            sys.path.insert(0, str(ROOT))
            from scripts.upload_model import download_model
            download_model(MODELS_DIR)
        except Exception:
            pass

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
    ["Overview", "Revenue", "Forecast", "Pricing", "Infrastructure ROI", "Geo Map"],
    label_visibility="collapsed",
)

features = load_features()
pricing = load_pricing()
rev_summary = load_revenue_summary()
rev_detail = load_revenue_detail()
roi_df = load_roi()
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

    # Occupancy by hour heatmap — reindex to all 7 days in case some are missing
    DAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    pivot = (
        features
        .groupby(["hour_of_day", "day_of_week"])["avg_occupancy_rate"]
        .mean()
        .unstack(fill_value=0)
        .reindex(columns=range(7), fill_value=0)
    )
    pivot.columns = [DAY_NAMES[c] for c in pivot.columns]
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
        styled.style.map(color_action, subset=["Action"]),
        use_container_width=True, hide_index=True,
    )

# ── Page: Infrastructure ROI ─────────────────────────────────────────────────

elif page == "Infrastructure ROI":
    st.title("🏗️ Infrastructure ROI Calculator")
    st.caption("Should Seattle build more parking? Cost-benefit analysis by area and infrastructure type.")

    if roi_df.empty:
        st.warning("No ROI data. Run `python scripts/infrastructure_roi.py`.")
        st.stop()

    # Summary KPIs
    viable = roi_df[roi_df["viable"]]
    col1, col2, col3 = st.columns(3)
    col1.metric("Scenarios Analyzed", len(roi_df))
    col2.metric("Viable Investments", len(viable), help="Breakeven occupancy < current occupancy")
    col3.metric("Best ROI", f"{viable['roi_percent'].max():.1f}%" if not viable.empty else "N/A")

    st.divider()

    # Filter controls
    col1, col2 = st.columns(2)
    region_filter = col1.selectbox("Region", ["All"] + sorted(roi_df["region"].unique()))
    infra_filter = col2.selectbox("Infrastructure Type", ["All"] + sorted(roi_df["infra_type"].unique()))

    disp = roi_df.copy()
    if region_filter != "All":
        disp = disp[disp["region"] == region_filter]
    if infra_filter != "All":
        disp = disp[disp["infra_type"] == infra_filter]

    # ROI chart
    fig = px.bar(
        disp.sort_values("roi_percent", ascending=False),
        x="region", y="roi_percent",
        color="infra_type",
        barmode="group",
        facet_col="n_new_spaces",
        title="ROI % by Region and Infrastructure Type",
        labels={"roi_percent": "ROI %", "region": "Region", "infra_type": "Type"},
        height=400,
        color_discrete_map={"surface_lot": "steelblue", "structured_garage": "darkorange"},
    )
    fig.add_hline(y=0, line_dash="solid", line_color="red", line_width=1)
    st.plotly_chart(fig, use_container_width=True)

    # Breakeven comparison
    fig2 = px.scatter(
        disp,
        x="current_occupancy", y="breakeven_occupancy",
        color="infra_type", size="n_new_spaces",
        hover_name="region",
        hover_data={"roi_percent": True, "payback_years": True},
        title="Current Occupancy vs Breakeven Occupancy",
        labels={"current_occupancy": "Current Occupancy",
                "breakeven_occupancy": "Breakeven Occupancy"},
        height=400,
    )
    fig2.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                   line=dict(dash="dash", color="gray"))
    fig2.update_xaxes(tickformat=".0%")
    fig2.update_yaxes(tickformat=".0%")
    st.caption("Points below the diagonal line are viable (breakeven < current demand).")
    st.plotly_chart(fig2, use_container_width=True)

    # Detailed table
    st.subheader("Scenario Details")
    table = disp[[
        "region", "infra_type", "n_new_spaces", "current_occupancy",
        "breakeven_occupancy", "total_construction_cost",
        "net_annual_income", "roi_percent", "payback_years", "viable"
    ]].copy()
    table["current_occupancy"] = table["current_occupancy"].map("{:.1%}".format)
    table["breakeven_occupancy"] = table["breakeven_occupancy"].map("{:.1%}".format)
    table["total_construction_cost"] = table["total_construction_cost"].map("${:,.0f}".format)
    table["net_annual_income"] = table["net_annual_income"].map("${:+,.0f}".format)
    table["roi_percent"] = table["roi_percent"].map("{:.1f}%".format)
    table.columns = ["Region", "Type", "Spaces", "Curr. Occ.", "Breakeven Occ.",
                     "Construction Cost", "Net Annual", "ROI %", "Payback (yrs)", "Viable"]
    st.dataframe(table.style.map(
        lambda v: "background-color: #e8f4fd" if v is True else
                  ("background-color: #fde8e8" if v is False else ""),
        subset=["Viable"]
    ), use_container_width=True, hide_index=True)

    st.caption(
        "Construction costs: surface lot $5K/space, structured garage $45K/space "
        "(Seattle 2024 benchmarks). Bond rate 4.5%, 20-year term, $800/space/yr operating."
    )

# ── Page: Geo Map ────────────────────────────────────────────────────────────

elif page == "Geo Map":
    st.title("🗺️ Parking Demand Map")
    st.caption("Average occupancy by region — circle size and color indicate demand level")

    if features.empty:
        st.warning("No data to display.")
        st.stop()

    REGION_COORDS = {
        "Downtown Seattle":    (47.6062, -122.3321),
        "Capitol Hill":        (47.6253, -122.3222),
        "South Lake Union":    (47.6278, -122.3367),
        "Ballard":             (47.6677, -122.3833),
        "Industrial District": (47.5873, -122.3294),
    }

    region_occ = (
        features.groupby("region")["avg_occupancy_rate"]
        .mean()
        .reset_index()
    )

    def occ_color(rate: float) -> str:
        if rate > 0.85:
            return "#c0392b"   # red — over target
        elif rate > 0.70:
            return "#27ae60"   # green — in target band
        else:
            return "#2980b9"   # blue — under target

    m = folium.Map(location=[47.615, -122.335], zoom_start=12, tiles="CartoDB positron")

    for _, row in region_occ.iterrows():
        coords = REGION_COORDS.get(row["region"], (47.61, -122.33))
        rate = row["avg_occupancy_rate"]
        radius = 200 + rate * 600  # 200m min, up to 800m at 100%
        color = occ_color(rate)

        folium.CircleMarker(
            location=coords,
            radius=radius / 20,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=folium.Popup(
                f"<b>{row['region']}</b><br>Avg Occupancy: {rate:.1%}<br>"
                f"Target: 70–85%",
                max_width=200,
            ),
            tooltip=f"{row['region']}: {rate:.1%}",
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:10px 14px;border-radius:8px;border:1px solid #ccc;font-size:13px;">
        <b>Occupancy Level</b><br>
        <span style="color:#c0392b;">●</span> &gt;85% — Over target (raise rates)<br>
        <span style="color:#27ae60;">●</span> 70–85% — In target band<br>
        <span style="color:#2980b9;">●</span> &lt;70% — Under target (lower rates)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    col1, col2 = st.columns([3, 1])
    with col1:
        st_folium(m, width=800, height=550)
    with col2:
        st.markdown("### Region Stats")
        for _, row in region_occ.sort_values("avg_occupancy_rate", ascending=False).iterrows():
            rate = row["avg_occupancy_rate"]
            icon = "🔴" if rate > 0.85 else ("🟢" if rate > 0.70 else "🔵")
            st.markdown(f"**{icon} {row['region']}**")
            st.markdown(f"&nbsp;&nbsp;{rate:.1%} avg occupancy")
            st.markdown("")
