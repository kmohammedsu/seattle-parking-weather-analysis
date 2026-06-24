"""
Seattle Parking Revenue Intelligence Platform
City of Seattle Decision-Support Dashboard
"""
import pickle
import json
from pathlib import Path

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

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Main background */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #2a2f3e; }

/* Remove default padding */
.block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }

/* KPI card */
.kpi-card {
    background: #1a2035;
    border: 1px solid #2a3550;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    height: 100%;
}
.kpi-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7b8db0;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 32px;
    font-weight: 700;
    color: #e8eaf6;
    line-height: 1.1;
}
.kpi-delta {
    font-size: 13px;
    margin-top: 6px;
    font-weight: 600;
}
.kpi-good  { color: #4caf8f; }
.kpi-warn  { color: #f5a623; }
.kpi-bad   { color: #e05c5c; }
.kpi-neutral { color: #7b8db0; }

/* Section header */
.section-header {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4a90d9;
    border-bottom: 1px solid #2a3550;
    padding-bottom: 6px;
    margin: 24px 0 16px 0;
}

/* Status badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.badge-red    { background: #3d1515; color: #e05c5c; border: 1px solid #5c2020; }
.badge-green  { background: #153d2a; color: #4caf8f; border: 1px solid #205c3e; }
.badge-blue   { background: #152a3d; color: #4a90d9; border: 1px solid #20405c; }
.badge-yellow { background: #3d2e15; color: #f5a623; border: 1px solid #5c4420; }

/* Sidebar nav */
[data-testid="stSidebar"] .stRadio label {
    font-size: 14px !important;
    font-weight: 500;
}

/* Hide streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def kpi(label: str, value: str, delta: str = "", delta_class: str = "kpi-neutral"):
    delta_html = f'<div class="kpi-delta {delta_class}">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def badge(text: str, color: str = "blue") -> str:
    return f'<span class="badge badge-{color}">{text}</span>'


PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,32,53,0.6)",
    font=dict(color="#c8d0e8", size=12),
    xaxis=dict(gridcolor="#2a3550", zeroline=False),
    yaxis=dict(gridcolor="#2a3550", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2a3550"),
    margin=dict(l=40, r=20, t=50, b=40),
)


# ── Loaders ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_features():
    f = DATA_DIR / "features.csv"
    return pd.read_csv(f, parse_dates=["hour"]) if f.exists() else pd.DataFrame()


@st.cache_data(ttl=3600)
def load_revenue_summary():
    f = DATA_DIR / "revenue_summary.csv"
    return pd.read_csv(f, parse_dates=["date"]) if f.exists() else pd.DataFrame()


@st.cache_data(ttl=3600)
def load_revenue_detail():
    f = DATA_DIR / "revenue_by_region_hour.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


@st.cache_data(ttl=3600)
def load_pricing():
    f = DATA_DIR / "pricing_recommendations.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


@st.cache_data(ttl=3600)
def load_roi():
    f = DATA_DIR / "infrastructure_roi.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


@st.cache_resource
def load_model():
    model_file = MODELS_DIR / "parking_demand_lgbm.pkl"
    feat_file = MODELS_DIR / "feature_columns.json"
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
    return pd.read_csv(f).iloc[0].to_dict() if f.exists() else {}


# ── Load data ─────────────────────────────────────────────────────────────────

features = load_features()
pricing = load_pricing()
rev_summary = load_revenue_summary()
rev_detail = load_revenue_detail()
roi_df = load_roi()
model, feat_cols = load_model()
perf = load_perf()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🏙️ City of Seattle")
    st.markdown("**Parking Revenue Intelligence**")
    st.caption("SMC 11.16.121 Optimization Platform")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📊 Overview", "💰 Revenue", "📈 Forecast", "🎯 Pricing", "🏗️ Infrastructure", "🗺️ Geo Map"],
        label_visibility="collapsed",
    )

    st.divider()
    if not features.empty:
        last_updated = features["hour"].max().strftime("%b %d %Y, %H:%M")
        n_days = features["hour"].dt.date.nunique()
        avg_occ = features["avg_occupancy_rate"].mean()
        st.markdown(f"**Data through:** {last_updated}")
        st.markdown(f"**Coverage:** {n_days} days")
        occ_color = "🟢" if 0.70 <= avg_occ <= 0.85 else ("🔴" if avg_occ > 0.85 else "🔵")
        st.markdown(f"**System occupancy:** {occ_color} {avg_occ:.1%}")
    if perf:
        st.markdown(f"**Model R²:** {perf.get('r2', 0):.3f}")
    st.divider()
    st.caption("Rate bounds: $0.50–$8.00/hr")
    st.caption("Target: 70–85% occupancy")


# ── Page: Overview ────────────────────────────────────────────────────────────

if page == "📊 Overview":
    st.title("Seattle Parking — Live Overview")

    if features.empty:
        st.error("No feature data available. Run the pipeline first.")
        st.stop()

    recent = features[features["hour"] >= features["hour"].max() - pd.Timedelta(hours=48)]
    avg_occ = features["avg_occupancy_rate"].mean()
    recent_occ = recent["avg_occupancy_rate"].mean() if not recent.empty else 0
    n_regions = features["region"].nunique()
    n_days = features["hour"].dt.date.nunique()

    occ_cls = "kpi-good" if 0.70 <= avg_occ <= 0.85 else ("kpi-bad" if avg_occ > 0.85 else "kpi-warn")
    r_cls = "kpi-good" if 0.70 <= recent_occ <= 0.85 else ("kpi-bad" if recent_occ > 0.85 else "kpi-warn")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi("System Occupancy", f"{avg_occ:.1%}",
            "▲ Above target" if avg_occ > 0.85 else ("In target band" if avg_occ >= 0.70 else "▼ Below target"),
            occ_cls)
    with col2:
        kpi("Last 48h Occupancy", f"{recent_occ:.1%}", "", r_cls)
    with col3:
        kpi("Regions Tracked", str(n_regions), "Active meter zones", "kpi-neutral")
    with col4:
        kpi("Days of Data", f"{n_days:,}", "In feature store", "kpi-neutral")

    st.markdown("<br>", unsafe_allow_html=True)

    # Region status cards
    section("REGION STATUS")
    region_occ = (
        features.groupby("region")["avg_occupancy_rate"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    cols = st.columns(len(region_occ))
    for i, (_, row) in enumerate(region_occ.iterrows()):
        rate = row["avg_occupancy_rate"]
        status = ("HIGH DEMAND", "red") if rate > 0.85 else (("ON TARGET", "green") if rate >= 0.70 else ("UNDERUTILIZED", "blue"))
        action = "Raise rates →" if rate > 0.85 else ("Hold →" if rate >= 0.70 else "Analyze demand →")
        with cols[i]:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{row['region']}</div>
                <div class="kpi-value">{rate:.0%}</div>
                <div style="margin-top:8px">{badge(*status)}</div>
                <div style="font-size:11px;color:#7b8db0;margin-top:6px">{action}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("OCCUPANCY PATTERNS")

    col_left, col_right = st.columns(2)

    with col_left:
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
            labels={"x": "Day", "y": "Hour", "color": "Occupancy"},
            color_continuous_scale="RdYlGn_r",
            zmin=0, zmax=1,
            title="Occupancy Heatmap: Hour × Day",
            height=380,
        )
        fig.update_layout(**PLOTLY_THEME)
        fig.update_coloraxes(colorbar=dict(tickformat=".0%", len=0.8))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # 7-day trend
        daily_avg = (
            features.groupby(features["hour"].dt.date)["avg_occupancy_rate"]
            .mean()
            .reset_index()
            .rename(columns={"hour": "date"})
            .tail(30)
        )
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=daily_avg["date"], y=daily_avg["avg_occupancy_rate"],
            fill="tozeroy", fillcolor="rgba(74,144,217,0.15)",
            line=dict(color="#4a90d9", width=2),
            name="Occupancy",
        ))
        fig2.add_hline(y=0.85, line_dash="dash", line_color="#e05c5c",
                       annotation_text="85% upper", annotation_position="top right")
        fig2.add_hline(y=0.70, line_dash="dash", line_color="#f5a623",
                       annotation_text="70% lower", annotation_position="bottom right")
        fig2.update_layout(
            title="Daily Average Occupancy (last 30 days)",
            height=380,
            **{**PLOTLY_THEME, "yaxis": dict(tickformat=".0%", range=[0, 1])},
        )
        st.plotly_chart(fig2, use_container_width=True)

    section("PEAK HOURS BY REGION")
    hourly_region = (
        features.groupby(["region", "hour_of_day"])["avg_occupancy_rate"]
        .mean()
        .reset_index()
    )
    fig3 = px.line(
        hourly_region, x="hour_of_day", y="avg_occupancy_rate", color="region",
        title="Average Occupancy by Hour of Day",
        labels={"hour_of_day": "Hour", "avg_occupancy_rate": "Occupancy", "region": "Region"},
        height=350,
    )
    fig3.add_hrect(y0=0.70, y1=0.85, fillcolor="rgba(76,175,143,0.08)",
                   line_width=0, annotation_text="Target band")
    fig3.update_yaxes(tickformat=".0%", range=[0, 1])
    fig3.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig3, use_container_width=True)


# ── Page: Revenue ─────────────────────────────────────────────────────────────

elif page == "💰 Revenue":
    st.title("Revenue Intelligence")
    st.caption("Current earnings vs full potential at 80% occupancy target with optimal pricing")

    if rev_summary.empty:
        st.error("Revenue data not generated. Run `python scripts/revenue_analyzer.py`.")
        st.stop()

    total_current = rev_summary["current_revenue"].sum()
    total_target = rev_summary.get("target_revenue", rev_summary["optimized_revenue"]).sum() \
        if "target_revenue" in rev_summary.columns else None
    total_optimized = rev_summary["optimized_revenue"].sum()
    total_uplift = total_optimized - total_current
    uplift_pct = total_uplift / max(total_current, 0.01) * 100

    # Annualise from the period we have
    n_days = rev_summary["date"].nunique()
    annual_current = total_current / max(n_days, 1) * 365
    annual_optimized = total_optimized / max(n_days, 1) * 365
    annual_uplift = annual_optimized - annual_current

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi("Revenue (Current Rates)", f"${total_current:,.0f}",
            f"Over {n_days} days", "kpi-neutral")
    with col2:
        kpi("Revenue (Optimized Rates)", f"${total_optimized:,.0f}",
            f"At 80% occupancy", "kpi-good")
    with col3:
        kpi("Potential Uplift", f"${total_uplift:+,.0f}",
            f"+{uplift_pct:.1f}% gain available", "kpi-good")
    with col4:
        kpi("Annual Projection", f"${annual_uplift:+,.0f}",
            "Incremental per year", "kpi-good")

    st.markdown(f"""
    <div style="background:#153d2a;border:1px solid #205c3e;border-radius:10px;
                padding:14px 20px;margin:16px 0;color:#c8d0e8;font-size:13px;">
        💡 <b>How to read this:</b> "Current" = what meters actually earned at today's rates and occupancy.
        "Optimized" = projected earnings if occupancy reaches the city's 80% target with SMC 11.16.121
        rate adjustments. The uplift is always positive — the city is leaving money on the table
        relative to target utilization.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("REVENUE TREND")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rev_summary["date"], y=rev_summary["current_revenue"],
        name="Current Revenue", marker_color="#4a90d9",
        opacity=0.8,
    ))
    if "target_revenue" in rev_summary.columns:
        fig.add_trace(go.Scatter(
            x=rev_summary["date"], y=rev_summary["target_revenue"],
            name="At 80% Occupancy (current rates)",
            line=dict(color="#f5a623", width=2, dash="dot"),
        ))
    fig.add_trace(go.Scatter(
        x=rev_summary["date"], y=rev_summary["optimized_revenue"],
        name="Optimized (80% occ + optimal rates)",
        line=dict(color="#4caf8f", width=2.5),
    ))
    fig.update_layout(
        title="Daily Revenue: Current vs Potential",
        yaxis_title="Revenue ($)",
        barmode="overlay",
        height=380,
        **PLOTLY_THEME,
    )
    st.plotly_chart(fig, use_container_width=True)

    if not rev_detail.empty:
        col_left, col_right = st.columns(2)

        with col_left:
            section("UPLIFT BY REGION")
            region_uplift = (
                rev_detail.groupby("region")
                .agg(
                    current=("current_revenue_per_hour", "sum"),
                    optimized=("optimized_revenue_per_hour", "sum"),
                )
                .assign(uplift=lambda x: x["optimized"] - x["current"])
                .sort_values("uplift", ascending=True)
                .reset_index()
            )
            fig2 = go.Figure(go.Bar(
                x=region_uplift["uplift"],
                y=region_uplift["region"],
                orientation="h",
                marker_color="#4caf8f",
                text=region_uplift["uplift"].map("${:,.0f}".format),
                textposition="outside",
            ))
            fig2.update_layout(
                title="Revenue Uplift by Region ($/period)",
                xaxis_title="Uplift ($)",
                height=320,
                **PLOTLY_THEME,
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col_right:
            section("UPLIFT HEATMAP: HOUR × REGION")
            pivot = rev_detail.pivot_table(
                index="hour_of_day", columns="region",
                values="revenue_uplift", aggfunc="mean",
            )
            fig3 = px.imshow(
                pivot,
                labels={"x": "Region", "y": "Hour", "color": "Uplift ($/hr)"},
                color_continuous_scale="Greens",
                title="Revenue Uplift per Hour by Region",
                height=320,
            )
            fig3.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig3, use_container_width=True)


# ── Page: Forecast ────────────────────────────────────────────────────────────

elif page == "📈 Forecast":
    st.title("Demand Forecast")
    st.caption("7-day occupancy predictions by region — LightGBM model")

    if model is None:
        st.error("No trained model found. Run `python scripts/train_model.py` first.")
        st.stop()
    if features.empty:
        st.error("No feature data. Run the pipeline first.")
        st.stop()

    perf_r2 = perf.get("r2", 0)
    perf_rmse = perf.get("rmse", 0)
    col1, col2, col3 = st.columns(3)
    with col1:
        kpi("Model R²", f"{perf_r2:.3f}", "Variance explained", "kpi-good" if perf_r2 > 0.85 else "kpi-warn")
    with col2:
        kpi("RMSE", f"{perf_rmse:.3f}", "Occupancy rate error", "kpi-neutral")
    with col3:
        kpi("Training Samples", f"{int(perf.get('n_train', 0)):,}", "Rolling 12-month window", "kpi-neutral")

    st.markdown("<br>", unsafe_allow_html=True)

    last_hour = features["hour"].max()
    future_hours = pd.date_range(
        start=last_hour + pd.Timedelta(hours=1), periods=7 * 24, freq="h"
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

    space_avgs = features.groupby("region")["total_spaces"].mean()
    future_df["total_spaces"] = future_df["region"].map(space_avgs).fillna(100)
    future_df["num_blockfaces"] = (
        features.groupby("region")["num_blockfaces"].mean()
        .reindex(future_df["region"]).values
    )

    for col in feat_cols:
        if col not in future_df.columns:
            future_df[col] = 0
    for col in feat_cols:
        if future_df[col].dtype == bool:
            future_df[col] = future_df[col].astype(int)

    future_df["predicted_occupancy"] = model.predict(
        future_df[[c for c in feat_cols if c in future_df.columns]].fillna(0)
    ).clip(0, 1)

    selected_region = st.selectbox("Select Region", sorted(regions))
    region_forecast = future_df[future_df["region"] == selected_region].copy()

    # Determine hours over/under target
    over = region_forecast[region_forecast["predicted_occupancy"] > 0.85]
    under = region_forecast[region_forecast["predicted_occupancy"] < 0.70]

    col1, col2, col3 = st.columns(3)
    with col1:
        kpi("Avg Predicted", f"{region_forecast['predicted_occupancy'].mean():.1%}", "Next 7 days", "kpi-neutral")
    with col2:
        kpi("Hours Over 85%", str(len(over)), "Rate increase window", "kpi-bad" if len(over) > 0 else "kpi-neutral")
    with col3:
        kpi("Hours Under 70%", str(len(under)), "Demand stimulus window", "kpi-warn" if len(under) > 0 else "kpi-neutral")

    st.markdown("<br>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=region_forecast["hour"], y=region_forecast["predicted_occupancy"],
        fill="tozeroy", fillcolor="rgba(74,144,217,0.1)",
        line=dict(color="#4a90d9", width=2),
        name="Predicted Occupancy",
    ))
    if not over.empty:
        fig.add_trace(go.Scatter(
            x=over["hour"], y=over["predicted_occupancy"],
            mode="markers", marker=dict(color="#e05c5c", size=6),
            name=">85% (raise rates)",
        ))
    fig.add_hrect(y0=0.70, y1=0.85, fillcolor="rgba(76,175,143,0.08)", line_width=0,
                  annotation_text="Target band", annotation_position="top left")
    fig.add_hline(y=0.85, line_dash="dash", line_color="#e05c5c", line_width=1)
    fig.add_hline(y=0.70, line_dash="dash", line_color="#f5a623", line_width=1)
    fig.update_layout(
        title=f"7-Day Occupancy Forecast — {selected_region}",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        height=400,
        **PLOTLY_THEME,
    )
    st.plotly_chart(fig, use_container_width=True)

    if not over.empty:
        section("HIGH-DEMAND HOURS (>85%) — ACTION REQUIRED")
        disp = (
            over[["hour", "predicted_occupancy"]]
            .copy()
            .assign(
                Date=lambda x: x["hour"].dt.strftime("%a %b %d"),
                Hour=lambda x: x["hour"].dt.strftime("%H:00"),
                Occupancy=lambda x: x["predicted_occupancy"].map("{:.1%}".format),
                Action=lambda x: "Raise rate by $0.25–$0.50",
            )
            [["Date", "Hour", "Occupancy", "Action"]]
        )
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ── Page: Pricing ─────────────────────────────────────────────────────────────

elif page == "🎯 Pricing":
    st.title("Dynamic Pricing Recommendations")
    st.caption("Seattle Municipal Code 11.16.121 — Performance-Based Parking Pricing")

    st.markdown("""
    <div style="background:#1a2035;border:1px solid #2a3550;border-radius:10px;
                padding:14px 20px;margin-bottom:20px;font-size:13px;color:#c8d0e8;">
        <b>⚖️ Legal authority:</b> SMC 11.16.121 authorizes meter rate adjustments to achieve
        70–85% occupancy. Rate bounds: <b>$0.50–$8.00/hour</b>.
        Rate changes require City Council approval.
    </div>
    """, unsafe_allow_html=True)

    if pricing.empty:
        st.error("No pricing recommendations. Run `python scripts/pricing_optimizer.py`.")
        st.stop()

    increases = (pricing["action"] == "increase").sum()
    decreases = (pricing["action"] == "decrease").sum()
    holds = (pricing["action"] == "hold").sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi("Rate Increases", str(increases), "Zones over 85%", "kpi-bad" if increases > 0 else "kpi-neutral")
    with col2:
        kpi("Rate Decreases", str(decreases), "Zones under 70%", "kpi-blue" if decreases > 0 else "kpi-neutral")
    with col3:
        kpi("No Change", str(holds), "In target band", "kpi-good")
    with col4:
        rev_impact = pricing["revenue_delta"].sum() if "revenue_delta" in pricing.columns else 0
        kpi("Total Rev. Impact", f"${rev_impact:+,.0f}/hr",
            "If all changes applied", "kpi-good" if rev_impact >= 0 else "kpi-warn")

    st.markdown("<br>", unsafe_allow_html=True)

    # Rate change waterfall by region
    if "revenue_delta" in pricing.columns:
        section("REVENUE IMPACT BY REGION")
        region_impact = (
            pricing.groupby("region")["revenue_delta"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        colors = ["#4caf8f" if v >= 0 else "#e05c5c" for v in region_impact["revenue_delta"]]
        fig = go.Figure(go.Bar(
            x=region_impact["region"],
            y=region_impact["revenue_delta"],
            marker_color=colors,
            text=region_impact["revenue_delta"].map("${:+,.0f}".format),
            textposition="outside",
        ))
        fig.add_hline(y=0, line_color="#7b8db0", line_width=1)
        fig.update_layout(
            title="Estimated Revenue Delta by Region ($/hr if all changes applied)",
            yaxis_title="Revenue Impact ($/hr)",
            height=320,
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig, use_container_width=True)

    section("RECOMMENDATIONS TABLE")
    region_filter = st.selectbox("Filter by Region", ["All"] + sorted(pricing["region"].unique()))
    action_filter = st.selectbox("Filter by Action", ["All", "increase", "decrease", "hold"])

    disp = pricing.copy()
    if region_filter != "All":
        disp = disp[disp["region"] == region_filter]
    if action_filter != "All":
        disp = disp[disp["action"] == action_filter]

    def color_action(val):
        if val == "increase":
            return "background-color:#3d1515;color:#e05c5c"
        if val == "decrease":
            return "background-color:#152a3d;color:#4a90d9"
        return "color:#7b8db0"

    styled = disp[["region", "hour_of_day", "avg_predicted_occupancy",
                    "current_rate", "recommended_rate", "rate_change",
                    "action", "revenue_delta"]].copy()
    styled["avg_predicted_occupancy"] = styled["avg_predicted_occupancy"].map("{:.1%}".format)
    styled["current_rate"] = styled["current_rate"].map("${:.2f}".format)
    styled["recommended_rate"] = styled["recommended_rate"].map("${:.2f}".format)
    styled["rate_change"] = styled["rate_change"].map("${:+.2f}".format)
    styled["revenue_delta"] = styled["revenue_delta"].map("${:+.2f}".format)
    styled.columns = ["Region", "Hour", "Pred. Occ.", "Current Rate",
                      "Rec. Rate", "Change", "Action", "Rev. Impact/hr"]
    st.dataframe(
        styled.style.map(color_action, subset=["Action"]),
        use_container_width=True, hide_index=True,
    )


# ── Page: Infrastructure ROI ──────────────────────────────────────────────────

elif page == "🏗️ Infrastructure":
    st.title("Infrastructure Investment ROI")
    st.caption("Cost-benefit analysis: should Seattle build more parking capacity?")

    if roi_df.empty:
        st.error("No ROI data. Run `python scripts/infrastructure_roi.py`.")
        st.stop()

    viable = roi_df[roi_df["viable"]]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi("Scenarios Analyzed", str(len(roi_df)), "", "kpi-neutral")
    with col2:
        kpi("Viable Investments", str(len(viable)),
            "Breakeven < current demand", "kpi-good" if len(viable) > 0 else "kpi-warn")
    with col3:
        best_roi = viable["roi_percent"].max() if not viable.empty else 0
        kpi("Best ROI", f"{best_roi:.1f}%", "", "kpi-good" if best_roi > 0 else "kpi-neutral")
    with col4:
        best_payback = viable["payback_years"].min() if not viable.empty else 0
        kpi("Fastest Payback", f"{best_payback:.1f} yrs" if best_payback > 0 else "N/A", "", "kpi-neutral")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    region_filter = col1.selectbox("Region", ["All"] + sorted(roi_df["region"].unique()))
    infra_filter = col2.selectbox("Infrastructure Type", ["All"] + sorted(roi_df["infra_type"].unique()))

    disp = roi_df.copy()
    if region_filter != "All":
        disp = disp[disp["region"] == region_filter]
    if infra_filter != "All":
        disp = disp[disp["infra_type"] == infra_filter]

    col_left, col_right = st.columns(2)

    with col_left:
        section("ROI BY SCENARIO")
        fig = px.bar(
            disp.sort_values("roi_percent", ascending=False),
            x="region", y="roi_percent", color="infra_type",
            barmode="group", facet_col="n_new_spaces",
            labels={"roi_percent": "ROI %", "region": "Region", "infra_type": "Type"},
            height=380,
            color_discrete_map={"surface_lot": "#4a90d9", "structured_garage": "#f5a623",
                                 "underground": "#e05c5c"},
        )
        fig.add_hline(y=0, line_dash="solid", line_color="#e05c5c", line_width=1)
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        section("BREAKEVEN vs CURRENT OCCUPANCY")
        fig2 = px.scatter(
            disp,
            x="current_occupancy", y="breakeven_occupancy",
            color="viable", size="n_new_spaces",
            hover_name="region",
            hover_data={"roi_percent": ":.1f", "payback_years": ":.1f"},
            color_discrete_map={True: "#4caf8f", False: "#e05c5c"},
            labels={"current_occupancy": "Current Occupancy",
                    "breakeven_occupancy": "Breakeven Occupancy"},
            height=380,
        )
        fig2.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                       line=dict(dash="dash", color="#7b8db0"))
        fig2.update_xaxes(tickformat=".0%")
        fig2.update_yaxes(tickformat=".0%")
        fig2.update_layout(**PLOTLY_THEME)
        st.caption("Points below diagonal line are viable (current demand > breakeven).")
        st.plotly_chart(fig2, use_container_width=True)

    section("SCENARIO DETAILS")
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
    table.columns = ["Region", "Type", "Spaces", "Curr. Occ.", "Breakeven",
                     "Construction Cost", "Net Annual", "ROI %", "Payback (yrs)", "Viable"]
    st.dataframe(
        table.style.map(
            lambda v: "background-color:#153d2a" if v is True else
                      ("background-color:#3d1515" if v is False else ""),
            subset=["Viable"]
        ),
        use_container_width=True, hide_index=True,
    )
    st.caption("Costs: surface lot $5K/space, structured garage $45K/space, underground $90K/space. Bond: 4.5%, 20yr.")


# ── Page: Geo Map ─────────────────────────────────────────────────────────────

elif page == "🗺️ Geo Map":
    import folium
    import json as _json
    from streamlit_folium import st_folium

    st.title("Parking Demand Map")
    st.caption("Neighborhood polygons from Seattle Open Data · Circle size = avg parking spaces · Click for full stats")

    if features.empty:
        st.error("No data to display.")
        st.stop()

    # Real coordinates from OpenStreetMap Nominatim
    REGION_COORDS = {
        "Downtown Seattle":      (47.6081, -122.3321),
        "Capitol Hill":          (47.6238, -122.3184),
        "South Lake Union":      (47.6232, -122.3384),
        "Ballard":               (47.6765, -122.3862),
        "International District":(47.5984, -122.3225),
    }

    # Build per-region stats
    reg_stats = features.groupby("region").agg(
        avg_occ=("avg_occupancy_rate", "mean"),
        peak_occ=("peak_occupancy_rate", "mean"),
        avg_spaces=("total_spaces", "mean"),
        num_blockfaces=("num_blockfaces", "mean"),
    ).reset_index()

    # Peak demand hour per region
    peak_hour = (
        features.groupby(["region", "hour_of_day"])["avg_occupancy_rate"]
        .mean()
        .reset_index()
    )
    peak_hour = peak_hour.loc[peak_hour.groupby("region")["avg_occupancy_rate"].idxmax()]
    peak_hour = peak_hour.rename(columns={"hour_of_day": "peak_hour", "avg_occupancy_rate": "_ph_occ"})
    reg_stats = reg_stats.merge(peak_hour[["region", "peak_hour"]], on="region", how="left")

    # Revenue per region
    if not rev_detail.empty:
        rev_by_region = rev_detail.groupby("region").agg(
            current_rate=("current_rate", "mean"),
            recommended_rate=("recommended_rate", "mean"),
            daily_current=("current_revenue_per_hour", "sum"),
            daily_optimized=("optimized_revenue_per_hour", "sum"),
        ).reset_index()
        reg_stats = reg_stats.merge(rev_by_region, on="region", how="left")
    else:
        reg_stats["current_rate"] = 2.00
        reg_stats["recommended_rate"] = 2.00
        reg_stats["daily_current"] = 0.0
        reg_stats["daily_optimized"] = 0.0

    # Pricing action
    if not pricing.empty and "action" in pricing.columns:
        top_action = (
            pricing.groupby("region")["action"]
            .agg(lambda x: x.value_counts().index[0])
            .reset_index()
            .rename(columns={"action": "top_action"})
        )
        reg_stats = reg_stats.merge(top_action, on="region", how="left")
    if "top_action" not in reg_stats.columns:
        reg_stats["top_action"] = "hold"
    else:
        reg_stats["top_action"] = reg_stats["top_action"].fillna("hold")

    def occ_color(rate):
        if rate > 0.85:
            return "#e05c5c"   # red — over target
        elif rate >= 0.70:
            return "#4caf8f"   # green — on target
        else:
            return "#4a90d9"   # blue — underutilized

    def occ_status(rate):
        if rate > 0.85:
            return "🔴 High Demand (&gt;85%)"
        elif rate >= 0.70:
            return "🟢 On Target (70–85%)"
        else:
            return "🔵 Underutilized (&lt;70%)"

    action_labels = {"increase": "⬆ Raise rates", "decrease": "⬇ Lower rates", "hold": "✓ Hold rates"}

    # Distinct color per district for polygon fills
    DISTRICT_COLORS = {
        "Downtown Seattle":      "#2563EB",
        "South Lake Union":      "#059669",
        "Capitol Hill":          "#DC2626",
        "International District":"#7C3AED",
        "Ballard":               "#EA580C",
    }

    # Circle radius scaled to avg_spaces
    min_sp = reg_stats["avg_spaces"].min()
    max_sp = reg_stats["avg_spaces"].max()
    def space_radius(spaces):
        if max_sp == min_sp:
            return 35
        return 18 + 32 * (spaces - min_sp) / (max_sp - min_sp)

    # Load neighborhood polygon GeoJSON
    _geojson_path = ROOT / "data" / "seattle_5_neighborhoods.geojson"
    _neighborhoods_geojson = None
    if _geojson_path.exists():
        with open(_geojson_path) as _f:
            _neighborhoods_geojson = _json.load(_f)

    _occ_by_district = dict(zip(reg_stats["region"], reg_stats["avg_occ"]))

    m = folium.Map(
        location=[47.630, -122.330],
        zoom_start=12,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )

    # Neighborhood polygon overlays — each district gets its own color
    if _neighborhoods_geojson:
        for feat in _neighborhoods_geojson["features"]:
            district = feat["properties"]["neighborhood"]
            rate = _occ_by_district.get(district, 0.5)
            color = DISTRICT_COLORS.get(district, "#888888")
            folium.GeoJson(
                feat,
                style_function=lambda x, c=color: {
                    "fillColor":   c,
                    "color":       c,
                    "weight":      2.5,
                    "fillOpacity": 0.20,
                },
                tooltip=folium.Tooltip(
                    f"<b>{district}</b><br>Avg occupancy: {rate:.0%}",
                    sticky=False,
                ),
            ).add_to(m)

    for _, row in reg_stats.iterrows():
        coords = REGION_COORDS.get(row["region"])
        if not coords:
            continue

        # District color for circle (matches polygon) — occupancy color for status badge only
        color = DISTRICT_COLORS.get(row["region"], occ_color(row["avg_occ"]))
        occ_col = occ_color(row["avg_occ"])
        radius = space_radius(row["avg_spaces"])
        action = row["top_action"]
        peak_h = int(row["peak_hour"]) if pd.notna(row.get("peak_hour")) else 0
        peak_label = f"{peak_h % 12 or 12}{'am' if peak_h < 12 else 'pm'}"

        popup_html = f"""
        <div style="font-family:Arial,sans-serif;min-width:220px;background:#12172b;color:#c8d0e8;
                    border-radius:8px;padding:14px;border-left:4px solid {color}">
            <div style="font-size:14px;font-weight:700;color:{color};margin-bottom:10px;
                        border-bottom:1px solid #2a3550;padding-bottom:6px">
                {row['region']}
            </div>
            <table style="width:100%;border-collapse:collapse;font-size:12px">
                <tr>
                    <td style="color:#7b8db0;padding:3px 0">Avg Occupancy</td>
                    <td style="text-align:right;font-weight:600;color:{occ_col}">{row['avg_occ']:.0%}</td>
                </tr>
                <tr>
                    <td style="color:#7b8db0;padding:3px 0">Peak Occupancy</td>
                    <td style="text-align:right;font-weight:600;color:#c8d0e8">{row['peak_occ']:.0%}</td>
                </tr>
                <tr>
                    <td style="color:#7b8db0;padding:3px 0">Peak Hour</td>
                    <td style="text-align:right;font-weight:600;color:#c8d0e8">{peak_label}</td>
                </tr>
                <tr>
                    <td style="color:#7b8db0;padding:3px 0">Avg Spaces</td>
                    <td style="text-align:right;font-weight:600;color:#c8d0e8">{int(row['avg_spaces']):,}</td>
                </tr>
                <tr>
                    <td style="color:#7b8db0;padding:3px 0">Blockfaces</td>
                    <td style="text-align:right;font-weight:600;color:#c8d0e8">{int(row['num_blockfaces'])}</td>
                </tr>
                <tr style="border-top:1px solid #2a3550">
                    <td style="color:#7b8db0;padding:5px 0 3px">Current Rate</td>
                    <td style="text-align:right;font-weight:600;color:#c8d0e8">${row.get('current_rate', 0):.2f}/hr</td>
                </tr>
                <tr>
                    <td style="color:#7b8db0;padding:3px 0">Recommended</td>
                    <td style="text-align:right;font-weight:600;color:{occ_col}">${row.get('recommended_rate', 0):.2f}/hr</td>
                </tr>
                <tr>
                    <td style="color:#7b8db0;padding:3px 0">Pricing Action</td>
                    <td style="text-align:right;font-weight:600;color:{occ_col}">{action_labels.get(action, action)}</td>
                </tr>
            </table>
            <div style="margin-top:8px;padding:5px 8px;background:{occ_col}22;border-radius:4px;
                        font-size:11px;color:{occ_col};font-weight:600;text-align:center">
                {occ_status(row['avg_occ'])}
            </div>
        </div>
        """

        tooltip_html = (
            f"<b style='color:{color}'>{row['region']}</b><br>"
            f"Occupancy: <b style='color:{occ_col}'>{row['avg_occ']:.0%}</b><br>"
            f"Spaces: {int(row['avg_spaces']):,} · Peak: {peak_label}"
        )

        # Outer glow ring
        folium.CircleMarker(
            location=coords,
            radius=radius + 12,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.08,
            weight=1,
            opacity=0.3,
        ).add_to(m)

        # Main circle with popup + tooltip
        folium.CircleMarker(
            location=coords,
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.55,
            weight=2,
            opacity=0.9,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=folium.Tooltip(tooltip_html, sticky=False),
        ).add_to(m)

        # Label in center of circle
        folium.Marker(
            location=coords,
            icon=folium.DivIcon(
                html=f"""<div style="font-family:Arial,sans-serif;font-size:11px;font-weight:700;
                                     color:white;text-align:center;width:80px;margin-left:-40px;
                                     text-shadow:0 1px 3px rgba(0,0,0,0.9)">
                            {row['avg_occ']:.0%}
                         </div>""",
                icon_size=(80, 20),
                icon_anchor=(40, 10),
            ),
        ).add_to(m)

    st_folium(m, use_container_width=True, height=540, returned_objects=[])

    st.markdown("<br>", unsafe_allow_html=True)
    section("REGION BREAKDOWN")
    cols = st.columns(len(reg_stats))
    for i, (_, row) in enumerate(reg_stats.sort_values("avg_occ", ascending=False).iterrows()):
        rate = row["avg_occ"]
        dcolor = DISTRICT_COLORS.get(row["region"], "#888888")
        occ_col = occ_color(rate)
        status = "HIGH DEMAND" if rate > 0.85 else ("ON TARGET" if rate >= 0.70 else "UNDERUTILIZED")
        action = row["top_action"]
        action_icon = {"increase": "⬆", "decrease": "⬇", "hold": "→"}.get(action, "→")
        peak_h = int(row["peak_hour"]) if pd.notna(row.get("peak_hour")) else 0
        peak_label = f"{peak_h % 12 or 12}{'am' if peak_h < 12 else 'pm'}"
        with cols[i]:
            st.markdown(f"""
            <div style="background:#1a2035;border:1px solid {dcolor}50;border-top:3px solid {dcolor};
                        border-radius:8px;padding:14px 16px;text-align:center">
                <div style="font-size:10px;color:{dcolor};font-weight:600;letter-spacing:.05em;margin-bottom:4px">{row['region'].upper()}</div>
                <div style="font-size:28px;font-weight:700;color:{occ_col};margin:4px 0">{rate:.0%}</div>
                <div style="font-size:10px;color:{occ_col};font-weight:600">{status}</div>
                <div style="font-size:11px;color:#7b8db0;margin-top:6px">{int(row['avg_spaces']):,} spaces</div>
                <div style="font-size:11px;color:#7b8db0">Peak: {peak_label}</div>
                <div style="font-size:11px;color:{occ_col};margin-top:4px">{action_icon} {action}</div>
            </div>
            """, unsafe_allow_html=True)
