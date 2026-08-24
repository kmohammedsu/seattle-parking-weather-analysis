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
    initial_sidebar_state="collapsed",
)

# ── Design tokens ────────────────────────────────────────────────────────────
# Civic decision-support system: Source Sans 3 for institutional UI text,
# Lexend for headlines and data numerals — a pairing purpose-built for
# enterprise/government/accessibility-focused products. A single restrained
# accent, and status conveyed by shape + text — never color alone. All
# text/background pairs verified >= 4.5:1 (WCAG AA).

INK          = "#0F172A"  # headings / primary text
BODY         = "#334155"  # body text
MUTED        = "#475569"  # secondary / caption text — 7.0:1 on PAPER, 7.6:1 on CARD
PAPER        = "#F7F8FA"
CARD         = "#FFFFFF"
LINE         = "#E4E7EC"
ACCENT       = "#0A5A8C"  # single considered accent — links, hero data
GOOD         = "#116149"
WARN         = "#8A4B0C"
BAD          = "#9C2626"

REGION_COLORS = {
    "Downtown Seattle":       "#0A5A8C",
    "South Lake Union":       "#116149",
    "Capitol Hill":           "#9C2626",
    "International District": "#5B3A8E",
    "Ballard":                "#8A4B0C",
}

NAV_ITEMS = [
    ("Overview", "overview"),
    ("Revenue", "revenue"),
    ("Forecast", "forecast"),
    ("Pricing", "pricing"),
    ("Infrastructure", "infrastructure"),
    ("Geo Map", "map"),
]

if "page" not in st.session_state:
    st.session_state.page = "Overview"


# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:ital,wght@0,400;0,500;0,600;0,700;0,800;1,500&family=Lexend:wght@500;600;700;800&display=swap');

html, body, [class*="css"], [data-testid], [data-testid] * {{
    font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}}

/* Lexend carries headlines and data numerals — Source Sans 3 stays on UI
   text/labels/body. Both are purpose-built for accessibility/enterprise
   government contexts; Lexend adds real presence without losing legibility.
   Two families, one job split, not decoration. */
h1, h2, h3, .stat-value, .region-value, .brand-title,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {{
    font-family: 'Lexend', 'Source Sans 3', sans-serif !important;
}}

@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{ animation-duration: 0.001ms !important; animation-iteration-count: 1 !important; transition-duration: 0.001ms !important; }}
}}

[data-testid="stAppViewContainer"] {{ background: {PAPER}; }}
[data-testid="stHeader"] {{ background: {PAPER}; }}
[data-testid="stSidebar"] {{ display: none; }}
[data-testid="collapsedControl"] {{ display: none; }}

.block-container {{
    padding-top: 1.25rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1440px;
}}

h1, h2, h3 {{ color: {INK} !important; letter-spacing: -0.01em; }}
p, span, label, div {{ color: {BODY}; }}
[data-testid="stCaptionContainer"] {{ color: {MUTED} !important; }}

.kpi-value, .region-value, td, th {{ font-variant-numeric: tabular-nums; }}

/* ── Motion ───────────────────────────────────────────────────────────── */
@keyframes riseIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes growUp {{
    from {{ transform: scaleY(0); }}
    to   {{ transform: scaleY(1); }}
}}
[data-testid="stHorizontalBlock"] > div:nth-of-type(1) .stat-cell,
[data-testid="stHorizontalBlock"] > div:nth-of-type(1) .region-cell {{ animation: riseIn 420ms cubic-bezier(0.16,1,0.3,1) both; }}
[data-testid="stHorizontalBlock"] > div:nth-of-type(2) .stat-cell,
[data-testid="stHorizontalBlock"] > div:nth-of-type(2) .region-cell {{ animation: riseIn 420ms cubic-bezier(0.16,1,0.3,1) 60ms both; }}
[data-testid="stHorizontalBlock"] > div:nth-of-type(3) .stat-cell,
[data-testid="stHorizontalBlock"] > div:nth-of-type(3) .region-cell {{ animation: riseIn 420ms cubic-bezier(0.16,1,0.3,1) 120ms both; }}
[data-testid="stHorizontalBlock"] > div:nth-of-type(4) .stat-cell,
[data-testid="stHorizontalBlock"] > div:nth-of-type(4) .region-cell {{ animation: riseIn 420ms cubic-bezier(0.16,1,0.3,1) 180ms both; }}
[data-testid="stHorizontalBlock"] > div:nth-of-type(5) .stat-cell,
[data-testid="stHorizontalBlock"] > div:nth-of-type(5) .region-cell {{ animation: riseIn 420ms cubic-bezier(0.16,1,0.3,1) 240ms both; }}

/* ── Top command bar ─────────────────────────────────────────────────── */
.topbar {{
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; row-gap: 12px;
    padding: 22px 0 16px 0;
    margin-bottom: 4px;
}}
.mark {{
    width: 40px; height: 40px; border-radius: 10px;
    background: {INK};
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-weight: 800; font-size: 17px; font-family: 'Source Sans 3', sans-serif;
    flex-shrink: 0;
}}
.brand-block {{ display: flex; align-items: center; gap: 12px; }}
.brand-eyebrow {{
    font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    color: {MUTED}; margin-bottom: 2px;
}}
.brand-title {{ font-size: 17px; font-weight: 700; color: {INK}; line-height: 1.2; }}
.meta-strip {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
.meta-pill {{
    font-size: 12px; font-weight: 600; color: {BODY}; white-space: nowrap;
    background: {CARD}; border: 1px solid {LINE}; border-radius: 999px;
    padding: 5px 14px;
}}
.meta-pill b {{ color: {INK}; }}

/* ── Nav row ──────────────────────────────────────────────────────────── */
.navrow-rule {{ border-bottom: 1px solid {LINE}; margin-bottom: 28px; }}
div[class*="st-key-navitem_"] {{ margin-right: 4px; }}
div[class*="st-key-navitem_"] button {{
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    color: {MUTED} !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 8px 4px 12px 4px !important;
    margin-right: 22px !important;
    white-space: nowrap !important;
    transition: color 180ms ease, border-color 180ms ease !important;
    box-shadow: none !important;
}}
div[class*="st-key-navitem_"] button:hover {{
    color: {INK} !important;
    border-bottom-color: {LINE} !important;
}}
div[class*="st-key-navitem_"] button:focus-visible {{
    outline: 2px solid {ACCENT} !important;
    outline-offset: 3px !important;
}}

/* ── Stat cells (asymmetric hero grid) ───────────────────────────────── */
.stat-cell {{
    background: {CARD};
    border: 1px solid {LINE};
    border-radius: 14px;
    padding: 20px 22px;
    height: 100%;
    position: relative;
    overflow: hidden;
    transition: transform 200ms cubic-bezier(0.16,1,0.3,1), box-shadow 200ms ease, border-color 200ms ease;
}}
.stat-cell:hover {{
    transform: translateY(-3px);
    border-color: #CBD5E1;
    box-shadow: 0 8px 20px -8px rgba(15,23,42,0.18);
}}
.stat-cell.hero {{
    background: {INK};
    border-color: {INK};
}}
.stat-cell.hero:hover {{
    box-shadow: 0 10px 24px -8px rgba(15,23,42,0.45);
}}
.stat-cell.hero::before {{
    content: "";
    position: absolute; top: -1px; left: -1px; right: -1px; height: 4px;
    background: {ACCENT};
}}
.stat-label {{
    font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
    color: {MUTED}; margin-bottom: 10px;
}}
.stat-cell.hero .stat-label {{ color: #94A3B8; }}
.stat-value {{ font-size: 30px; font-weight: 800; color: {INK}; line-height: 1.05; letter-spacing: -0.01em; }}
.stat-cell.hero .stat-value {{ color: #fff; font-size: 42px; }}
.stat-delta {{ font-size: 13px; margin-top: 8px; font-weight: 600; display: flex; align-items: center; gap: 5px; }}
.stat-cell.hero .stat-delta {{ color: #CBD5E1; }}
.kpi-good {{ color: {GOOD}; }}
.kpi-warn {{ color: {WARN}; }}
.kpi-bad  {{ color: {BAD}; }}
.kpi-neutral {{ color: {MUTED}; }}

.spark-row {{ display: flex; align-items: flex-end; gap: 4px; height: 34px; margin-top: 14px; }}
.spark-bar {{
    flex: 1; background: rgba(255,255,255,0.28); border-radius: 2px 2px 0 0; min-height: 3px;
    transform-origin: bottom; animation: growUp 500ms cubic-bezier(0.16,1,0.3,1) both;
}}
.spark-bar.top {{ background: {ACCENT}; }}
.spark-row .spark-bar:nth-child(1)  {{ animation-delay: 0ms; }}
.spark-row .spark-bar:nth-child(2)  {{ animation-delay: 25ms; }}
.spark-row .spark-bar:nth-child(3)  {{ animation-delay: 50ms; }}
.spark-row .spark-bar:nth-child(4)  {{ animation-delay: 75ms; }}
.spark-row .spark-bar:nth-child(5)  {{ animation-delay: 100ms; }}
.spark-row .spark-bar:nth-child(6)  {{ animation-delay: 125ms; }}
.spark-row .spark-bar:nth-child(7)  {{ animation-delay: 150ms; }}
.spark-row .spark-bar:nth-child(8)  {{ animation-delay: 175ms; }}
.spark-row .spark-bar:nth-child(9)  {{ animation-delay: 200ms; }}
.spark-row .spark-bar:nth-child(10) {{ animation-delay: 225ms; }}
.spark-row .spark-bar:nth-child(11) {{ animation-delay: 250ms; }}
.spark-row .spark-bar:nth-child(12) {{ animation-delay: 275ms; }}
.spark-row .spark-bar:nth-child(13) {{ animation-delay: 300ms; }}
.spark-row .spark-bar:nth-child(14) {{ animation-delay: 325ms; }}

/* ── Section label ────────────────────────────────────────────────────── */
.section-label {{
    display: flex; align-items: center; gap: 8px;
    font-size: 15px; font-weight: 700; color: {INK};
    margin: 32px 0 14px 0;
}}
.section-label .dot {{ width: 7px; height: 7px; border-radius: 2px; background: {ACCENT}; flex-shrink: 0; }}

/* ── Region / breakdown cells ────────────────────────────────────────── */
.region-cell {{
    background: {CARD};
    border: 1px solid {LINE};
    border-top: 3px solid var(--rc, {ACCENT});
    border-radius: 4px 4px 12px 12px;
    padding: 16px 16px 18px 16px;
    transition: transform 200ms cubic-bezier(0.16,1,0.3,1), box-shadow 200ms ease;
}}
.region-cell:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 20px -8px rgba(15,23,42,0.16);
}}
.region-name {{ font-size: 11px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--rc, {ACCENT}); margin-bottom: 6px; }}
.region-value {{ font-size: 26px; font-weight: 800; color: {INK}; margin-bottom: 4px; }}
.region-sub {{ font-size: 12px; color: {MUTED}; }}

/* ── Badge — text + icon, never color alone ──────────────────────────── */
.badge {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 9px; border-radius: 4px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.02em;
    border: 1px solid;
}}
.badge-red    {{ background: #FBEEEE; color: {BAD};  border-color: #F1D4D4; }}
.badge-green  {{ background: #EAF3EF; color: {GOOD}; border-color: #CFE5DA; }}
.badge-blue   {{ background: #EAF1F6; color: {ACCENT}; border-color: #CBDEE9; }}
.badge-yellow {{ background: #F7EEE2; color: {WARN}; border-color: #ECD9BC; }}

/* ── Callout — flush rule, not a boxed card ──────────────────────────── */
.callout {{
    border-left: 3px solid {ACCENT};
    padding: 4px 0 4px 18px;
    margin: 18px 0;
    font-size: 13.5px;
    line-height: 1.6;
    color: {BODY};
}}
.callout.legal {{ border-left-color: {INK}; }}

/* Native Streamlit form controls */
[data-testid="stSelectbox"] label {{ font-weight: 600 !important; color: {INK} !important; font-size: 13px !important; }}
.stDataFrame {{ border: 1px solid {LINE}; border-radius: 10px; overflow: hidden; }}

#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ── Icons (hand-built inline SVG — not a generic icon-pack import) ────────────

def icon(name: str, color: str, size: int = 12) -> str:
    stroke = f'stroke="{color}" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"'
    paths = {
        "up":    f'<path {stroke} d="M6 2v8M2.5 5.5 6 2l3.5 3.5"/>',
        "down":  f'<path {stroke} d="M6 10V2M2.5 6.5 6 10l3.5-3.5"/>',
        "flat":  f'<path {stroke} d="M2 6h8"/>',
        "dot":   f'<circle cx="6" cy="6" r="3.5" fill="{color}"/>',
    }
    return f'<svg width="{size}" height="{size}" viewBox="0 0 12 12" aria-hidden="true">{paths.get(name, "")}</svg>'


# ── Helpers ───────────────────────────────────────────────────────────────────

def stat_cell(label: str, value: str, delta: str = "", delta_class: str = "kpi-neutral", hero: bool = False):
    delta_html = f'<div class="stat-delta {delta_class}">{delta}</div>' if delta else ""
    cls = "stat-cell hero" if hero else "stat-cell"
    st.markdown(f"""
    <div class="{cls}">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def hero_with_sparkline(label: str, value: str, delta_html: str, spark_values):
    lo, hi = min(spark_values), max(spark_values)
    rng = (hi - lo) or 1
    bars = ""
    for i, v in enumerate(spark_values):
        h = max(4, int(((v - lo) / rng) * 34))
        cls = "spark-bar top" if i == len(spark_values) - 1 else "spark-bar"
        bars += f'<div class="{cls}" style="height:{h}px"></div>'
    st.markdown(f"""
    <div class="stat-cell hero">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        <div class="stat-delta">{delta_html}</div>
        <div class="spark-row">{bars}</div>
    </div>
    """, unsafe_allow_html=True)


def stat_row(cells):
    widths = [1.7 if c.get("hero") else 1 for c in cells]
    cols = st.columns(widths)
    for col, c in zip(cols, cells):
        with col:
            stat_cell(c["label"], c["value"], c.get("delta", ""), c.get("delta_class", "kpi-neutral"), hero=c.get("hero", False))


def section(title: str):
    st.markdown(f'<div class="section-label"><span class="dot"></span>{title}</div>', unsafe_allow_html=True)


def badge(text: str, color: str = "blue", ic: str = None) -> str:
    ic_html = icon(ic, {"blue": ACCENT, "green": GOOD, "red": BAD, "yellow": WARN}.get(color, ACCENT)) if ic else ""
    return f'<span class="badge badge-{color}">{ic_html}{text}</span>'


def callout(text_html: str, kind: str = ""):
    cls = f"callout {kind}".strip()
    st.markdown(f'<div class="{cls}">{text_html}</div>', unsafe_allow_html=True)


PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=BODY, size=12, family="Source Sans 3, sans-serif"),
    title_font=dict(color=INK, size=15, family="Lexend, Source Sans 3, sans-serif"),
    xaxis=dict(gridcolor=LINE, zeroline=False, linecolor=LINE),
    yaxis=dict(gridcolor=LINE, zeroline=False, linecolor=LINE),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=LINE),
    margin=dict(l=40, r=20, t=50, b=40),
    colorway=[ACCENT, GOOD, WARN, "#5B3A8E", BAD, "#0E7A8C"],
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


# ── Top command bar ───────────────────────────────────────────────────────────

meta_html = ""
if not features.empty:
    last_updated = features["hour"].max().strftime("%b %d, %Y")
    n_days = features["hour"].dt.date.nunique()
    avg_occ_meta = features["avg_occupancy_rate"].mean()
    meta_html += f'<span class="meta-pill">Data through <b>{last_updated}</b></span>'
    meta_html += f'<span class="meta-pill">{n_days} days tracked</span>'
if perf:
    meta_html += f'<span class="meta-pill">Model R&sup2; <b>{perf.get("r2", 0):.3f}</b></span>'

st.markdown(f"""
<div class="topbar">
    <div class="brand-block">
        <div class="mark">S</div>
        <div>
            <div class="brand-eyebrow">City of Seattle &middot; SMC 11.16.121</div>
            <div class="brand-title">Parking Revenue Intelligence</div>
        </div>
    </div>
    <div class="meta-strip">{meta_html}</div>
</div>
""", unsafe_allow_html=True)

nav_widths = [1.0, 0.9, 1.0, 0.9, 1.5, 1.0, 3.5]
nav_cols = st.columns(nav_widths)
for (label, slug), col in zip(NAV_ITEMS, nav_cols[:-1]):
    with col:
        with st.container(key=f"navitem_{slug}"):
            if st.button(label, key=f"navbtn_{slug}", use_container_width=False):
                st.session_state.page = label

active_slug = dict((l, s) for l, s in NAV_ITEMS)[st.session_state.page]
st.markdown(f"""
<style>
div[class*="st-key-navitem_{active_slug}"] button {{
    color: {INK} !important;
    border-bottom-color: {ACCENT} !important;
}}
</style>
<div class="navrow-rule"></div>
""", unsafe_allow_html=True)

page = st.session_state.page


# ── Page: Overview ────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("Live overview")

    if features.empty:
        st.error("No feature data available. Run the pipeline first.")
        st.stop()

    recent = features[features["hour"] >= features["hour"].max() - pd.Timedelta(hours=48)]
    avg_occ = features["avg_occupancy_rate"].mean()
    recent_occ = recent["avg_occupancy_rate"].mean() if not recent.empty else 0
    n_regions = features["region"].nunique()
    n_days = features["hour"].dt.date.nunique()

    daily_trend = (
        features.groupby(features["hour"].dt.date)["avg_occupancy_rate"]
        .mean()
        .tail(14)
    )

    occ_cls = "kpi-good" if 0.70 <= avg_occ <= 0.85 else ("kpi-bad" if avg_occ > 0.85 else "kpi-warn")
    r_cls = "kpi-good" if 0.70 <= recent_occ <= 0.85 else ("kpi-bad" if recent_occ > 0.85 else "kpi-warn")
    ic = "up" if avg_occ > 0.85 else ("flat" if avg_occ >= 0.70 else "down")
    delta_color = BAD if avg_occ > 0.85 else (GOOD if avg_occ >= 0.70 else WARN)
    delta_text = "Above target" if avg_occ > 0.85 else ("In target band" if avg_occ >= 0.70 else "Below target")

    c1, c2 = st.columns([1.7, 1])
    with c1:
        hero_with_sparkline(
            "System occupancy",
            f"{avg_occ:.1%}",
            f'{icon(ic, "#CBD5E1")} {delta_text} &middot; 14-day trend',
            daily_trend.tolist(),
        )
    with c2:
        cc1, cc2 = st.columns(2)
        with cc1:
            stat_cell("Last 48h", f"{recent_occ:.1%}", "", r_cls)
        with cc2:
            stat_cell("Regions", str(n_regions), "Active zones", "kpi-neutral")
        stat_cell("Days of data", f"{n_days:,}", "In feature store", "kpi-neutral")

    st.markdown("<br>", unsafe_allow_html=True)

    section("Region status")
    region_occ = (
        features.groupby("region")["avg_occupancy_rate"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    cols = st.columns(len(region_occ))
    for i, (_, row) in enumerate(region_occ.iterrows()):
        rate = row["avg_occupancy_rate"]
        rc = REGION_COLORS.get(row["region"], ACCENT)
        status = ("HIGH DEMAND", "red", "up") if rate > 0.85 else (("ON TARGET", "green", "flat") if rate >= 0.70 else ("UNDERUTILIZED", "blue", "down"))
        action = "Raise rates" if rate > 0.85 else ("Hold" if rate >= 0.70 else "Analyze demand")
        with cols[i]:
            st.markdown(f"""
            <div class="region-cell" style="--rc:{rc}">
                <div class="region-name">{row['region']}</div>
                <div class="region-value">{rate:.0%}</div>
                <div style="margin:6px 0">{badge(status[0], status[1], status[2])}</div>
                <div class="region-sub">{action} &rarr;</div>
            </div>
            """, unsafe_allow_html=True)

    section("Occupancy patterns")

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
            color_continuous_scale=[[0, "#F1F5F9"], [1, ACCENT]],
            zmin=0, zmax=1,
            title="Occupancy heatmap — hour × day",
            height=380,
        )
        fig.update_layout(**PLOTLY_THEME)
        fig.update_coloraxes(colorbar=dict(tickformat=".0%", len=0.8))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
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
            fill="tozeroy", fillcolor="rgba(10,90,140,0.08)",
            line=dict(color=ACCENT, width=2),
            name="Occupancy",
        ))
        fig2.add_hline(y=0.85, line_dash="dash", line_color=BAD,
                       annotation_text="85% upper", annotation_position="top right")
        fig2.add_hline(y=0.70, line_dash="dash", line_color=WARN,
                       annotation_text="70% lower", annotation_position="bottom right")
        fig2.update_layout(
            title="Daily average occupancy — last 30 days",
            height=380,
            **{**PLOTLY_THEME, "yaxis": dict(tickformat=".0%", range=[0, 1], gridcolor=LINE)},
        )
        st.plotly_chart(fig2, use_container_width=True)

    section("Peak hours by region")
    hourly_region = (
        features.groupby(["region", "hour_of_day"])["avg_occupancy_rate"]
        .mean()
        .reset_index()
    )
    fig3 = px.line(
        hourly_region, x="hour_of_day", y="avg_occupancy_rate", color="region",
        title="Average occupancy by hour of day",
        labels={"hour_of_day": "Hour", "avg_occupancy_rate": "Occupancy", "region": "Region"},
        height=350,
        color_discrete_map=REGION_COLORS,
    )
    fig3.add_hrect(y0=0.70, y1=0.85, fillcolor="rgba(17,97,73,0.06)",
                   line_width=0, annotation_text="Target band")
    fig3.update_yaxes(tickformat=".0%", range=[0, 1])
    fig3.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig3, use_container_width=True)


# ── Page: Revenue ─────────────────────────────────────────────────────────────

elif page == "Revenue":
    st.title("Revenue intelligence")
    st.caption("Current earnings vs full potential at 80% occupancy target with optimal pricing")

    if rev_summary.empty:
        st.error("Revenue data not generated. Run `python scripts/revenue_analyzer.py`.")
        st.stop()

    total_current = rev_summary["current_revenue"].sum()
    total_optimized = rev_summary["optimized_revenue"].sum()
    total_uplift = total_optimized - total_current
    uplift_pct = total_uplift / max(total_current, 0.01) * 100

    n_days = rev_summary["date"].nunique()
    annual_current = total_current / max(n_days, 1) * 365
    annual_optimized = total_optimized / max(n_days, 1) * 365
    annual_uplift = annual_optimized - annual_current

    daily_series = rev_summary.sort_values("date")["current_revenue"].tail(14)

    c1, c2 = st.columns([1.7, 1])
    with c1:
        hero_with_sparkline(
            "Potential uplift",
            f"${total_uplift:+,.0f}",
            f"+{uplift_pct:.1f}% gain available &middot; revenue trend, 14 days",
            daily_series.tolist(),
        )
    with c2:
        cc1, cc2 = st.columns(2)
        with cc1:
            stat_cell("Current revenue", f"${total_current:,.0f}", f"Over {n_days} days", "kpi-neutral")
        with cc2:
            stat_cell("Optimized revenue", f"${total_optimized:,.0f}", "At 80% occupancy", "kpi-good")
        stat_cell("Annual projection", f"${annual_uplift:+,.0f}", "Incremental per year", "kpi-good")

    callout("""
        <b>How to read this.</b> "Current" is what meters actually earned at today's rates and
        occupancy. "Optimized" is projected earnings if occupancy reaches the city's 80% target
        with SMC 11.16.121 rate adjustments. The uplift is always positive — the city is leaving
        money on the table relative to target utilization.
    """)

    section("Revenue trend")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rev_summary["date"], y=rev_summary["current_revenue"],
        name="Current revenue", marker_color=ACCENT,
        opacity=0.85,
    ))
    if "target_revenue" in rev_summary.columns:
        fig.add_trace(go.Scatter(
            x=rev_summary["date"], y=rev_summary["target_revenue"],
            name="At 80% occupancy (current rates)",
            line=dict(color=WARN, width=2, dash="dot"),
        ))
    fig.add_trace(go.Scatter(
        x=rev_summary["date"], y=rev_summary["optimized_revenue"],
        name="Optimized (80% occ + optimal rates)",
        line=dict(color=GOOD, width=2.5),
    ))
    fig.update_layout(
        title="Daily revenue — current vs potential",
        yaxis_title="Revenue ($)",
        barmode="overlay",
        height=380,
        **PLOTLY_THEME,
    )
    st.plotly_chart(fig, use_container_width=True)

    if not rev_detail.empty:
        col_left, col_right = st.columns(2)

        with col_left:
            section("Uplift by region")
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
                marker_color=GOOD,
                text=region_uplift["uplift"].map("${:,.0f}".format),
                textposition="outside",
            ))
            fig2.update_layout(
                title="Revenue uplift by region ($/period)",
                xaxis_title="Uplift ($)",
                height=320,
                **PLOTLY_THEME,
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col_right:
            section("Uplift heatmap — hour × region")
            pivot = rev_detail.pivot_table(
                index="hour_of_day", columns="region",
                values="revenue_uplift", aggfunc="mean",
            )
            fig3 = px.imshow(
                pivot,
                labels={"x": "Region", "y": "Hour", "color": "Uplift ($/hr)"},
                color_continuous_scale=[[0, "#F1F5F9"], [1, GOOD]],
                title="Revenue uplift per hour by region",
                height=320,
            )
            fig3.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig3, use_container_width=True)


# ── Page: Forecast ────────────────────────────────────────────────────────────

elif page == "Forecast":
    st.title("Demand forecast")
    st.caption("7-day occupancy predictions by region — LightGBM model")

    if model is None:
        st.error("No trained model found. Run `python scripts/train_model.py` first.")
        st.stop()
    if features.empty:
        st.error("No feature data. Run the pipeline first.")
        st.stop()

    perf_r2 = perf.get("r2", 0)
    perf_rmse = perf.get("rmse", 0)
    stat_row([
        {"label": "Model R²", "value": f"{perf_r2:.3f}", "delta": "Variance explained",
         "delta_class": "kpi-good" if perf_r2 > 0.85 else "kpi-warn", "hero": True},
        {"label": "RMSE", "value": f"{perf_rmse:.3f}", "delta": "Occupancy error"},
        {"label": "Samples", "value": f"{int(perf.get('n_train', 0)):,}", "delta": "12-mo window"},
    ])

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

    selected_region = st.selectbox("Select region", sorted(regions))
    region_forecast = future_df[future_df["region"] == selected_region].copy()

    over = region_forecast[region_forecast["predicted_occupancy"] > 0.85]
    under = region_forecast[region_forecast["predicted_occupancy"] < 0.70]

    stat_row([
        {"label": "Avg predicted", "value": f"{region_forecast['predicted_occupancy'].mean():.1%}",
         "delta": "Next 7 days", "hero": True},
        {"label": "Hours over 85%", "value": str(len(over)), "delta": "Rate increase window",
         "delta_class": "kpi-bad" if len(over) > 0 else "kpi-neutral"},
        {"label": "Hours under 70%", "value": str(len(under)), "delta": "Demand stimulus window",
         "delta_class": "kpi-warn" if len(under) > 0 else "kpi-neutral"},
    ])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=region_forecast["hour"], y=region_forecast["predicted_occupancy"],
        fill="tozeroy", fillcolor="rgba(10,90,140,0.08)",
        line=dict(color=ACCENT, width=2),
        name="Predicted occupancy",
    ))
    if not over.empty:
        fig.add_trace(go.Scatter(
            x=over["hour"], y=over["predicted_occupancy"],
            mode="markers", marker=dict(color=BAD, size=6, symbol="diamond"),
            name=">85% (raise rates)",
        ))
    fig.add_hrect(y0=0.70, y1=0.85, fillcolor="rgba(17,97,73,0.06)", line_width=0,
                  annotation_text="Target band", annotation_position="top left")
    fig.add_hline(y=0.85, line_dash="dash", line_color=BAD, line_width=1)
    fig.add_hline(y=0.70, line_dash="dash", line_color=WARN, line_width=1)
    fig.update_layout(
        title=f"7-day occupancy forecast — {selected_region}",
        height=400,
        **{**PLOTLY_THEME, "yaxis": dict(tickformat=".0%", range=[0, 1], gridcolor=LINE)},
    )
    st.plotly_chart(fig, use_container_width=True)

    if not over.empty:
        section("High-demand hours (>85%) — action required")
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

elif page == "Pricing":
    st.title("Dynamic pricing recommendations")
    st.caption("Seattle Municipal Code 11.16.121 — Performance-Based Parking Pricing")

    callout("""
        <b>Legal authority.</b> SMC 11.16.121 authorizes meter rate adjustments to achieve
        70–85% occupancy. Rate bounds: <b>$0.50–$8.00/hour</b>.
        Rate changes require City Council approval.
    """, kind="legal")

    if pricing.empty:
        st.error("No pricing recommendations. Run `python scripts/pricing_optimizer.py`.")
        st.stop()

    increases = (pricing["action"] == "increase").sum()
    decreases = (pricing["action"] == "decrease").sum()
    holds = (pricing["action"] == "hold").sum()
    rev_impact = pricing["revenue_delta"].sum() if "revenue_delta" in pricing.columns else 0

    stat_row([
        {"label": "Total revenue impact", "value": f"${rev_impact:+,.0f}/hr", "delta": "If all changes applied",
         "delta_class": "kpi-good" if rev_impact >= 0 else "kpi-warn", "hero": True},
        {"label": "Rate increases", "value": str(increases), "delta": "Zones over 85%",
         "delta_class": "kpi-bad" if increases > 0 else "kpi-neutral"},
        {"label": "Rate decreases", "value": str(decreases), "delta": "Zones under 70%"},
        {"label": "No change", "value": str(holds), "delta": "In target band", "delta_class": "kpi-good"},
    ])

    if "revenue_delta" in pricing.columns:
        section("Revenue impact by region")
        region_impact = (
            pricing.groupby("region")["revenue_delta"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        colors = [GOOD if v >= 0 else BAD for v in region_impact["revenue_delta"]]
        fig = go.Figure(go.Bar(
            x=region_impact["region"],
            y=region_impact["revenue_delta"],
            marker_color=colors,
            text=region_impact["revenue_delta"].map("${:+,.0f}".format),
            textposition="outside",
        ))
        fig.add_hline(y=0, line_color=MUTED, line_width=1)
        fig.update_layout(
            title="Estimated revenue delta by region ($/hr if all changes applied)",
            yaxis_title="Revenue impact ($/hr)",
            height=320,
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig, use_container_width=True)

    section("Recommendations table")
    fc1, fc2 = st.columns(2)
    region_filter = fc1.selectbox("Filter by region", ["All"] + sorted(pricing["region"].unique()))
    action_filter = fc2.selectbox("Filter by action", ["All", "increase", "decrease", "hold"])

    disp = pricing.copy()
    if region_filter != "All":
        disp = disp[disp["region"] == region_filter]
    if action_filter != "All":
        disp = disp[disp["action"] == action_filter]

    def color_action(val):
        if val == "increase":
            return f"background-color:#FBEEEE;color:{BAD}"
        if val == "decrease":
            return f"background-color:#EAF1F6;color:{ACCENT}"
        return f"color:{MUTED}"

    styled = disp[["region", "hour_of_day", "avg_predicted_occupancy",
                    "current_rate", "recommended_rate", "rate_change",
                    "action", "revenue_delta"]].copy()
    styled["avg_predicted_occupancy"] = styled["avg_predicted_occupancy"].map("{:.1%}".format)
    styled["current_rate"] = styled["current_rate"].map("${:.2f}".format)
    styled["recommended_rate"] = styled["recommended_rate"].map("${:.2f}".format)
    styled["rate_change"] = styled["rate_change"].map("${:+.2f}".format)
    styled["revenue_delta"] = styled["revenue_delta"].map("${:+.2f}".format)
    styled.columns = ["Region", "Hour", "Pred. occ.", "Current rate",
                      "Rec. rate", "Change", "Action", "Rev. impact/hr"]
    st.dataframe(
        styled.style.map(color_action, subset=["Action"]),
        use_container_width=True, hide_index=True,
    )


# ── Page: Infrastructure ROI ──────────────────────────────────────────────────

elif page == "Infrastructure":
    st.title("Infrastructure investment ROI")
    st.caption("Cost-benefit analysis: should Seattle build more parking capacity?")

    if roi_df.empty:
        st.error("No ROI data. Run `python scripts/infrastructure_roi.py`.")
        st.stop()

    viable = roi_df[roi_df["viable"]]
    best_roi = viable["roi_percent"].max() if not viable.empty else 0
    best_payback = viable["payback_years"].min() if not viable.empty else 0

    stat_row([
        {"label": "Best ROI", "value": f"{best_roi:.1f}%", "delta": "Top viable scenario",
         "delta_class": "kpi-good" if best_roi > 0 else "kpi-neutral", "hero": True},
        {"label": "Scenarios analyzed", "value": str(len(roi_df))},
        {"label": "Viable investments", "value": str(len(viable)), "delta": "Breakeven < demand",
         "delta_class": "kpi-good" if len(viable) > 0 else "kpi-warn"},
        {"label": "Fastest payback", "value": f"{best_payback:.1f} yrs" if best_payback > 0 else "N/A"},
    ])

    col1, col2 = st.columns(2)
    region_filter = col1.selectbox("Region", ["All"] + sorted(roi_df["region"].unique()))
    infra_filter = col2.selectbox("Infrastructure type", ["All"] + sorted(roi_df["infra_type"].unique()))

    disp = roi_df.copy()
    if region_filter != "All":
        disp = disp[disp["region"] == region_filter]
    if infra_filter != "All":
        disp = disp[disp["infra_type"] == infra_filter]

    col_left, col_right = st.columns(2)

    with col_left:
        section("ROI by scenario")
        fig = px.bar(
            disp.sort_values("roi_percent", ascending=False),
            x="region", y="roi_percent", color="infra_type",
            barmode="group", facet_col="n_new_spaces",
            labels={"roi_percent": "ROI %", "region": "Region", "infra_type": "Type"},
            height=380,
            color_discrete_map={"surface_lot": ACCENT, "structured_garage": WARN,
                                 "underground": BAD},
        )
        fig.add_hline(y=0, line_dash="solid", line_color=BAD, line_width=1)
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        section("Breakeven vs current occupancy")
        fig2 = px.scatter(
            disp,
            x="current_occupancy", y="breakeven_occupancy",
            color="viable", size="n_new_spaces",
            hover_name="region",
            hover_data={"roi_percent": ":.1f", "payback_years": ":.1f"},
            color_discrete_map={True: GOOD, False: BAD},
            labels={"current_occupancy": "Current occupancy",
                    "breakeven_occupancy": "Breakeven occupancy"},
            height=380,
        )
        fig2.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                       line=dict(dash="dash", color=MUTED))
        fig2.update_xaxes(tickformat=".0%")
        fig2.update_yaxes(tickformat=".0%")
        fig2.update_layout(**PLOTLY_THEME)
        st.caption("Points below the diagonal are viable (current demand exceeds breakeven).")
        st.plotly_chart(fig2, use_container_width=True)

    section("Scenario details")
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
    table.columns = ["Region", "Type", "Spaces", "Curr. occ.", "Breakeven",
                     "Construction cost", "Net annual", "ROI %", "Payback (yrs)", "Viable"]
    st.dataframe(
        table.style.map(
            lambda v: "background-color:#EAF3EF" if v is True else
                      ("background-color:#FBEEEE" if v is False else ""),
            subset=["Viable"]
        ),
        use_container_width=True, hide_index=True,
    )
    st.caption("Costs: surface lot $5K/space, structured garage $45K/space, underground $90K/space. Bond: 4.5%, 20yr.")


# ── Page: Geo Map ─────────────────────────────────────────────────────────────

elif page == "Geo Map":
    import json as _json

    st.title("Parking demand map")
    st.caption("Neighborhood polygons from Seattle Open Data · Circle size = avg parking spaces · Click for full stats")

    if features.empty:
        st.error("No data to display.")
        st.stop()

    REGION_COORDS = {
        "Downtown Seattle":      (47.6081, -122.3321),
        "Capitol Hill":          (47.6238, -122.3184),
        "South Lake Union":      (47.6232, -122.3384),
        "Ballard":               (47.6765, -122.3862),
        "International District":(47.5984, -122.3225),
    }

    reg_stats = features.groupby("region").agg(
        avg_occ=("avg_occupancy_rate", "mean"),
        peak_occ=("peak_occupancy_rate", "mean"),
        avg_spaces=("total_spaces", "mean"),
        num_blockfaces=("num_blockfaces", "mean"),
    ).reset_index()

    peak_hour = (
        features.groupby(["region", "hour_of_day"])["avg_occupancy_rate"]
        .mean()
        .reset_index()
    )
    peak_hour = peak_hour.loc[peak_hour.groupby("region")["avg_occupancy_rate"].idxmax()]
    peak_hour = peak_hour.rename(columns={"hour_of_day": "peak_hour", "avg_occupancy_rate": "_ph_occ"})
    reg_stats = reg_stats.merge(peak_hour[["region", "peak_hour"]], on="region", how="left")

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
            return BAD
        elif rate >= 0.70:
            return GOOD
        else:
            return ACCENT

    def occ_status(rate):
        if rate > 0.85:
            return "High demand (&gt;85%)"
        elif rate >= 0.70:
            return "On target (70–85%)"
        else:
            return "Underutilized (&lt;70%)"

    action_labels = {"increase": "Raise rates", "decrease": "Lower rates", "hold": "Hold rates"}
    DISTRICT_COLORS = REGION_COLORS

    min_sp = reg_stats["avg_spaces"].min()
    max_sp = reg_stats["avg_spaces"].max()
    def space_radius(spaces):
        if max_sp == min_sp:
            return 35
        return 18 + 32 * (spaces - min_sp) / (max_sp - min_sp)

    _geojson_path = ROOT / "data" / "seattle_5_neighborhoods.geojson"
    _neighborhoods_geojson = None
    if _geojson_path.exists():
        with open(_geojson_path) as _f:
            _neighborhoods_geojson = _json.load(_f)

    _occ_by_district = dict(zip(reg_stats["region"], reg_stats["avg_occ"]))

    m = folium.Map(
        location=[47.630, -122.330],
        zoom_start=12,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

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
                    "fillOpacity": 0.10,
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

        color = DISTRICT_COLORS.get(row["region"], occ_color(row["avg_occ"]))
        occ_col = occ_color(row["avg_occ"])
        radius = space_radius(row["avg_spaces"])
        action = row["top_action"]
        peak_h = int(row["peak_hour"]) if pd.notna(row.get("peak_hour")) else 0
        peak_label = f"{peak_h % 12 or 12}{'am' if peak_h < 12 else 'pm'}"

        popup_html = f"""
        <div style="font-family:'Source Sans 3',Arial,sans-serif;min-width:220px;background:{CARD};color:{BODY};
                    border-radius:8px;padding:14px;border-left:4px solid {color};box-shadow:0 1px 4px rgba(15,23,42,0.12)">
            <div style="font-size:14px;font-weight:700;color:{color};margin-bottom:10px;
                        border-bottom:1px solid {LINE};padding-bottom:6px">
                {row['region']}
            </div>
            <table style="width:100%;border-collapse:collapse;font-size:12px">
                <tr>
                    <td style="color:{MUTED};padding:3px 0">Avg occupancy</td>
                    <td style="text-align:right;font-weight:700;color:{occ_col}">{row['avg_occ']:.0%}</td>
                </tr>
                <tr>
                    <td style="color:{MUTED};padding:3px 0">Peak occupancy</td>
                    <td style="text-align:right;font-weight:700;color:{INK}">{row['peak_occ']:.0%}</td>
                </tr>
                <tr>
                    <td style="color:{MUTED};padding:3px 0">Peak hour</td>
                    <td style="text-align:right;font-weight:700;color:{INK}">{peak_label}</td>
                </tr>
                <tr>
                    <td style="color:{MUTED};padding:3px 0">Avg spaces</td>
                    <td style="text-align:right;font-weight:700;color:{INK}">{int(row['avg_spaces']):,}</td>
                </tr>
                <tr>
                    <td style="color:{MUTED};padding:3px 0">Blockfaces</td>
                    <td style="text-align:right;font-weight:700;color:{INK}">{int(row['num_blockfaces'])}</td>
                </tr>
                <tr style="border-top:1px solid {LINE}">
                    <td style="color:{MUTED};padding:5px 0 3px">Current rate</td>
                    <td style="text-align:right;font-weight:700;color:{INK}">${row.get('current_rate', 0):.2f}/hr</td>
                </tr>
                <tr>
                    <td style="color:{MUTED};padding:3px 0">Recommended</td>
                    <td style="text-align:right;font-weight:700;color:{occ_col}">${row.get('recommended_rate', 0):.2f}/hr</td>
                </tr>
                <tr>
                    <td style="color:{MUTED};padding:3px 0">Pricing action</td>
                    <td style="text-align:right;font-weight:700;color:{occ_col}">{action_labels.get(action, action)}</td>
                </tr>
            </table>
            <div style="margin-top:8px;padding:5px 8px;background:{occ_col}14;border-radius:4px;
                        font-size:11px;color:{occ_col};font-weight:700;text-align:center">
                {occ_status(row['avg_occ'])}
            </div>
        </div>
        """

        tooltip_html = (
            f"<b style='color:{color}'>{row['region']}</b><br>"
            f"Occupancy: <b style='color:{occ_col}'>{row['avg_occ']:.0%}</b><br>"
            f"Spaces: {int(row['avg_spaces']):,} · Peak: {peak_label}"
        )

        folium.CircleMarker(
            location=coords, radius=radius + 12, color=color, fill=True,
            fill_color=color, fill_opacity=0.08, weight=1, opacity=0.3,
        ).add_to(m)

        folium.CircleMarker(
            location=coords, radius=radius, color=color, fill=True,
            fill_color=color, fill_opacity=0.85, weight=2, opacity=0.95,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=folium.Tooltip(tooltip_html, sticky=False),
        ).add_to(m)

        folium.Marker(
            location=coords,
            icon=folium.DivIcon(
                html=f"""<div style="font-family:'Source Sans 3',Arial,sans-serif;font-size:11px;font-weight:700;
                                     color:white;text-align:center;width:80px;margin-left:-40px;
                                     text-shadow:0 1px 3px rgba(0,0,0,0.6)">
                            {row['avg_occ']:.0%}
                         </div>""",
                icon_size=(80, 20),
                icon_anchor=(40, 10),
            ),
        ).add_to(m)

    st_folium(m, use_container_width=True, height=540, returned_objects=[])

    section("Region breakdown")
    cols = st.columns(len(reg_stats))
    for i, (_, row) in enumerate(reg_stats.sort_values("avg_occ", ascending=False).iterrows()):
        rate = row["avg_occ"]
        dcolor = DISTRICT_COLORS.get(row["region"], "#888888")
        occ_col = occ_color(rate)
        status = "HIGH DEMAND" if rate > 0.85 else ("ON TARGET" if rate >= 0.70 else "UNDERUTILIZED")
        action = row["top_action"]
        peak_h = int(row["peak_hour"]) if pd.notna(row.get("peak_hour")) else 0
        peak_label = f"{peak_h % 12 or 12}{'am' if peak_h < 12 else 'pm'}"
        with cols[i]:
            st.markdown(f"""
            <div class="region-cell" style="--rc:{dcolor}">
                <div class="region-name">{row['region']}</div>
                <div class="region-value" style="color:{occ_col}">{rate:.0%}</div>
                <div class="region-sub" style="color:{occ_col};font-weight:700">{status}</div>
                <div class="region-sub" style="margin-top:6px">{int(row['avg_spaces']):,} spaces &middot; peak {peak_label}</div>
                <div class="region-sub" style="color:{occ_col}">{action}</div>
            </div>
            """, unsafe_allow_html=True)
