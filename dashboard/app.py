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

# Seattle has 23 official paid parking areas — too many for hand-picked
# colours, so assign deterministically from an accessible categorical ramp.
AREA_PALETTE = [
    "#0A5A8C", "#116149", "#9C2626", "#5B3A8E", "#8A4B0C",
    "#0E7A8C", "#7A1F5C", "#2F5F1F", "#8C3D0A", "#3D3D8C",
]


def area_color(area: str) -> str:
    return AREA_PALETTE[hash(str(area)) % len(AREA_PALETTE)]

NAV_ITEMS = [
    ("Overview", "overview"),
    ("Meters", "meters"),
    ("Pricing", "pricing"),
    ("Utilization", "utilization"),
    ("Infrastructure", "infrastructure"),
    ("Map", "map"),
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
    colorway=AREA_PALETTE,
)



# ── Loaders ───────────────────────────────────────────────────────────────────
# Every artifact here is small and committed. The full meter-level feature table
# (15M rows) is deliberately NOT loaded — it is a training intermediate, far too
# large for Streamlit Cloud.

def _read(name, **kw):
    f = DATA_DIR / name
    return pd.read_csv(f, **kw) if f.exists() else pd.DataFrame()


@st.cache_data(ttl=3600)
def load_meters():
    return _read("meter_summary.csv")


@st.cache_data(ttl=3600)
def load_recommendations():
    return _read("meter_recommendations.csv")


@st.cache_data(ttl=3600)
def load_area_daily():
    return _read("area_daily.csv", parse_dates=["date"])


@st.cache_data(ttl=3600)
def load_area_profile():
    return _read("area_hour_profile.csv")


@st.cache_data(ttl=3600)
def load_utilization():
    return _read("revenue_summary.csv", parse_dates=["date"])


@st.cache_data(ttl=3600)
def load_area_hour():
    return _read("revenue_by_area_hour.csv")


@st.cache_data(ttl=3600)
def load_roi():
    return _read("infrastructure_roi.csv")


@st.cache_data(ttl=3600)
def load_perf():
    f = MODELS_DIR / "performance" / "final_model_performance.csv"
    return pd.read_csv(f).iloc[0].to_dict() if f.exists() else {}


meters   = load_meters()
recs     = load_recommendations()
area_day = load_area_daily()
profile  = load_area_profile()
util     = load_utilization()
area_hr  = load_area_hour()
roi_df   = load_roi()
perf     = load_perf()

ACTION_LABEL = {
    "increase": ("RAISE RATE", "red", "up"),
    "decrease": ("LOWER RATE", "blue", "down"),
    "hold":     ("ON TARGET", "green", "flat"),
    "unknown":  ("NO DATA", "yellow", "dot"),
}


def fmt_hour(h):
    h = int(h)
    return f"{h % 12 or 12}{'am' if h < 12 else 'pm'}"


def adjustment_text(v):
    if pd.isna(v) or v == 0:
        return "no change"
    return f"{v:+.2f}/hr"


# ── Top command bar ───────────────────────────────────────────────────────────

meta_html = ""
if not area_day.empty:
    meta_html += f'<span class="meta-pill">Data through <b>{area_day["date"].max():%b %d, %Y}</b></span>'
if not meters.empty:
    meta_html += f'<span class="meta-pill"><b>{len(meters):,}</b> blockfaces</span>'
    meta_html += f'<span class="meta-pill"><b>{meters["paidparkingarea"].nunique()}</b> official areas</span>'
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

nav_cols = st.columns([1.0, 0.9, 0.9, 1.1, 1.4, 0.8, 3.4])
for (label, slug), col in zip(NAV_ITEMS, nav_cols[:-1]):
    with col:
        with st.container(key=f"navitem_{slug}"):
            if st.button(label, key=f"navbtn_{slug}", use_container_width=False):
                st.session_state.page = label

active_slug = dict(NAV_ITEMS)[st.session_state.page]
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

if meters.empty:
    st.error("No meter data found. Run the pipeline: `python run_pipeline.py`")
    st.stop()


# ── Page: Overview ────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("Citywide overview")
    st.caption("Every paid blockface in Seattle, graded against the SMC 11.16.121 "
               "target band of 70–85% occupancy")

    citywide_occ = float((meters["avg_occupancy"] * meters["spaces"]).sum() / meters["spaces"].sum())
    counts = meters["primary_action"].value_counts()
    on_target = int(counts.get("hold", 0))
    total_spaces = int(meters["spaces"].sum())

    trend = (
        area_day.groupby("date")["avg_occupancy_rate"].mean().tail(14).tolist()
        if not area_day.empty else [citywide_occ] * 14
    )

    c1, c2 = st.columns([1.7, 1])
    with c1:
        hero_with_sparkline(
            "Citywide occupancy",
            f"{citywide_occ:.1%}",
            f'{icon("down", "#CBD5E1")} Below the 70% target floor &middot; 14-day trend',
            trend,
        )
    with c2:
        cc1, cc2 = st.columns(2)
        with cc1:
            stat_cell("On target", f"{on_target}", f"of {len(meters):,} blockfaces",
                      "kpi-good" if on_target else "kpi-warn")
        with cc2:
            stat_cell("Paid spaces", f"{total_spaces:,}", "citywide", "kpi-neutral")
        stat_cell("Below target", f"{int(counts.get('decrease', 0)):,}",
                  "blockfaces under 70% occupancy", "kpi-warn")

    callout("""
        <b>How to read this.</b> Seattle's paid parking is running well below the
        70–85% band the municipal code targets. Under SMC 11.16.121 that points
        toward <b>lower</b> rates to attract parkers, not higher ones — raising
        prices on half-empty blocks would push utilization further down.
        Rate changes below are expressed as an adjustment to whatever rate is
        currently posted, because Seattle publishes no per-meter rate data.
    """)

    section("Areas needing attention")
    area_stats = (
        meters.groupby("paidparkingarea")
        .apply(lambda g: pd.Series({
            "occupancy": (g["avg_occupancy"] * g["spaces"]).sum() / g["spaces"].sum(),
            "blockfaces": len(g),
            "spaces": g["spaces"].sum(),
            "on_target": int((g["primary_action"] == "hold").sum()),
        }), include_groups=False)
        .reset_index()
        .sort_values("occupancy", ascending=False)
    )

    top = area_stats.head(5)
    cols = st.columns(len(top))
    for i, (_, row) in enumerate(top.iterrows()):
        rate = row["occupancy"]
        label, colour, ic = ACTION_LABEL[
            "increase" if rate > 0.85 else ("hold" if rate >= 0.70 else "decrease")
        ]
        with cols[i]:
            st.markdown(f"""
            <div class="region-cell" style="--rc:{area_color(row['paidparkingarea'])}">
                <div class="region-name">{row['paidparkingarea']}</div>
                <div class="region-value">{rate:.0%}</div>
                <div style="margin:6px 0">{badge(label, colour, ic)}</div>
                <div class="region-sub">{int(row['blockfaces'])} blockfaces &middot;
                    {int(row['spaces'])} spaces</div>
            </div>
            """, unsafe_allow_html=True)

    section("Occupancy by area")
    fig = px.bar(
        area_stats.sort_values("occupancy"),
        x="occupancy", y="paidparkingarea", orientation="h",
        labels={"occupancy": "Average occupancy", "paidparkingarea": ""},
        height=560, text=area_stats.sort_values("occupancy")["occupancy"].map("{:.0%}".format),
    )
    fig.update_traces(marker_color=ACCENT, textposition="outside")
    fig.add_vrect(x0=0.70, x1=0.85, fillcolor="rgba(17,97,73,0.10)", line_width=0,
                  annotation_text="70–85% target band", annotation_position="top")
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    fig.update_layout(title="Every official paid parking area vs the target band",
                      **PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)

    if not profile.empty:
        section("When demand actually happens")
        DAYS = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        piv = (profile.groupby(["hour_of_day", "day_of_week"])["avg_occupancy_rate"]
               .mean().unstack(fill_value=0))
        piv.columns = [DAYS.get(c, c) for c in piv.columns]
        fig2 = px.imshow(
            piv, labels={"x": "Day", "y": "Hour", "color": "Occupancy"},
            color_continuous_scale=[[0, "#F1F5F9"], [1, ACCENT]], zmin=0, zmax=1,
            title="Citywide occupancy — hour x day", height=420,
        )
        fig2.update_layout(**PLOTLY_THEME)
        fig2.update_coloraxes(colorbar=dict(tickformat=".0%", len=0.8))
        st.plotly_chart(fig2, use_container_width=True)


# ── Page: Meters (per-blockface drill-down) ──────────────────────────────────

elif page == "Meters":
    st.title("Blockface detail")
    st.caption("Every metered blockface, its measured occupancy, and the rate "
               "adjustment SMC 11.16.121 implies")

    f1, f2, f3 = st.columns([2, 1.4, 1.4])
    area_pick = f1.selectbox("Area", ["All areas"] + sorted(meters["paidparkingarea"].unique()))
    action_pick = f2.selectbox("Recommended action",
                               ["All", "decrease", "hold", "increase"])
    sort_pick = f3.selectbox("Sort by",
                             ["Lowest occupancy", "Highest occupancy", "Most spaces"])

    view = meters.copy()
    if area_pick != "All areas":
        view = view[view["paidparkingarea"] == area_pick]
    if action_pick != "All":
        view = view[view["primary_action"] == action_pick]

    view = view.sort_values(
        {"Lowest occupancy": "avg_occupancy",
         "Highest occupancy": "avg_occupancy",
         "Most spaces": "spaces"}[sort_pick],
        ascending=(sort_pick == "Lowest occupancy"),
    )

    search = st.text_input("Find a blockface", placeholder="e.g. PIKE ST, 1ST AVE, BROADWAY")
    if search:
        view = view[view["blockfacename"].str.contains(search, case=False, na=False)]

    stat_row([
        {"label": "Blockfaces shown", "value": f"{len(view):,}", "hero": True,
         "delta": f"of {len(meters):,} citywide"},
        {"label": "Spaces", "value": f"{int(view['spaces'].sum()):,}"},
        {"label": "Mean occupancy",
         "value": f"{view['avg_occupancy'].mean():.1%}" if len(view) else "—"},
    ])

    if view.empty:
        st.info("No blockfaces match those filters.")
        st.stop()

    section("Blockfaces")
    table = view[[
        "blockfacename", "paidparkingarea", "avg_occupancy", "peak_hour_occupancy",
        "peak_hour", "spaces", "mean_adjustment", "primary_action",
    ]].copy()
    table["avg_occupancy"] = table["avg_occupancy"].map("{:.1%}".format)
    table["peak_hour_occupancy"] = table["peak_hour_occupancy"].map("{:.1%}".format)
    table["peak_hour"] = table["peak_hour"].map(fmt_hour)
    table["spaces"] = table["spaces"].round(0).astype(int)
    table["mean_adjustment"] = table["mean_adjustment"].map(adjustment_text)
    table.columns = ["Blockface", "Area", "Avg occ.", "Peak occ.", "Peak hour",
                     "Spaces", "Rate change", "Action"]

    def colour_action(v):
        return {"increase": f"background-color:#FBEEEE;color:{BAD}",
                "decrease": f"background-color:#EAF1F6;color:{ACCENT}",
                "hold": f"background-color:#EAF3EF;color:{GOOD}"}.get(v, f"color:{MUTED}")

    st.dataframe(table.style.map(colour_action, subset=["Action"]),
                 use_container_width=True, hide_index=True, height=460)

    section("Hour-by-hour detail")
    pick = st.selectbox("Blockface", view["blockfacename"].tolist())
    detail = recs[recs["blockfacename"] == pick].sort_values("hour_of_day")

    if detail.empty:
        st.info("No hourly detail for this blockface.")
    else:
        figd = go.Figure()
        figd.add_trace(go.Bar(
            x=detail["hour_of_day"], y=detail["avg_occupancy"],
            marker_color=[BAD if a == "increase" else (GOOD if a == "hold" else ACCENT)
                          for a in detail["action"]],
            text=detail["rate_adjustment"].map(lambda v: "" if v == 0 else f"{v:+.2f}"),
            textposition="outside",
            hovertemplate="%{x}:00<br>Occupancy %{y:.1%}<extra></extra>",
        ))
        figd.add_hrect(y0=0.70, y1=0.85, fillcolor="rgba(17,97,73,0.10)", line_width=0,
                       annotation_text="target band")
        figd.update_yaxes(tickformat=".0%", range=[0, 1])
        figd.update_layout(title=f"{pick} — occupancy and recommended rate change by hour",
                           xaxis_title="Hour of day", height=380, **PLOTLY_THEME)
        st.plotly_chart(figd, use_container_width=True)

        meta = view[view["blockfacename"] == pick].iloc[0]
        st.caption(
            f"{meta['paidparkingarea']} · {int(meta['spaces'])} spaces · "
            f"{int(meta.get('n_meters', 1))} meter(s) · "
            f"{int(meta['time_limit']) if pd.notna(meta.get('time_limit')) else '—'} min limit"
        )


# ── Page: Pricing ─────────────────────────────────────────────────────────────

elif page == "Pricing":
    st.title("Rate recommendations")
    st.caption("Seattle Municipal Code 11.16.121 — Performance-Based Parking Pricing")

    callout("""
        <b>Rate changes are relative.</b> Seattle publishes no per-meter posted
        rate — the <code>paidparkingrate</code> field in the city's own occupancy
        feed is empty on all 25.2M records. Rather than invent a baseline, every
        figure below is an adjustment to apply to the rate currently posted at
        that blockface. Legal bounds: <b>$0.50–$8.00/hr</b>. Rate changes require
        City Council approval.
    """, kind="legal")

    counts = meters["primary_action"].value_counts()
    stat_row([
        {"label": "Blockfaces to lower", "value": f"{int(counts.get('decrease', 0)):,}",
         "delta": "under 70% occupancy", "delta_class": "kpi-warn", "hero": True},
        {"label": "To raise", "value": f"{int(counts.get('increase', 0)):,}",
         "delta": "over 85%", "delta_class": "kpi-bad"},
        {"label": "Leave alone", "value": f"{int(counts.get('hold', 0)):,}",
         "delta": "in target band", "delta_class": "kpi-good"},
    ])

    section("Average recommended change by area")
    by_area = (
        meters.groupby("paidparkingarea")
        .agg(mean_adjustment=("mean_adjustment", "mean"),
             blockfaces=("blockfacename", "count"))
        .reset_index().sort_values("mean_adjustment")
    )
    figp = go.Figure(go.Bar(
        x=by_area["mean_adjustment"], y=by_area["paidparkingarea"], orientation="h",
        marker_color=[BAD if v > 0 else ACCENT for v in by_area["mean_adjustment"]],
        text=by_area["mean_adjustment"].map("{:+.2f}".format), textposition="outside",
        hovertemplate="%{y}<br>%{x:+.2f}/hr<extra></extra>",
    ))
    figp.add_vline(x=0, line_color=MUTED, line_width=1)
    figp.update_layout(title="Recommended change vs posted rate ($/hr)",
                       xaxis_title="Adjustment to posted rate ($/hr)",
                       height=560, **PLOTLY_THEME)
    st.plotly_chart(figp, use_container_width=True)

    section("Recommendations by hour")
    ha = st.selectbox("Area", ["All areas"] + sorted(recs["paidparkingarea"].unique()))
    hv = recs if ha == "All areas" else recs[recs["paidparkingarea"] == ha]
    hourly = hv.groupby("hour_of_day").agg(
        occupancy=("avg_occupancy", "mean"),
        adjustment=("rate_adjustment", "mean"),
        blockfaces=("blockfacename", "nunique"),
    ).reset_index()

    figh = go.Figure()
    figh.add_trace(go.Scatter(x=hourly["hour_of_day"], y=hourly["occupancy"],
                              name="Occupancy", line=dict(color=ACCENT, width=2.5)))
    figh.add_trace(go.Bar(x=hourly["hour_of_day"], y=hourly["adjustment"],
                          name="Rate change ($/hr)", marker_color=WARN,
                          opacity=0.55, yaxis="y2"))
    figh.add_hrect(y0=0.70, y1=0.85, fillcolor="rgba(17,97,73,0.10)", line_width=0)
    figh.update_layout(
        title=f"Occupancy and recommended rate change — {ha}",
        xaxis_title="Hour of day", height=400,
        yaxis=dict(title="Occupancy", tickformat=".0%", range=[0, 1], gridcolor=LINE),
        yaxis2=dict(title="Rate change ($/hr)", overlaying="y", side="right",
                    showgrid=False),
        **{k: v for k, v in PLOTLY_THEME.items() if k not in ("yaxis",)},
    )
    st.plotly_chart(figh, use_container_width=True)


# ── Page: Utilization ─────────────────────────────────────────────────────────

elif page == "Utilization":
    st.title("Utilization and revenue opportunity")
    st.caption("Measured in space-hours — the quantity the city actually observes")

    if util.empty:
        st.error("Run `python scripts/revenue_analyzer.py` first.")
        st.stop()

    occupied = util["occupied_space_hours"].sum()
    target = util["target_space_hours"].sum()
    unsold = util["unsold_space_hours"].sum()
    pct = occupied / target * 100 if target else 0

    c1, c2 = st.columns([1.7, 1])
    with c1:
        hero_with_sparkline(
            "Unsold space-hours vs target",
            f"{unsold/1e6:,.1f}M",
            f"Running at {pct:.0f}% of the 80% target &middot; daily trend",
            util.sort_values("date")["occupied_space_hours"].tail(14).tolist(),
        )
    with c2:
        cc1, cc2 = st.columns(2)
        with cc1:
            stat_cell("Occupied", f"{occupied/1e6:,.1f}M", "space-hours sold", "kpi-neutral")
        with cc2:
            stat_cell("Utilization", f"{util['utilization_pct'].mean():.0f}%",
                      "of capacity", "kpi-warn")
        stat_cell("At 80% target", f"{target/1e6:,.1f}M", "space-hours", "kpi-neutral")

    callout("""
        <b>Why there are no dollar totals.</b> Revenue equals rate x occupied
        space-hours. Seattle publishes the occupancy but not the per-meter rate,
        so any dollar figure here would be built on an invented number. Multiply
        the space-hours by the real posted rate to get dollars — at $2.00/hr,
        the unsold gap above is worth roughly
        <b>$""" + f"{unsold*2/1e6:,.0f}M" + """</b> in foregone revenue.
    """)

    section("Utilization over time")
    figu = go.Figure()
    figu.add_trace(go.Scatter(x=util["date"], y=util["occupied_space_hours"],
                              name="Occupied", fill="tozeroy",
                              fillcolor="rgba(10,90,140,0.12)",
                              line=dict(color=ACCENT, width=2)))
    figu.add_trace(go.Scatter(x=util["date"], y=util["target_space_hours"],
                              name="80% target", line=dict(color=GOOD, width=2, dash="dot")))
    figu.update_layout(title="Daily space-hours: occupied vs target",
                       yaxis_title="Space-hours", height=400, **PLOTLY_THEME)
    st.plotly_chart(figu, use_container_width=True)

    if not area_hr.empty:
        section("Where the gap is largest")
        gap = (area_hr.groupby("paidparkingarea")
               .agg(unsold=("unsold_space_hours", "sum"),
                    occupancy=("avg_occupancy", "mean"))
               .reset_index().sort_values("unsold", ascending=True).tail(15))
        figg = go.Figure(go.Bar(
            x=gap["unsold"], y=gap["paidparkingarea"], orientation="h",
            marker_color=WARN,
            text=gap["occupancy"].map("{:.0%} occ".format), textposition="outside",
        ))
        figg.update_layout(title="Unsold space-hours per average hour, by area",
                           xaxis_title="Unsold space-hours", height=520, **PLOTLY_THEME)
        st.plotly_chart(figg, use_container_width=True)


# ── Page: Infrastructure ──────────────────────────────────────────────────────

elif page == "Infrastructure":
    st.title("Infrastructure investment")
    st.caption("Should Seattle build more paid parking capacity?")

    if roi_df.empty:
        st.error("Run `python scripts/infrastructure_roi.py` first.")
        st.stop()

    viable = roi_df[roi_df["viable"]]
    cheapest = roi_df["breakeven_rate_per_hour"].min()

    stat_row([
        {"label": "Viable projects", "value": f"{len(viable)}",
         "delta": "of {} scenarios".format(len(roi_df)),
         "delta_class": "kpi-good" if len(viable) else "kpi-warn", "hero": True},
        {"label": "Lowest breakeven rate", "value": f"${cheapest:.2f}/hr",
         "delta": "to cover costs"},
        {"label": "Areas assessed", "value": f"{roi_df['paidparkingarea'].nunique()}"},
    ])

    if viable.empty:
        callout("""
            <b>No expansion is justified right now.</b> A project only makes sense
            if existing capacity is already near the 80% target — and no area in
            Seattle currently is. Adding spaces where blocks sit two-thirds empty
            would lower utilization further and add debt service against
            unsold capacity. The figures below show what each project
            <em>would</em> require if demand recovered.
        """)

    section("Breakeven rate by project type")
    figr = px.box(
        roi_df, x="infra_type", y="breakeven_rate_per_hour", color="infra_type",
        labels={"infra_type": "", "breakeven_rate_per_hour": "Breakeven rate ($/hr)"},
        height=420, color_discrete_sequence=AREA_PALETTE,
    )
    figr.add_hline(y=8.00, line_dash="dash", line_color=BAD,
                   annotation_text="$8.00 legal cap (SMC 11.16.121)")
    figr.update_layout(title="Hourly rate required to cover debt service and operations",
                       showlegend=False, **PLOTLY_THEME)
    st.plotly_chart(figr, use_container_width=True)

    section("Scenario detail")
    itype = st.selectbox("Project type", ["All"] + sorted(roi_df["infra_type"].unique()))
    rv = roi_df if itype == "All" else roi_df[roi_df["infra_type"] == itype]
    tbl = rv[["paidparkingarea", "infra_type", "n_new_spaces", "current_occupancy",
              "total_construction_cost", "total_annual_cost",
              "breakeven_rate_per_hour", "viable"]].copy()
    tbl["current_occupancy"] = tbl["current_occupancy"].map("{:.1%}".format)
    tbl["total_construction_cost"] = tbl["total_construction_cost"].map("${:,.0f}".format)
    tbl["total_annual_cost"] = tbl["total_annual_cost"].map("${:,.0f}".format)
    tbl["breakeven_rate_per_hour"] = tbl["breakeven_rate_per_hour"].map("${:.2f}".format)
    tbl.columns = ["Area", "Type", "New spaces", "Current occ.",
                   "Build cost", "Annual cost", "Breakeven rate", "Viable"]
    st.dataframe(tbl, use_container_width=True, hide_index=True, height=420)
    st.caption("Surface lot $5K/space · structured garage $45K/space · "
               "underground $90K/space. Municipal bond 4.5% over 20 years.")


# ── Page: Map ─────────────────────────────────────────────────────────────────

elif page == "Map":
    from folium.plugins import MarkerCluster

    st.title("Blockface map")
    st.caption("Every metered blockface in Seattle. Zoom in to separate clusters; "
               "click a marker for detail.")

    geo = meters.dropna(subset=["lat", "lon"])
    if geo.empty:
        st.error("No coordinates available. Run `python scripts/fetch_meter_registry.py`.")
        st.stop()

    mc1, mc2 = st.columns([2, 2])
    map_area = mc1.selectbox("Area", ["All areas"] + sorted(geo["paidparkingarea"].unique()))
    map_action = mc2.selectbox("Show", ["All blockfaces", "Below target (under 70%)",
                                        "On target (70–85%)", "Above target (over 85%)"])

    view = geo if map_area == "All areas" else geo[geo["paidparkingarea"] == map_area]
    action_filter = {"Below target (under 70%)": "decrease",
                     "On target (70–85%)": "hold",
                     "Above target (over 85%)": "increase"}.get(map_action)
    if action_filter:
        view = view[view["primary_action"] == action_filter]

    st.caption(f"Showing {len(view):,} of {len(geo):,} blockfaces")

    if view.empty:
        st.info("No blockfaces match those filters.")
        st.stop()

    m = folium.Map(
        location=[view["lat"].mean(), view["lon"].mean()],
        zoom_start=13 if map_area != "All areas" else 12,
        tiles="CartoDB positron", prefer_canvas=True,
    )
    cluster = MarkerCluster(name="Blockfaces").add_to(m)

    for _, r in view.iterrows():
        occ = r["avg_occupancy"]
        colour = BAD if occ > 0.85 else (GOOD if occ >= 0.70 else ACCENT)
        status = ("Above target" if occ > 0.85
                  else ("On target" if occ >= 0.70 else "Below target"))
        popup = f"""
        <div style="font-family:'Source Sans 3',Arial,sans-serif;min-width:230px">
          <div style="font-size:13px;font-weight:700;color:{colour};
                      border-bottom:1px solid {LINE};padding-bottom:6px;margin-bottom:8px">
            {r['blockfacename']}
          </div>
          <table style="width:100%;font-size:12px;border-collapse:collapse">
            <tr><td style="color:{MUTED}">Area</td>
                <td style="text-align:right;font-weight:700">{r['paidparkingarea']}</td></tr>
            <tr><td style="color:{MUTED}">Avg occupancy</td>
                <td style="text-align:right;font-weight:700;color:{colour}">{occ:.0%}</td></tr>
            <tr><td style="color:{MUTED}">Peak hour</td>
                <td style="text-align:right;font-weight:700">{fmt_hour(r['peak_hour'])}</td></tr>
            <tr><td style="color:{MUTED}">Spaces</td>
                <td style="text-align:right;font-weight:700">{int(r['spaces'])}</td></tr>
            <tr><td style="color:{MUTED}">Rate change</td>
                <td style="text-align:right;font-weight:700;color:{colour}">
                    {adjustment_text(r['mean_adjustment'])}</td></tr>
          </table>
          <div style="margin-top:8px;padding:4px 8px;background:{colour}14;
                      border-radius:4px;font-size:11px;color:{colour};
                      font-weight:700;text-align:center">{status}</div>
        </div>"""

        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=4 + min(float(r["spaces"]), 40) / 6,
            color=colour, fill=True, fill_color=colour, fill_opacity=0.75,
            weight=1.5, opacity=0.9,
            popup=folium.Popup(popup, max_width=280),
            tooltip=f"{r['blockfacename']} — {occ:.0%}",
        ).add_to(cluster)

    st_folium(m, use_container_width=True, height=600, returned_objects=[])

    section("Legend")
    l1, l2, l3 = st.columns(3)
    for col, (lbl, colour, desc) in zip(
        [l1, l2, l3],
        [("Below target", ACCENT, "under 70% — lower the rate"),
         ("On target", GOOD, "70–85% — no change"),
         ("Above target", BAD, "over 85% — raise the rate")],
    ):
        with col:
            st.markdown(
                f'<div class="region-cell" style="--rc:{colour}">'
                f'<div class="region-name" style="color:{colour}">{lbl}</div>'
                f'<div class="region-sub">{desc}</div></div>',
                unsafe_allow_html=True)
