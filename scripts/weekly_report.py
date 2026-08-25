"""
Weekly parking intelligence summary report.
Generates a markdown report summarizing the past 7 days of data.
Run by GitHub Actions every Monday morning (or manually).

Output: reports/weekly_report_YYYY-WW.md
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

AREA_DAILY_FILE = DATA_DIR / "area_daily.csv"
METERS_FILE = DATA_DIR / "meter_summary.csv"
REVENUE_FILE = DATA_DIR / "revenue_summary.csv"
PRICING_FILE = DATA_DIR / "meter_recommendations.csv"
ROI_FILE = DATA_DIR / "infrastructure_roi.csv"
PERF_FILE = ROOT / "models" / "performance" / "final_model_performance.csv"


def load_week(df: pd.DataFrame, date_col: str = "hour", days: int = 7) -> pd.DataFrame:
    cutoff = df[date_col].max() - pd.Timedelta(days=days)
    return df[df[date_col] >= cutoff]


def run():
    REPORTS_DIR.mkdir(exist_ok=True)

    now = datetime.utcnow()
    week_str = now.strftime("%Y-W%W")
    report_file = REPORTS_DIR / f"weekly_report_{week_str}.md"

    lines = [
        f"# Seattle Parking Intelligence — Weekly Report",
        f"**Period:** {(now - timedelta(days=7)).strftime('%B %d')} – {now.strftime('%B %d, %Y')}",
        f"**Generated:** {now.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    # ── Occupancy summary ────────────────────────────────────────────────────
    lines.append("## Occupancy")
    if AREA_DAILY_FILE.exists():
        area_day = pd.read_csv(AREA_DAILY_FILE, parse_dates=["date"])
        cutoff = area_day["date"].max() - timedelta(days=7)
        week = area_day[area_day["date"] >= cutoff]

        if not week.empty:
            avg_occ = week["avg_occupancy_rate"].mean()
            by_area = (week.groupby("paidparkingarea")["avg_occupancy_rate"]
                       .mean().reset_index()
                       .sort_values("avg_occupancy_rate", ascending=False))
            busiest = by_area.iloc[0]

            lines += [
                f"- **Average occupancy:** {avg_occ:.1%}",
                f"- **Busiest area:** {busiest['paidparkingarea']} "
                f"({busiest['avg_occupancy_rate']:.1%})",
                f"- **Areas tracked:** {by_area['paidparkingarea'].nunique()} "
                f"(Seattle official paid parking areas)",
                "",
                "| Area | Avg Occupancy | Status |",
                "|------|--------------|--------|",
            ]
            for _, r in by_area.iterrows():
                occ = r["avg_occupancy_rate"]
                status = ("Above target" if occ > 0.85
                          else ("On target" if occ >= 0.70 else "Below target"))
                lines.append(f"| {r['paidparkingarea']} | {occ:.1%} | {status} |")
            lines.append("")
        else:
            lines.append("_No data for the past 7 days._\n")
    else:
        lines.append("_area_daily.csv not found._\n")

    # ── Revenue summary ──────────────────────────────────────────────────────
    lines.append("## Utilization")
    if REVENUE_FILE.exists():
        revenue = pd.read_csv(REVENUE_FILE, parse_dates=["date"])
        week_rev = revenue[revenue["date"] >= revenue["date"].max() - pd.Timedelta(days=7)]

        if not week_rev.empty:
            occupied = week_rev["occupied_space_hours"].sum()
            target = week_rev["target_space_hours"].sum()
            unsold = week_rev["unsold_space_hours"].sum()
            util_pct = week_rev["utilization_pct"].mean()

            lines += [
                f"- **Occupied (7d):** {occupied:,.0f} space-hours",
                f"- **At 80% target:** {target:,.0f} space-hours",
                f"- **Unsold vs target:** {unsold:,.0f} space-hours",
                f"- **Utilization:** {util_pct:.1f}% of capacity",
                "",
                "_Revenue equals posted rate x occupied space-hours. Seattle publishes "
                "no per-meter rate, so utilization is reported in space-hours; multiply "
                f"by the real posted rate for dollars (e.g. at $2.00/hr the unsold gap "
                f"is worth ~${unsold * 2:,.0f})._",
                "",
            ]
        else:
            lines.append("_No utilization data for the past 7 days._\n")
    else:
        lines.append("_revenue_summary.csv not found._\n")

    # ── Pricing recommendations ──────────────────────────────────────────────
    lines.append("## Pricing Recommendations")
    if PRICING_FILE.exists() and METERS_FILE.exists():
        pricing = pd.read_csv(PRICING_FILE)
        meters = pd.read_csv(METERS_FILE)
        counts = meters["primary_action"].value_counts()

        lines += [
            f"- Blockfaces to **lower**: **{int(counts.get('decrease', 0)):,}** (under 70% occupancy)",
            f"- Blockfaces to **raise**: **{int(counts.get('increase', 0)):,}** (over 85%)",
            f"- Blockfaces to **hold**: **{int(counts.get('hold', 0)):,}** (in target band)",
            "",
            "_Rate changes are adjustments to the currently posted rate. Seattle "
            "publishes no per-meter rate data, so no absolute dollar rate is implied._",
            "",
        ]

        top_inc = meters[meters["primary_action"] == "increase"].nlargest(3, "avg_occupancy")
        if not top_inc.empty:
            lines.append("**Blockfaces most over target:**")
            for _, r in top_inc.iterrows():
                lines.append(f"- {r['blockfacename']} ({r['paidparkingarea']}): "
                              f"{r['avg_occupancy']:.0%} occupancy, "
                              f"recommend {r['mean_adjustment']:+.2f}/hr")
            lines.append("")
    else:
        lines.append("_meter_recommendations.csv not found._\n")

    # ── Model performance ────────────────────────────────────────────────────
    lines.append("## Model Performance")
    if PERF_FILE.exists():
        perf = pd.read_csv(PERF_FILE).iloc[0]
        lines += [
            f"- R²: **{perf['r2']:.4f}**",
            f"- RMSE: **{perf['rmse']:.4f}**",
            f"- MAE: **{perf['mae']:.4f}**",
            f"- Training samples: **{int(perf['n_train']):,}**",
            "",
        ]
    else:
        lines.append("_Model performance data not found._\n")

    # ── Infrastructure highlights ────────────────────────────────────────────
    lines.append("## Infrastructure Opportunities")
    if ROI_FILE.exists():
        roi = pd.read_csv(ROI_FILE)
        viable = roi[roi["viable"]]
        if not viable.empty:
            lines.append("Top viable investments this week:")
            for _, r in viable.nsmallest(3, "breakeven_rate_per_hour").iterrows():
                lines.append(f"- **{r['paidparkingarea']}** — {int(r['n_new_spaces'])} space "
                              f"{r['infra_type'].replace('_',' ')}: "
                              f"breakeven rate ${r['breakeven_rate_per_hour']:.2f}/hr")
        else:
            lines.append("No expansion is justified — no area currently reaches the "
                         "80% utilization target, so added capacity would sit unused.")
        lines.append("")
    else:
        lines.append("_infrastructure_roi.csv not found._\n")

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"weekly_report_{now:%Y_W%W}.md"
    out.write_text("\n".join(lines))
    print(f"Weekly report saved -> {out}")
    return out


if __name__ == "__main__":
    run()
