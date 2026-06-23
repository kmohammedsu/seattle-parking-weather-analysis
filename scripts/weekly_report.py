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

FEATURES_FILE = DATA_DIR / "features.csv"
REVENUE_FILE = DATA_DIR / "revenue_summary.csv"
PRICING_FILE = DATA_DIR / "pricing_recommendations.csv"
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
    if FEATURES_FILE.exists():
        features = pd.read_csv(FEATURES_FILE, parse_dates=["hour"])
        week_data = load_week(features)

        if not week_data.empty:
            avg_occ = week_data["avg_occupancy_rate"].mean()
            peak_occ = week_data["avg_occupancy_rate"].max()
            peak_hour_row = week_data.loc[week_data["avg_occupancy_rate"].idxmax()]
            busiest_region = week_data.groupby("region")["avg_occupancy_rate"].mean().idxmax()
            avg_by_region = week_data.groupby("region")["avg_occupancy_rate"].mean().reset_index()

            lines += [
                f"- **Average occupancy:** {avg_occ:.1%}",
                f"- **Peak occupancy:** {peak_occ:.1%} ({peak_hour_row['region']} at {peak_hour_row['hour'].strftime('%a %H:%M')})",
                f"- **Busiest region:** {busiest_region}",
                "",
                "| Region | Avg Occupancy |",
                "|--------|--------------|",
            ]
            for _, r in avg_by_region.sort_values("avg_occupancy_rate", ascending=False).iterrows():
                status = "🔴" if r["avg_occupancy_rate"] > 0.85 else ("🟡" if r["avg_occupancy_rate"] > 0.70 else "🟢")
                lines.append(f"| {r['region']} | {status} {r['avg_occupancy_rate']:.1%} |")
            lines.append("")
        else:
            lines.append("_No data for the past 7 days._\n")
    else:
        lines.append("_features.csv not found._\n")

    # ── Revenue summary ──────────────────────────────────────────────────────
    lines.append("## Revenue")
    if REVENUE_FILE.exists():
        revenue = pd.read_csv(REVENUE_FILE, parse_dates=["date"])
        week_rev = revenue[revenue["date"] >= revenue["date"].max() - pd.Timedelta(days=7)]

        if not week_rev.empty:
            total_current = week_rev["current_revenue"].sum()
            total_optimized = week_rev["optimized_revenue"].sum()
            uplift = total_optimized - total_current
            uplift_pct = uplift / max(total_current, 0.01) * 100

            lines += [
                f"- **Current revenue (7d):** ${total_current:,.0f}",
                f"- **Optimized revenue (7d):** ${total_optimized:,.0f}",
                f"- **Potential uplift:** ${uplift:+,.0f} ({uplift_pct:+.1f}%)",
                "",
            ]
        else:
            lines.append("_No revenue data for the past 7 days._\n")
    else:
        lines.append("_revenue_summary.csv not found._\n")

    # ── Pricing recommendations ──────────────────────────────────────────────
    lines.append("## Pricing Recommendations")
    if PRICING_FILE.exists():
        pricing = pd.read_csv(PRICING_FILE)
        increases = (pricing["action"] == "increase").sum()
        decreases = (pricing["action"] == "decrease").sum()
        holds = (pricing["action"] == "hold").sum()

        lines += [
            f"- Rate increases recommended: **{increases}** zone-hours",
            f"- Rate decreases recommended: **{decreases}** zone-hours",
            f"- No change: **{holds}** zone-hours",
            "",
        ]

        # Top increases
        top_inc = pricing[pricing["action"] == "increase"].nlargest(3, "revenue_delta")
        if not top_inc.empty:
            lines.append("**Top increase opportunities:**")
            for _, r in top_inc.iterrows():
                lines.append(f"- {r['region']} at hour {int(r['hour_of_day'])}: "
                              f"${r['current_rate']:.2f} → ${r['recommended_rate']:.2f} "
                              f"(+${r['revenue_delta']:.2f}/hr)")
            lines.append("")
    else:
        lines.append("_pricing_recommendations.csv not found._\n")

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
        viable = roi[roi["viable"]].nlargest(3, "roi_percent")
        if not viable.empty:
            lines.append("Top viable investments this week:")
            for _, r in viable.iterrows():
                lines.append(f"- **{r['region']}** — {r['n_new_spaces']} space "
                              f"{r['infra_type'].replace('_',' ')}: "
                              f"ROI {r['roi_percent']:.1f}%, "
                              f"payback {r['payback_years']:.1f} yrs")
        else:
            lines.append("_No viable infrastructure investments identified._")
        lines.append("")

    # ── Footer ───────────────────────────────────────────────────────────────
    lines += [
        "---",
        "*Generated automatically by Seattle Parking Intelligence Platform. "
        "Data: Seattle Open Data Portal + Open-Meteo + Ticketmaster. "
        "Pricing authority: SMC 11.16.121.*",
    ]

    report_content = "\n".join(lines)
    report_file.write_text(report_content)
    print(f"Weekly report saved → {report_file}")

    # Also write a "latest_report.md" symlink-style copy for easy access
    (REPORTS_DIR / "latest_weekly_report.md").write_text(report_content)
    return report_file


if __name__ == "__main__":
    run()
