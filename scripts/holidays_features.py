import holidays
import pandas as pd
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "seattle_holidays.csv"

# Seattle-specific observances on top of US federal holidays
SEATTLE_LOCAL_EVENTS = {
    # Seafair (annually, late July – early August)
    "Seafair Weekend":      [(7, 28), (7, 29), (7, 30), (7, 31), (8, 1)],
    # Pride Parade (last Sunday of June)
    "Seattle Pride":        [(6, 25), (6, 26)],
    # Seahawks home opener (approximate — updated via events fetcher)
}

START_YEAR = 2012
END_YEAR = date.today().year + 1


def build_holiday_df() -> pd.DataFrame:
    us_holidays = holidays.US(state="WA", years=range(START_YEAR, END_YEAR + 1))

    records = []
    for dt, name in sorted(us_holidays.items()):
        records.append({
            "date": dt,
            "holiday_name": name,
            "holiday_type": "federal",
            "is_holiday": True,
        })

    # Add Seattle local events
    for year in range(START_YEAR, END_YEAR + 1):
        for event_name, month_days in SEATTLE_LOCAL_EVENTS.items():
            for month, day in month_days:
                try:
                    records.append({
                        "date": date(year, month, day),
                        "holiday_name": event_name,
                        "holiday_type": "local",
                        "is_holiday": True,
                    })
                except ValueError:
                    pass  # invalid date (e.g. Feb 30)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date", "holiday_name"]).sort_values("date")
    return df


def is_holiday(dt: pd.Timestamp, holiday_df: pd.DataFrame) -> tuple[bool, str]:
    match = holiday_df[holiday_df["date"].dt.date == dt.date()]
    if match.empty:
        return False, ""
    return True, match.iloc[0]["holiday_name"]


def run():
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Building holiday calendar {START_YEAR}–{END_YEAR}...")
    df = build_holiday_df()
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} holiday records to {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    run()
