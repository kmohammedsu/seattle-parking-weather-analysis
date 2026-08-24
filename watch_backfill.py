"""
Run this in a separate terminal to watch backfill progress live.
    python watch_backfill.py
Press Ctrl+C to stop.
"""
import json, subprocess, time, os
from pathlib import Path

PROGRESS_FILE = Path(__file__).parent / "data" / "backfill_progress.json"
ALL_MONTHS = [f"{y}-{m:02d}" for y in range(2018, 2026) for m in range(1, 13) if f"{y}-{m:02d}" <= "2025-12"]
TOTAL = len(ALL_MONTHS)

def is_running():
    r = subprocess.run(["pgrep", "-f", "backfill_parking"], capture_output=True)
    return r.returncode == 0

def read_progress():
    if not PROGRESS_FILE.exists():
        return set()
    return set(json.loads(PROGRESS_FILE.read_text()).get("completed_months", []))

def render(done, prev_count):
    n = len(done)
    pct = n / TOTAL * 100
    filled = int(pct / 2)
    bar = f"[{'█' * filled}{'░' * (50 - filled)}]"
    running = is_running()
    status = "● RUNNING" if running else "■ STOPPED"

    lines = []
    lines.append(f"  Seattle Parking Backfill  —  {status}")
    lines.append(f"  {bar}  {pct:.0f}%")
    lines.append(f"  {n} / {TOTAL} months complete  ·  {TOTAL - n} remaining")
    lines.append(f"  Latest: {max(done) if done else '—'}")
    lines.append("")
    lines.append("      Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec")
    for y in range(2018, 2026):
        cells = ""
        for m in range(1, 13):
            key = f"{y}-{m:02d}"
            cells += " ■ " if key in done else " □ "
        lines.append(f"  {y}  {cells}")
    lines.append("")
    if not running:
        lines.append("  Backfill finished! Run: python scripts/rebuild_after_backfill.py")
    else:
        new = n - prev_count
        lines.append(f"  +{new} month(s) since last check  ·  refreshing every 15s  ·  Ctrl+C to exit")

    os.system("clear")
    print("\n".join(lines))
    return n, running

prev = 0
while True:
    done = read_progress()
    prev, still_running = render(done, prev)
    if not still_running:
        break
    time.sleep(15)
