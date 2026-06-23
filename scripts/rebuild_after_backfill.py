"""
Run this ONCE after backfill_parking.py completes to rebuild the full feature
matrix and train the model on all historical data.

Usage:
    python scripts/rebuild_after_backfill.py

This is a one-time setup step. After this, the daily pipeline (run_pipeline.py)
handles incremental updates automatically.
"""
import sys
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

PROGRESS_FILE = ROOT / "data" / "backfill_progress.json"
FEATURES_FILE = ROOT / "data" / "features.csv"


def check_backfill_complete() -> bool:
    if not PROGRESS_FILE.exists():
        return False
    data = json.loads(PROGRESS_FILE.read_text())
    completed = set(data.get("completed_months", []))
    # Expect at least 2018-01 through 2025-12 (minus any gaps)
    expected = {f"{y}-{m:02d}" for y in range(2018, 2026) for m in range(1, 13)
                if f"{y}-{m:02d}" <= "2026-06"}  # up to current month
    coverage = len(completed & expected) / len(expected) * 100
    print(f"Backfill coverage: {len(completed)}/{len(expected)} months ({coverage:.0f}%)")
    return len(completed) > 50  # at least 50% complete before rebuilding


def run():
    print("=" * 60)
    print("Post-backfill rebuild")
    print("=" * 60)

    if not check_backfill_complete():
        print("\nBackfill not complete enough yet.")
        print("Check progress: cat data/backfill_progress.json | python3 -c \"import json,sys; d=json.load(sys.stdin); print(len(d['completed_months']), 'months done')\"")
        print("Run more: python scripts/backfill_parking.py")
        return

    # Step 1: Delete existing features.csv to force full rebuild
    if FEATURES_FILE.exists():
        rows_before = sum(1 for _ in open(FEATURES_FILE)) - 1
        print(f"\nExisting features.csv: {rows_before:,} rows (will rebuild from full history)")
        FEATURES_FILE.unlink()

    # Step 2: Rebuild features
    print("\n[1/3] Rebuilding features from all historical data...")
    import aggregate_features
    features = aggregate_features.run()
    if features is None or features.empty:
        print("FAILED: aggregate_features returned nothing")
        return
    print(f"Features built: {len(features):,} rows")

    # Step 3: Train model on full history
    print("\n[2/3] Training model on full historical feature set...")
    import train_model
    metrics = train_model.run()
    if metrics:
        print(f"Model trained — R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
        print(f"Training samples: {metrics['n_train']:,}")

    # Step 4: Generate revenue analysis
    print("\n[3/3] Running pricing and revenue analysis...")
    import pricing_optimizer
    import revenue_analyzer
    pricing_optimizer.run()
    revenue_analyzer.run()

    print("\n" + "=" * 60)
    print("Rebuild complete.")
    print("Next steps:")
    print("  1. git add data/features.csv data/revenue_summary.csv data/pricing_recommendations.csv")
    print("  2. git add data/revenue_by_region_hour.csv models/performance/final_model_performance.csv")
    print("  3. git commit -m 'data: initial historical backfill complete'")
    print("  4. git push")
    print("  5. streamlit run streamlit_app.py  — to verify dashboard")
    print("=" * 60)


if __name__ == "__main__":
    run()
