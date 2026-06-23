import sys
import json
import traceback
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
LOG_FILE = ROOT / "logs" / "pipeline.log"

sys.path.insert(0, str(ROOT / "scripts"))


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_step(name: str, module_name: str) -> bool:
    log(f"▶ {name}")
    try:
        import importlib
        mod = importlib.import_module(module_name)
        mod.run()
        log(f"  ✓ {name} done")
        return True
    except Exception as e:
        log(f"  ✗ {name} failed: {e}")
        traceback.print_exc()
        return False


def main():
    log("=" * 60)
    log("Seattle Parking Intelligence — Pipeline Start")
    log("=" * 60)

    results = {}

    # Step 1 — Fetch all data sources
    results["weather"]        = run_step("Fetch weather",           "weather_fetch")
    results["weather_process"]= run_step("Process weather data",    "weather_eda")
    results["parking"]        = run_step("Fetch live parking",      "fetch_parking")
    results["events"]         = run_step("Fetch events",            "fetch_events")
    results["permits"]        = run_step("Fetch city permits",      "fetch_permits")
    results["closures"]       = run_step("Fetch road closures",     "fetch_closures")
    results["holidays"]       = run_step("Build holiday calendar",  "holidays_features")

    # Step 2 — Aggregate raw data into features
    results["aggregate"]      = run_step("Aggregate features",      "aggregate_features")

    # Step 3 — Train model on latest feature window
    results["train"]          = run_step("Train model",             "train_model")

    # Step 4 — Revenue intelligence
    results["pricing"]        = run_step("Pricing recommendations", "pricing_optimizer")
    results["revenue"]        = run_step("Revenue analysis",        "revenue_analyzer")
    results["roi"]            = run_step("Infrastructure ROI",      "infrastructure_roi")

    # Summary
    log("-" * 60)
    passed = sum(v for v in results.values())
    total  = len(results)
    log(f"Pipeline complete: {passed}/{total} steps succeeded")

    if passed < total:
        failed = [k for k, v in results.items() if not v]
        log(f"Failed steps: {', '.join(failed)}")
        sys.exit(1)

    log("All steps complete.")


if __name__ == "__main__":
    main()
