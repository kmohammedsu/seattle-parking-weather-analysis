"""
Upload the trained model to Hugging Face Hub so Streamlit Cloud can load it
without needing to re-train on every cold start.

Usage:
    export HUGGINGFACE_TOKEN=hf_...
    python scripts/upload_model.py

The dashboard's load_model() will try local path first, fall back to HF Hub.
Set HF_REPO in environment to override the default repo name.
"""
import os
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "parking_demand_lgbm.pkl"
FEATURE_LIST_FILE = ROOT / "models" / "feature_columns.json"
PERF_FILE = ROOT / "models" / "performance" / "final_model_performance.csv"

HF_REPO = os.getenv("HF_REPO", "kmohammedsu/seattle-parking-model")


def run():
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("HUGGINGFACE_TOKEN not set. Skipping HF Hub upload.")
        return

    if not MODEL_FILE.exists():
        print(f"Model not found at {MODEL_FILE}. Run train_model.py first.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        return

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=HF_REPO, repo_type="model", exist_ok=True)
        print(f"Repo: https://huggingface.co/{HF_REPO}")
    except Exception as e:
        print(f"Could not create repo: {e}")
        return

    files_to_upload = [
        (MODEL_FILE, "parking_demand_lgbm.pkl"),
        (FEATURE_LIST_FILE, "feature_columns.json"),
    ]
    if PERF_FILE.exists():
        files_to_upload.append((PERF_FILE, "final_model_performance.csv"))

    for local_path, repo_filename in files_to_upload:
        if local_path.exists():
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_filename,
                repo_id=HF_REPO,
                repo_type="model",
            )
            print(f"  Uploaded: {repo_filename}")
        else:
            print(f"  Skipped (not found): {local_path.name}")

    print(f"\nModel available at: https://huggingface.co/{HF_REPO}")


def download_model(local_dir: Path = ROOT / "models"):
    """Download model from HF Hub to local models/ directory."""
    token = os.getenv("HUGGINGFACE_TOKEN")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False

    files = ["parking_demand_lgbm.pkl", "feature_columns.json"]
    local_dir.mkdir(exist_ok=True)

    for filename in files:
        try:
            path = hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                local_dir=str(local_dir),
                token=token,
            )
            print(f"  Downloaded: {filename} → {path}")
        except Exception as e:
            print(f"  Could not download {filename}: {e}")
            return False
    return True


if __name__ == "__main__":
    run()
