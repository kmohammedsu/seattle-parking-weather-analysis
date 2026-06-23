# Streamlit Cloud entry point — delegates to dashboard/app.py
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
runpy.run_path(str(Path(__file__).parent / "dashboard" / "app.py"), run_name="__main__")
