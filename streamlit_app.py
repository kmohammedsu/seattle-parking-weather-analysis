import os
from pathlib import Path

_app = Path(__file__).resolve().parent / "dashboard" / "app.py"

with open(_app) as _f:
    exec(compile(_f.read(), str(_app), "exec"), {"__file__": str(_app), "__name__": "__main__"})
