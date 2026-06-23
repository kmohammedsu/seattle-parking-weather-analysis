# Streamlit Cloud entry point — sets __file__ correctly before exec so that
# Path(__file__).resolve().parent.parent resolves to the repo root inside app.py
import os
_app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "app.py")
with open(_app) as _f:
    exec(compile(_f.read(), _app, "exec"))
