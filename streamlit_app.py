# Streamlit Cloud entry point
# Streamlit Cloud looks for streamlit_app.py at the repo root by default.
# This file re-exports the dashboard so we don't duplicate code.
exec(open("dashboard/app.py").read())
