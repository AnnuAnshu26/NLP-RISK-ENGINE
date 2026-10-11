"""Smoke test for the Streamlit dashboard (no browser needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "src" / "app.py")


def test_dashboard_renders_and_example_button_works():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    at.button(key="ex_🕊️ Ceasefire").click().run()
    assert not at.exception
    assert "Ceasefire" in at.session_state["headline"]