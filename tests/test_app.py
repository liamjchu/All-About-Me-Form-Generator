"""Streamlit app smoke tests via AppTest (real Streamlit runtime)."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from pdf_filler import fill_form_pdf, merge_profiles_pdf


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_loads_and_shows_title() -> None:
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.run()
    assert not at.exception
    assert any("All About Me Profile Generator" in title.value for title in at.title)


def test_generate_without_uploads_warns() -> None:
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.run()
    at.button[0].click().run()
    assert at.warning
    assert "Upload at least one" in at.warning[0].value


def test_app_shows_generated_profiles_from_session_state() -> None:
    pdf_bytes = fill_form_pdf(
        {
            "name": "Riley",
            "favorite_things_1": "books",
            "allergies": "N/A",
            "bathroom_needs": "N/A",
        }
    )
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.session_state["generated_profiles"] = [
        {
            "stem": "riley",
            "markdown": "# All About Me\n\n**Name:** Riley\n",
            "pdf_bytes": pdf_bytes,
        }
    ]
    at.run()
    assert not at.exception
    assert at.success
    assert "Created 1 profile" in at.success[0].value
    assert at.expander
    assert any(expander.label == "riley" for expander in at.expander)


def test_app_wipe_profiles_clears_session() -> None:
    pdf_bytes = fill_form_pdf({"name": "Riley", "allergies": "N/A"})
    at = AppTest.from_file(str(APP_PATH), default_timeout=10)
    at.session_state["generated_profiles"] = [
        {
            "stem": "riley",
            "markdown": "# All About Me\n\n**Name:** Riley\n",
            "pdf_bytes": pdf_bytes,
        }
    ]
    at.run()
    wipe = next(button for button in at.button if "Wipe profiles" in button.label)
    wipe.click().run()
    assert not at.exception
    assert at.session_state["generated_profiles"] == []


def test_merge_profiles_used_by_app_is_valid_pdf() -> None:
    """Guard the download path the app uses after generation."""
    one = fill_form_pdf({"name": "One"})
    two = fill_form_pdf({"name": "Two"})
    merged = merge_profiles_pdf(
        [
            {"stem": "one", "markdown": "a", "pdf_bytes": one},
            {"stem": "two", "markdown": "b", "pdf_bytes": two},
        ]
    )
    assert merged.startswith(b"%PDF")
