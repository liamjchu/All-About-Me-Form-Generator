"""Tests for PDF form filling and field normalization (uses real template PDF)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from pypdf import PdfReader

import pdf_filler
from pdf_filler import (
    empty_form_data,
    fill_form_pdf,
    merge_profiles_pdf,
    normalize_form_data,
    _map_box,
    _map_x,
    _na_if_blank_or_none,
    _sanitize_bathroom_needs,
    _template_pdf,
    _wrap_text,
)
from PIL import Image, ImageDraw


def test_template_pdf_resolves_project_file() -> None:
    path = _template_pdf()
    assert path.is_file()
    assert path.name in {"formTemplate.pdf", "All About Me Template.pdf"}


def test_template_pdf_missing_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pdf_filler, "PROJECT_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="PDF template not found"):
        _template_pdf()


def test_empty_form_data_has_all_keys() -> None:
    data = empty_form_data()
    assert data["name"] == ""
    assert data["favorite_things_3"] == ""
    assert data["reinforcers_4"] == ""
    assert data["behavioral_management"] == ""
    assert len(data) == 14


def test_normalize_form_data_from_lists() -> None:
    normalized = normalize_form_data(
        {
            "name": "  Avery  ",
            "favorite_things": ["art", "reading", "puzzles", "ignored"],
            "favorite_reinforcers": ["hugs", "music"],
            "parent_name": "Jamie",
            "parent_phone": "555-0111",
            "parent_email": "jamie@example.com",
            "allergies": "none",
            "bathroom_needs": "independent walker",
            "behavioral_management": "Take breaks.",
        }
    )
    assert normalized["name"] == "Avery"
    assert normalized["favorite_things_1"] == "art"
    assert normalized["favorite_things_3"] == "puzzles"
    assert normalized["reinforcers_1"] == "hugs"
    assert normalized["reinforcers_2"] == "music"
    assert normalized["reinforcers_3"] == ""
    assert normalized["allergies"] == "N/A"
    assert normalized["bathroom_needs"] == "N/A"
    assert normalized["behavioral_management"] == "Take breaks."


def test_normalize_form_data_flat_keys_and_reinforcers_alias() -> None:
    normalized = normalize_form_data(
        {
            "name": "Blake",
            "favorite_things_2": "drums",
            "reinforcers": ["token", "break", "snack", "park"],
            "allergies": "Dairy",
            "bathroom_needs": "Needs help with buttons",
        }
    )
    assert normalized["favorite_things_2"] == "drums"
    assert normalized["reinforcers_4"] == "park"
    assert normalized["allergies"] == "Dairy"
    assert "buttons" in normalized["bathroom_needs"]


def test_normalize_form_data_non_dict_returns_empty() -> None:
    assert normalize_form_data("not a dict") == empty_form_data()  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", "N/A"),
        ("   ", "N/A"),
        ("n/a", "N/A"),
        ("None", "N/A"),
        ("no.", "N/A"),
        ("not applicable", "N/A"),
        ("Peanut butter", "Peanut butter"),
    ],
)
def test_na_if_blank_or_none(raw: str, expected: str) -> None:
    assert _na_if_blank_or_none(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ""),
        ("independent walker", "N/A"),
        ("Independent Walker", "N/A"),
        ("no help needed", "N/A"),
        ("does not need help", "N/A"),
        ("independently in the restroom", "N/A"),
        ("Needs help with zippers", "Needs help with zippers"),
        ("Uses an independent walker at school", "Uses an independent at school"),
    ],
)
def test_sanitize_bathroom_needs(raw: str, expected: str) -> None:
    assert _sanitize_bathroom_needs(raw) == expected


def test_map_box_and_map_x_scale_from_preview() -> None:
    mapped = _map_box((100, 200, 300, 400), (2380, 3082))
    assert mapped[0] == int(100 * 2380 / 1190)
    assert mapped[1] == int(200 * 3082 / 1541)
    assert _map_x(119, (2380, 3082)) == int(119 * 2380 / 1190)


def test_wrap_text_splits_long_lines() -> None:
    image = Image.new("RGB", (400, 100), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = pdf_filler._load_font(24)
    lines = _wrap_text(draw, "one two three four five six seven", font, max_width=80)
    assert len(lines) > 1
    assert " ".join(lines).startswith("one")


def test_wrap_text_empty_returns_blank_line() -> None:
    image = Image.new("RGB", (100, 40), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = pdf_filler._load_font(20)
    assert _wrap_text(draw, "   ", font, max_width=50) == [""]


def test_fill_form_pdf_returns_two_page_pdf(sample_form_data: dict[str, str]) -> None:
    pdf_bytes = fill_form_pdf(sample_form_data)
    assert pdf_bytes.startswith(b"%PDF")
    reader = PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) == 2
    assert float(reader.pages[0].mediabox.width) == 612.0


def test_fill_form_pdf_handles_long_multiline_and_empty_rows() -> None:
    long_body = (
        "Avoid bright lights and loud rooms when possible. "
        "Offer a quiet corner with preferred fidgets and a visual timer. "
        "Staff should narrate transitions early and keep instructions short."
    )
    pdf_bytes = fill_form_pdf(
        {
            "name": "A Very Long Participant Name That Needs Shrinking",
            "favorite_things_1": "extraordinarily-long-single-token-interest-name",
            "allergies": long_body,
            "bathroom_needs": long_body,
            "behavioral_management": long_body,
            "parent_email": "very.long.email.address.for.shrinking@example.com",
        }
    )
    assert len(PdfReader(io.BytesIO(pdf_bytes)).pages) == 2


def test_merge_profiles_pdf_concatenates_pages(sample_form_data: dict[str, str]) -> None:
    first = fill_form_pdf(sample_form_data)
    second = fill_form_pdf({**sample_form_data, "name": "Second Child"})
    merged = merge_profiles_pdf(
        [
            {"stem": "first", "markdown": "# a", "pdf_bytes": first},
            {"stem": "second", "markdown": "# b", "pdf_bytes": second},
        ]
    )
    assert len(PdfReader(io.BytesIO(merged)).pages) == 4


def test_load_font_falls_back_when_candidates_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pdf_filler, "_FONT_CANDIDATES", ("/nonexistent/font.ttf",))
    with pytest.warns(UserWarning, match="No preferred form font"):
        font = pdf_filler._load_font(32)
    assert font is not None
