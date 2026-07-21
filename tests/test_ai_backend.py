"""Tests for Ollama-backed profile generation.

Network calls hit a real local HTTP server fixture (not unittest.mock), except
where we intentionally force transport errors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ai_backend
from ai_backend import (
    extract_form_data,
    form_data_to_markdown,
    generate_all_about_me_pdf,
    generate_all_about_me_profile,
    _base_url,
    _env_or_dotenv,
    _parse_form_json,
    _text_model,
    _vision_model,
)
from file_inputs import FooterMark
from tests.helpers import FakeOllamaHandler, make_png_bytes


def test_env_or_dotenv_reads_process_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "  custom-model  ")
    assert _env_or_dotenv("OLLAMA_MODEL", "fallback") == "custom-model"


def test_env_or_dotenv_reads_dotenv_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OLLAMA_VISION_MODEL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n\nOLLAMA_VISION_MODEL='vision-from-file'\nOTHER=1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ai_backend, "PROJECT_ROOT", tmp_path)
    assert _env_or_dotenv("OLLAMA_VISION_MODEL") == "vision-from-file"


def test_env_or_dotenv_returns_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MISSING_SETTING", raising=False)
    monkeypatch.setattr(ai_backend, "PROJECT_ROOT", tmp_path)
    assert _env_or_dotenv("MISSING_SETTING", "default-value") == "default-value"


def test_base_url_strips_openai_v1_suffix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1/")
    monkeypatch.setattr(ai_backend, "PROJECT_ROOT", tmp_path)
    assert _base_url() == "http://localhost:11434"


def test_model_helpers_use_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_VISION_MODEL", raising=False)
    monkeypatch.setattr(ai_backend, "PROJECT_ROOT", tmp_path)
    assert _text_model() == ai_backend.DEFAULT_MODEL
    assert _vision_model() == ai_backend.DEFAULT_VISION_MODEL


def test_parse_form_json_plain_object() -> None:
    data = _parse_form_json(
        json.dumps(
            {
                "name": "Chris",
                "favorite_things": ["Lego"],
                "favorite_reinforcers": [],
                "allergies": "",
                "bathroom_needs": "no",
            }
        )
    )
    assert data["name"] == "Chris"
    assert data["favorite_things_1"] == "Lego"
    assert data["allergies"] == "N/A"
    assert data["bathroom_needs"] == "N/A"


def test_parse_form_json_strips_markdown_fence() -> None:
    fenced = """```json
{"name": "Dana", "favorite_things": ["yoga"], "allergies": "N/A", "bathroom_needs": "N/A"}
```"""
    assert _parse_form_json(fenced)["name"] == "Dana"


def test_parse_form_json_recovers_embedded_object() -> None:
    noisy = 'Sure!\n{"name": "Evan", "favorite_things": ["chess"], "allergies": "N/A", "bathroom_needs": "N/A"}\nThanks'
    assert _parse_form_json(noisy)["name"] == "Evan"


def test_parse_form_json_rejects_non_object() -> None:
    with pytest.raises(RuntimeError, match="not an object"):
        _parse_form_json("[1, 2, 3]")


def test_parse_form_json_rejects_unusable_text() -> None:
    with pytest.raises(RuntimeError, match="usable JSON"):
        _parse_form_json("no braces here")


def test_form_data_to_markdown_includes_sections(sample_form_data: dict[str, str]) -> None:
    markdown = form_data_to_markdown(sample_form_data)
    assert markdown.startswith("# All About Me")
    assert "**Name:** Jordan Lee" in markdown
    assert "- painting" in markdown
    assert "- praise" in markdown
    assert "taylor.lee@example.com" in markdown
    assert "Latex" in markdown
    assert "calm voice" in markdown


def test_form_data_to_markdown_empty_lists_show_placeholder() -> None:
    markdown = form_data_to_markdown({"name": "Only Name"})
    assert "## My Favorite Things\n- \n" in markdown
    assert "## Favorite Reinforcers\n- \n" in markdown
    assert "N/A" in markdown


def test_extract_form_data_requires_input() -> None:
    with pytest.raises(ValueError, match="Provide raw_text"):
        extract_form_data()


def test_extract_form_data_rejects_bad_image_mime() -> None:
    with pytest.raises(ValueError, match="image MIME"):
        extract_form_data(image_bytes=b"abc", image_mime_type="application/pdf")


def test_extract_form_data_text_via_local_server(
    monkeypatch: pytest.MonkeyPatch, ollama_server: str
) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", ollama_server)
    monkeypatch.setenv("OLLAMA_MODEL", "test-text")
    data = extract_form_data("Name: Alex Rivera\nFavorite interests: swimming")
    assert data["name"] == "Alex Rivera"
    assert data["favorite_things_1"] == "swimming"
    assert FakeOllamaHandler.last_payload is not None
    assert FakeOllamaHandler.last_payload["model"] == "test-text"
    assert FakeOllamaHandler.last_payload["format"] == "json"
    assert "images" not in FakeOllamaHandler.last_payload["messages"][1]


def test_extract_form_data_image_uses_vision_model(
    monkeypatch: pytest.MonkeyPatch, ollama_server: str
) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", ollama_server)
    monkeypatch.setenv("OLLAMA_VISION_MODEL", "test-vision")
    data = extract_form_data(image_bytes=make_png_bytes(), image_mime_type="image/png")
    assert data["name"] == "Alex Rivera"
    assert FakeOllamaHandler.last_payload is not None
    assert FakeOllamaHandler.last_payload["model"] == "test-vision"
    assert FakeOllamaHandler.last_payload["keep_alive"] == "30m"
    assert FakeOllamaHandler.last_payload["options"]["num_ctx"] == ai_backend.DEFAULT_NUM_CTX
    assert "images" in FakeOllamaHandler.last_payload["messages"][1]
    user_content = FakeOllamaHandler.last_payload["messages"][1]["content"]
    assert "photo" in user_content.lower()
    assert "handwritten" in user_content.lower()


def test_extract_footer_mark_uses_ocr_without_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai_backend import extract_footer_mark

    called = {"chat": False}

    def _fail_chat(*_args, **_kwargs):  # noqa: ANN001
        called["chat"] = True
        raise AssertionError("vision fallback should not run when OCR succeeds")

    monkeypatch.setattr(ai_backend, "_chat", _fail_chat)
    monkeypatch.setattr(
        ai_backend,
        "extract_footer_mark_ocr",
        lambda _bytes: FooterMark(
            date_line="Jun 30 2026 1:21PM ET",
            page=45,
            total=85,
        ),
    )
    mark = extract_footer_mark(make_png_bytes(), image_mime_type="image/png")
    assert mark.page == 45
    assert mark.total == 85
    assert called["chat"] is False


def test_default_vision_model_is_qwen() -> None:
    assert ai_backend.DEFAULT_VISION_MODEL == "qwen2.5vl:7b"


def test_extract_form_data_multiple_images(
    monkeypatch: pytest.MonkeyPatch, ollama_server: str
) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", ollama_server)
    monkeypatch.setenv("OLLAMA_VISION_MODEL", "test-vision")
    data = extract_form_data(
        images=[
            (make_png_bytes(color=(10, 20, 30)), "image/png"),
            (make_png_bytes(color=(40, 50, 60)), "image/png"),
        ]
    )
    assert data["name"] == "Alex Rivera"
    assert FakeOllamaHandler.last_payload is not None
    images = FakeOllamaHandler.last_payload["messages"][1]["images"]
    assert len(images) == 2


def test_extract_footer_mark_reads_page_and_date(
    monkeypatch: pytest.MonkeyPatch, ollama_server: str
) -> None:
    from ai_backend import extract_footer_mark

    FakeOllamaHandler.response_content = json.dumps(
        {
            "date_line": "Jun 30 2026 1:21PM ET",
            "page": 45,
            "total": 85,
        }
    )
    monkeypatch.setenv("OLLAMA_BASE_URL", ollama_server)
    monkeypatch.setenv("OLLAMA_VISION_MODEL", "test-vision")
    mark = extract_footer_mark(make_png_bytes(), image_mime_type="image/png")
    assert mark.date_line == "Jun 30 2026 1:21PM ET"
    assert mark.page == 45
    assert mark.total == 85


def test_extract_form_data_rejects_empty_model_content(
    monkeypatch: pytest.MonkeyPatch, ollama_server: str
) -> None:
    FakeOllamaHandler.response_content = "   "
    monkeypatch.setenv("OLLAMA_BASE_URL", ollama_server)
    with pytest.raises(RuntimeError, match="empty response"):
        extract_form_data("Name: Someone")


def test_extract_form_data_rejects_empty_participant_payload(
    monkeypatch: pytest.MonkeyPatch, ollama_server: str
) -> None:
    FakeOllamaHandler.response_content = json.dumps(
        {
            "name": "",
            "favorite_things": ["", "", ""],
            "favorite_reinforcers": ["", "", "", ""],
            "allergies": "N/A",
            "bathroom_needs": "N/A",
            "behavioral_management": "",
        }
    )
    monkeypatch.setenv("OLLAMA_BASE_URL", ollama_server)
    with pytest.raises(RuntimeError, match="usable participant details"):
        extract_form_data("garbage input with no facts")


def test_extract_form_data_unreachable_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    with pytest.raises(RuntimeError, match="Could not reach Ollama"):
        extract_form_data("Name: Nobody")


def test_generate_all_about_me_profile_returns_markdown_and_pdf(
    monkeypatch: pytest.MonkeyPatch, ollama_server: str
) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", ollama_server)
    markdown, pdf_bytes = generate_all_about_me_profile(
        "Name: Alex Rivera\nFavorite reinforcers: stickers"
    )
    assert "# All About Me" in markdown
    assert "Alex Rivera" in markdown
    assert pdf_bytes.startswith(b"%PDF")


def test_generate_all_about_me_pdf_wrapper(
    monkeypatch: pytest.MonkeyPatch, ollama_server: str
) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", ollama_server)
    pdf_bytes = generate_all_about_me_pdf("Name: Alex Rivera")
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000
