"""Tests for uploaded-file normalization (real PDFs/images, no mocks)."""

from __future__ import annotations

import pytest

from file_inputs import PreparedInput, extract_text_from_pdf, prepare_upload
from tests.helpers import make_empty_pdf, make_jpeg_bytes, make_png_bytes, make_text_pdf


def test_extract_text_from_pdf_joins_pages() -> None:
    pdf_bytes = make_text_pdf("Participant strengths and favorite interests", pages=2)
    text = extract_text_from_pdf(pdf_bytes)
    assert "Participant strengths and favorite interests (page 1)" in text
    assert "page 2" in text


def test_extract_text_from_pdf_rejects_blank_pages() -> None:
    with pytest.raises(ValueError, match="no extractable text"):
        extract_text_from_pdf(make_empty_pdf())


def test_extract_text_from_pdf_rejects_corrupt_bytes() -> None:
    with pytest.raises(ValueError, match="Could not read PDF"):
        extract_text_from_pdf(b"%PDF-not-a-real-file")


def test_prepare_upload_text_file() -> None:
    prepared = prepare_upload(
        file_name="notes.txt",
        file_bytes=b"Name: Casey\nFavorite reinforcers: stickers",
        mime_type="text/plain",
    )
    assert isinstance(prepared, PreparedInput)
    assert prepared.raw_text is not None
    assert "Casey" in prepared.raw_text
    assert prepared.image_bytes is None


def test_prepare_upload_csv_replaces_bad_utf8() -> None:
    prepared = prepare_upload(
        file_name="roster.csv",
        file_bytes=b"name,note\nPat,\xff weird",
        mime_type="text/csv",
    )
    assert prepared.raw_text is not None
    assert "Pat" in prepared.raw_text
    assert "\ufffd" in prepared.raw_text or "weird" in prepared.raw_text


def test_prepare_upload_pdf_by_extension() -> None:
    prepared = prepare_upload(
        file_name="intake.PDF",
        file_bytes=make_text_pdf("Does the participant need help in the restroom? No"),
        mime_type=None,
    )
    assert prepared.raw_text is not None
    assert "restroom" in prepared.raw_text


def test_prepare_upload_pdf_by_mime() -> None:
    prepared = prepare_upload(
        file_name="scan.bin",
        file_bytes=make_text_pdf("Participant allergies: none"),
        mime_type="application/pdf",
    )
    assert "allergies" in (prepared.raw_text or "")


def test_prepare_upload_png_by_mime() -> None:
    png = make_png_bytes()
    prepared = prepare_upload(
        file_name="photo.bin",
        file_bytes=png,
        mime_type="image/png",
    )
    assert prepared.image_bytes == png
    assert prepared.image_mime_type == "image/png"
    assert prepared.raw_text is None


def test_prepare_upload_jpeg_by_extension_defaults_mime() -> None:
    jpeg = make_jpeg_bytes()
    prepared = prepare_upload(
        file_name="cam.JPG",
        file_bytes=jpeg,
        mime_type=None,
    )
    assert prepared.image_bytes == jpeg
    assert prepared.image_mime_type == "image/png"


def test_prepare_upload_jpeg_keeps_image_mime() -> None:
    jpeg = make_jpeg_bytes()
    prepared = prepare_upload(
        file_name="cam.jpeg",
        file_bytes=jpeg,
        mime_type="image/jpeg",
    )
    assert prepared.image_mime_type == "image/jpeg"
