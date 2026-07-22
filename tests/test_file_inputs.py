"""Tests for uploaded PDF normalization (real PDFs, no mocks)."""

from __future__ import annotations

import pytest

from file_inputs import (
    PreparedInput,
    extract_text_from_pdf,
    is_pdf_upload,
    prepare_upload,
    profile_stem,
)
from tests.helpers import make_empty_pdf, make_text_pdf


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


def test_is_pdf_upload_by_mime_or_extension() -> None:
    assert is_pdf_upload(file_name="intake.PDF", mime_type=None)
    assert is_pdf_upload(file_name="scan.bin", mime_type="application/pdf")
    assert not is_pdf_upload(file_name="photo.jpg", mime_type="image/jpeg")
    assert not is_pdf_upload(file_name="notes.txt", mime_type="text/plain")


def test_prepare_upload_pdf_by_extension() -> None:
    prepared = prepare_upload(
        file_name="intake.PDF",
        file_bytes=make_text_pdf("Does the participant need help in the restroom? No"),
        mime_type=None,
    )
    assert isinstance(prepared, PreparedInput)
    assert "restroom" in prepared.raw_text


def test_prepare_upload_pdf_by_mime() -> None:
    prepared = prepare_upload(
        file_name="scan.bin",
        file_bytes=make_text_pdf("Participant allergies: none"),
        mime_type="application/pdf",
    )
    assert "allergies" in prepared.raw_text


def test_prepare_upload_rejects_non_pdf() -> None:
    with pytest.raises(ValueError, match="Only PDF"):
        prepare_upload(
            file_name="notes.txt",
            file_bytes=b"Name: Casey",
            mime_type="text/plain",
        )


def test_prepare_upload_rejects_oversize_pdf() -> None:
    from file_inputs import MAX_UPLOAD_BYTES

    with pytest.raises(ValueError, match="10 MB"):
        prepare_upload(
            file_name="huge.pdf",
            file_bytes=b"%PDF" + (b"x" * (MAX_UPLOAD_BYTES + 1)),
            mime_type="application/pdf",
        )


def test_profile_stem_uses_file_name() -> None:
    assert profile_stem("Bob-Joe.pdf") == "Bob-Joe"
