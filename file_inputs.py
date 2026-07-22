"""Helpers for turning uploaded PDFs into text the AI backend can use."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class PreparedInput:
    raw_text: str


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Return concatenated text from a PDF, or raise if none is readable."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as error:
        raise ValueError(f"Could not read PDF: {error}") from error

    parts: list[str] = []
    for page in reader.pages:
        try:
            text = (page.extract_text() or "").strip()
        except Exception as error:
            raise ValueError(f"Could not read PDF: {error}") from error
        if text:
            parts.append(text)

    combined = "\n\n".join(parts).strip()
    if not combined:
        raise ValueError(
            "This PDF has no extractable text (it may be a scanned image). "
            "Upload a text-based PDF instead."
        )
    return combined


def is_pdf_upload(*, file_name: str, mime_type: str | None) -> bool:
    name = file_name.lower()
    mime = (mime_type or "").lower()
    return mime == "application/pdf" or name.endswith(".pdf")


def prepare_upload(
    *,
    file_name: str,
    file_bytes: bytes,
    mime_type: str | None,
) -> PreparedInput:
    """Extract text from an uploaded PDF for generation."""
    if not is_pdf_upload(file_name=file_name, mime_type=mime_type):
        raise ValueError("Only PDF uploads are supported.")
    return PreparedInput(raw_text=extract_text_from_pdf(file_bytes))


def profile_stem(file_name: str) -> str:
    """Human-readable label for a profile built from one PDF."""
    return Path(file_name).stem
