"""Helpers for turning uploaded files into text/images the AI backend can use."""

from __future__ import annotations

import io
from dataclasses import dataclass

from pypdf import PdfReader


@dataclass(frozen=True)
class PreparedInput:
    raw_text: str | None = None
    image_bytes: bytes | None = None
    image_mime_type: str = "image/png"


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Return concatenated text from a PDF, or raise if none is readable."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as error:
        raise ValueError(f"Could not read PDF: {error}") from error

    parts: list[str] = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            parts.append(text)

    combined = "\n\n".join(parts).strip()
    if not combined:
        raise ValueError(
            "This PDF has no extractable text (it may be a scanned image). "
            "Upload a text PDF, or a PNG/JPG of the page instead."
        )
    return combined


def prepare_upload(
    *,
    file_name: str,
    file_bytes: bytes,
    mime_type: str | None,
) -> PreparedInput:
    """Normalize an uploaded file into text and/or image input for generation."""
    name = file_name.lower()
    mime = (mime_type or "").lower()

    if mime.startswith("image/") or name.endswith((".png", ".jpg", ".jpeg")):
        return PreparedInput(
            image_bytes=file_bytes,
            image_mime_type=mime if mime.startswith("image/") else "image/png",
        )

    if mime == "application/pdf" or name.endswith(".pdf"):
        return PreparedInput(raw_text=extract_text_from_pdf(file_bytes))

    # txt / csv / other text-like uploads
    return PreparedInput(raw_text=file_bytes.decode("utf-8", errors="replace"))
