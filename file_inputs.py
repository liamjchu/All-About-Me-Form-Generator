"""Helpers for turning uploaded files into text/images the AI backend can use."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageEnhance, ImageOps
from pypdf import PdfReader

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional until installed
    pytesseract = None  # type: ignore[assignment]

# Only rotate when the best upright score clearly beats the next option.
_ORIENTATION_MARGIN: float = 1.15
_ORIENTATION_MAX_SIDE: int = 400
# Bottom band that usually holds the printed date (left) and page mark (right).
_FOOTER_HEIGHT_RATIO: float = 0.12
_FOOTER_MAX_WIDTH: int = 900
# Vision models read form text well around this size; larger phone photos
# mostly add glare/noise and slow the request down.
_VISION_MAX_SIDE: int = 1400
_VISION_CONTRAST: float = 1.12
_VISION_SHARPNESS: float = 1.2
_VISION_JPEG_QUALITY: int = 85

_PAGE_MARK_RE = re.compile(
    r"(?P<page>\d+)\s*(?:of|/)\s*(?P<total>\d+)",
    re.IGNORECASE,
)
_DATE_LINE_RE = re.compile(
    r"(?P<date>"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\.?\s+\d{1,2}\s+\d{4}\s+\d{1,2}:\d{2}\s*[AP]M(?:\s*[A-Z]{2,4})?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PreparedInput:
    raw_text: str | None = None
    image_bytes: bytes | None = None
    image_mime_type: str = "image/png"


@dataclass(frozen=True)
class FooterMark:
    """Printed footer cues used to pair pages of the same intake form."""

    date_line: str | None = None
    page: int | None = None
    total: int | None = None


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
            "Upload a text PDF, or a PNG/JPG of the page instead."
        )
    return combined


def _downscale_for_orientation(image: Image.Image) -> Image.Image:
    """Smaller grayscale copy so orientation scoring stays cheap."""
    gray = image.convert("L")
    width, height = gray.size
    longest = max(width, height)
    if longest <= _ORIENTATION_MAX_SIDE:
        return gray
    scale = _ORIENTATION_MAX_SIDE / longest
    return gray.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.BILINEAR,
    )


def _upright_score(image: Image.Image) -> float:
    """Higher score ≈ more horizontal structure (typical of upright text lines)."""
    gray = _downscale_for_orientation(image)
    width, height = gray.size
    pixels = gray.load()
    energy = 0.0
    for y in range(1, height):
        row = 0.0
        for x in range(width):
            delta = pixels[x, y] - pixels[x, y - 1]
            row += delta * delta
        energy += row
    return energy / (width * height)


def _best_content_rotation(image: Image.Image) -> int:
    """Return 0/90/180/270 degrees to rotate counter-clockwise for upright text.

    Horizontal text-line energy cannot tell 0° from 180° (or 90° from 270°),
    so we only flip between axis pairs when one clearly wins. Upside-down
    pages (180°) are left as-is without OCR.
    """
    scores = {
        degrees: _upright_score(
            image if degrees == 0 else image.rotate(degrees, expand=True)
        )
        for degrees in (0, 90, 180, 270)
    }
    axis_0 = max(scores[0], scores[180])
    axis_90 = max(scores[90], scores[270])

    if axis_90 >= axis_0 * _ORIENTATION_MARGIN and axis_90 > 0:
        return 90 if scores[90] >= scores[270] else 270
    return 0


def enhance_form_photo(image: Image.Image) -> Image.Image:
    """Boost contrast/sharpness and cap size so vision OCR stays reliable."""
    if image.mode not in ("RGB", "L"):
        working = image.convert("RGB")
    elif image.mode == "L":
        working = image.convert("RGB")
    else:
        working = image.copy()

    working = ImageOps.autocontrast(working, cutoff=1)
    working = ImageEnhance.Contrast(working).enhance(_VISION_CONTRAST)
    working = ImageEnhance.Sharpness(working).enhance(_VISION_SHARPNESS)

    width, height = working.size
    longest = max(width, height)
    if longest > _VISION_MAX_SIDE:
        scale = _VISION_MAX_SIDE / longest
        working = working.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
    return working


def normalize_image_orientation(
    image_bytes: bytes,
    *,
    mime_type: str = "image/png",
) -> tuple[bytes, str]:
    """Apply EXIF orientation, then auto-correct sideways document scans.

    Phone photos usually only need EXIF. Flatbed/phone scans with no EXIF
    are corrected when horizontal text-line structure clearly prefers another
    rotation. Ambiguous images (blank color blocks, etc.) are left alone.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            image = ImageOps.exif_transpose(opened)
            # Copy so the file handle can close before we mutate/re-encode.
            image = image.copy()
    except Exception as error:
        raise ValueError(f"Could not read image: {error}") from error

    degrees = _best_content_rotation(image)
    if degrees:
        image = image.rotate(degrees, expand=True)

    image = enhance_form_photo(image)
    return _encode_image(image, mime_type)


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
        resolved_mime = mime if mime.startswith("image/") else (
            "image/jpeg" if name.endswith((".jpg", ".jpeg")) else "image/png"
        )
        image_bytes, image_mime = normalize_image_orientation(
            file_bytes,
            mime_type=resolved_mime,
        )
        return PreparedInput(
            image_bytes=image_bytes,
            image_mime_type=image_mime,
        )

    if mime == "application/pdf" or name.endswith(".pdf"):
        return PreparedInput(raw_text=extract_text_from_pdf(file_bytes))

    # txt / csv / other text-like uploads
    return PreparedInput(raw_text=file_bytes.decode("utf-8", errors="replace"))


def _encode_image(image: Image.Image, mime_type: str) -> tuple[bytes, str]:
    out = io.BytesIO()
    if mime_type in ("image/jpeg", "image/jpg"):
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(out, format="JPEG", quality=_VISION_JPEG_QUALITY)
        return out.getvalue(), "image/jpeg"

    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(out, format="PNG")
    return out.getvalue(), "image/png"


def crop_footer_band(
    image_bytes: bytes,
    *,
    mime_type: str = "image/png",
    height_ratio: float = _FOOTER_HEIGHT_RATIO,
) -> tuple[bytes, str]:
    """Crop the bottom band that holds the printed date and page number."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            image = opened.copy()
    except Exception as error:
        raise ValueError(f"Could not read image: {error}") from error

    width, height = image.size
    band_height = max(1, int(height * height_ratio))
    footer = image.crop((0, height - band_height, width, height))

    # Keep the vision request small — only the footer text matters.
    if footer.width > _FOOTER_MAX_WIDTH:
        scale = _FOOTER_MAX_WIDTH / footer.width
        footer = footer.resize(
            (max(1, int(footer.width * scale)), max(1, int(footer.height * scale))),
            Image.Resampling.BILINEAR,
        )

    return _encode_image(footer, mime_type)


def normalize_date_line(value: str | None) -> str | None:
    """Collapse whitespace so footer dates can be compared reliably."""
    if value is None:
        return None
    cleaned = " ".join(value.split()).strip()
    return cleaned or None


def parse_footer_ocr_text(text: str) -> FooterMark:
    """Pull date/page/total out of OCR text from a footer crop."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return FooterMark()

    page = None
    total = None
    page_match = _PAGE_MARK_RE.search(cleaned)
    if page_match:
        page = int(page_match.group("page"))
        total = int(page_match.group("total"))

    date_line = None
    date_match = _DATE_LINE_RE.search(cleaned)
    if date_match:
        date_line = normalize_date_line(date_match.group("date"))

    return FooterMark(date_line=date_line, page=page, total=total)


def extract_footer_mark_ocr(footer_image_bytes: bytes) -> FooterMark:
    """Fast local OCR for footer date + page mark (no LLM)."""
    if not footer_image_bytes or pytesseract is None:
        return FooterMark()

    try:
        with Image.open(io.BytesIO(footer_image_bytes)) as opened:
            image = ImageOps.grayscale(opened.convert("RGB"))
            image = ImageOps.autocontrast(image, cutoff=1)
            # Upscale thin footer bands so Tesseract can resolve small print.
            if image.height < 80:
                scale = 80 / image.height
                image = image.resize(
                    (
                        max(1, int(image.width * scale)),
                        max(1, int(image.height * scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )
            text = pytesseract.image_to_string(image, config="--psm 6")
    except Exception:
        return FooterMark()

    return parse_footer_ocr_text(text)


def footers_belong_together(marks: Sequence[FooterMark]) -> bool:
    """Return True when adjacent pages look like one multi-page form.

    Pages belong together when:
    - bottom-right page numbers are consecutive (and totals agree), and any
      readable bottom-left dates do not conflict; or
    - page numbers are unreadable, but every page shares the same exact
      bottom-left date/time stamp.
    """
    if len(marks) <= 1:
        return True

    dates = [normalize_date_line(mark.date_line) for mark in marks]
    present_dates = [date for date in dates if date]
    if len(set(present_dates)) > 1:
        return False

    pages = [mark.page for mark in marks]
    totals = {mark.total for mark in marks if mark.total is not None}
    if len(totals) > 1:
        return False

    if all(page is not None for page in pages):
        first = pages[0]
        assert first is not None
        # Readable page marks win: only consecutive pages of the same form pair.
        return all(page == first + offset for offset, page in enumerate(pages))

    # Page OCR failed — require an exact matching date on every page.
    return len(present_dates) == len(marks) and len(set(present_dates)) == 1


def group_upload_indices(
    *,
    is_image: Sequence[bool],
    footers: Sequence[FooterMark | None],
    pages_per_form: int = 2,
) -> list[list[int]]:
    """Group uploads in order into one-profile bundles.

    Non-image files stay alone (they already hold a full participant). Image
    runs are packed into ``pages_per_form``-sized groups when footer cues say
    the adjacent pages belong together; otherwise each image stays alone.
    """
    if pages_per_form < 1:
        raise ValueError("pages_per_form must be at least 1.")
    if len(is_image) != len(footers):
        raise ValueError("is_image and footers must be the same length.")

    groups: list[list[int]] = []
    index = 0
    total = len(is_image)
    while index < total:
        if not is_image[index]:
            groups.append([index])
            index += 1
            continue

        end = index + pages_per_form
        if (
            pages_per_form > 1
            and end <= total
            and all(is_image[pos] for pos in range(index, end))
        ):
            marks = [
                footers[pos] if footers[pos] is not None else FooterMark()
                for pos in range(index, end)
            ]
            if footers_belong_together(marks):
                groups.append(list(range(index, end)))
                index = end
                continue

        groups.append([index])
        index += 1

    return groups


def group_label(*, file_names: Sequence[str], footers: Sequence[FooterMark | None]) -> str:
    """Human-readable label for a grouped profile."""
    pages = [
        mark.page
        for mark in footers
        if mark is not None and mark.page is not None
    ]
    if pages and len(pages) == len(file_names):
        if len(pages) == 1:
            return f"page-{pages[0]}"
        return f"pages-{pages[0]}-{pages[-1]}"

    stems = [Path(name).stem for name in file_names]
    if len(stems) == 1:
        return stems[0]
    return f"{stems[0]} (+{len(stems) - 1} pages)"
