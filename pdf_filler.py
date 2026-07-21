"""Fill formTemplate.pdf by drawing text over the page-image placeholders."""

from __future__ import annotations

import io
import re
import warnings
from pathlib import Path
from typing import Final

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

PROJECT_ROOT: Final = Path(__file__).resolve().parent
_TEMPLATE_NAMES: Final = ("formTemplate.pdf", "All About Me Template.pdf")
_FONTS_DIR: Final = PROJECT_ROOT / "fonts"

# Preview calibration used 1190x1541; page images are typically 2550x3300.
_PREVIEW_SIZE: Final = (1190, 1541)

# Boxes in preview coordinates (x0, y0, x1, y1), top-left origin.
# Sized so filled text can match nearby template labels/headers.
_PAGE1_BOXES: Final[dict[str, tuple[int, int, int, int]]] = {
    "name": (250, 420, 940, 545),
    "favorite_things_1": (90, 760, 940, 855),
    "favorite_things_2": (90, 865, 940, 960),
    "favorite_things_3": (90, 970, 940, 1065),
    "reinforcers_1": (340, 1090, 1100, 1190),
    "reinforcers_2": (340, 1200, 1100, 1290),
    "reinforcers_3": (340, 1300, 1100, 1390),
    "reinforcers_4": (340, 1400, 1100, 1490),
}

# Bulleted list layout on page 1 (preview coordinates, relative to each cover box).
_PAGE1_BULLET_FROM_BOX_X: Final = 50
_PAGE1_BULLET_TEXT_FROM_BOX_X: Final = 110
_BULLET_RADIUS_PREVIEW: Final = 8

_PAGE2_BOXES: Final[dict[str, tuple[int, int, int, int]]] = {
    "parent_name": (280, 245, 1040, 335),
    "parent_phone": (290, 330, 1040, 420),
    "parent_email": (280, 415, 1040, 505),
    "allergies": (140, 605, 1040, 760),
    "bathroom_needs": (140, 860, 1040, 1045),
    "behavioral_management": (130, 1150, 1050, 1420),
}

# Target font sizes (pixels at full page resolution). Shrink floors keep long
# values readable instead of collapsing to tiny text.
_FONT_NAME: Final = 240
_FONT_BULLET: Final = 110
_FONT_PAGE2_LINE: Final = 115
_FONT_PAGE2_EMAIL: Final = 105
_FONT_PAGE2_BODY: Final = 100
_FONT_MIN_NAME: Final = 140
_FONT_MIN_BULLET: Final = 70
_FONT_MIN_PAGE2: Final = 70

_FIELD_KEYS: Final[tuple[str, ...]] = (
    "name",
    "favorite_things_1",
    "favorite_things_2",
    "favorite_things_3",
    "reinforcers_1",
    "reinforcers_2",
    "reinforcers_3",
    "reinforcers_4",
    "parent_name",
    "parent_phone",
    "parent_email",
    "allergies",
    "bathroom_needs",
    "behavioral_management",
)

_FONT_CANDIDATES: Final[tuple[str, ...]] = (
    str(_FONTS_DIR / "Chewy-Regular.ttf"),
    "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf",
    "C:/Windows/Fonts/ARLRDBD.TTF",
    str(_FONTS_DIR / "Nunito-ExtraBold.ttf"),
    str(_FONTS_DIR / "Nunito-Bold.ttf"),
    "/System/Library/Fonts/SFCompactRounded.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)


def _template_pdf() -> Path:
    for name in _TEMPLATE_NAMES:
        path = PROJECT_ROOT / name
        if path.is_file():
            return path
    raise RuntimeError(
        "PDF template not found. Add formTemplate.pdf or All About Me Template.pdf "
        "to the project folder (each page must be a full-page embedded bitmap)."
    )


def empty_form_data() -> dict[str, str]:
    return {key: "" for key in _FIELD_KEYS}


def normalize_form_data(raw: dict) -> dict[str, str]:
    """Map model JSON (lists or flat keys) into the flat field dict used by the filler."""
    data = empty_form_data()
    if not isinstance(raw, dict):
        return data

    if isinstance(raw.get("name"), str):
        data["name"] = raw["name"].strip()

    things = raw.get("favorite_things")
    if isinstance(things, list):
        for index, value in enumerate(things[:3], start=1):
            data[f"favorite_things_{index}"] = str(value).strip()
    for index in range(1, 4):
        key = f"favorite_things_{index}"
        if isinstance(raw.get(key), str) and raw[key].strip():
            data[key] = raw[key].strip()

    reinforcers = raw.get("favorite_reinforcers") or raw.get("reinforcers")
    if isinstance(reinforcers, list):
        for index, value in enumerate(reinforcers[:4], start=1):
            data[f"reinforcers_{index}"] = str(value).strip()
    for index in range(1, 5):
        key = f"reinforcers_{index}"
        if isinstance(raw.get(key), str) and raw[key].strip():
            data[key] = raw[key].strip()

    for key in (
        "parent_name",
        "parent_phone",
        "parent_email",
        "allergies",
        "bathroom_needs",
        "behavioral_management",
    ):
        if isinstance(raw.get(key), str):
            data[key] = raw[key].strip()

    # Mobility phrases (e.g. "independent walker") must not land in toileting.
    data["bathroom_needs"] = _sanitize_bathroom_needs(data["bathroom_needs"])
    data["allergies"] = _na_if_blank_or_none(data["allergies"])
    data["bathroom_needs"] = _na_if_blank_or_none(data["bathroom_needs"])

    return data


def _na_if_blank_or_none(text: str) -> str:
    """Normalize empty / none-style answers to N/A for medical-style fields."""
    if not text or not text.strip():
        return "N/A"
    if re.fullmatch(
        r"(?:n/?a|none|no|nil|nill|not\s+applicable)(?:\s*[.!]*)?",
        text.strip(),
        flags=re.IGNORECASE,
    ):
        return "N/A"
    return text


def _sanitize_bathroom_needs(text: str) -> str:
    """Keep toileting independence; drop mobility wording like 'independent walker'."""
    if not text:
        return text
    cleaned = re.sub(
        r"\bindependent\s+walker\b",
        "independent",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;.-")
    # Bare "no help" style answers become N/A via _na_if_blank_or_none.
    if re.fullmatch(
        r"(?:no(?:\s+help(?:\s+needed)?)?|does\s+not\s+need\s+help|"
        r"independent(?:ly)?(?:\s+in\s+the\s+restroom)?)\.?",
        cleaned,
        flags=re.IGNORECASE,
    ):
        return "N/A"
    return cleaned


def _map_box(box: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    rw, rh = size
    pw, ph = _PREVIEW_SIZE
    return (
        int(x0 * rw / pw),
        int(y0 * rh / ph),
        int(x1 * rw / pw),
        int(y1 * rh / ph),
    )


def _map_x(x: int, size: tuple[int, int]) -> int:
    pw = _PREVIEW_SIZE[0]
    return int(x * size[0] / pw)


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    warnings.warn(
        "No preferred form font found; using Pillow default at the requested size.",
        stacklevel=2,
    )
    return ImageFont.load_default(size=size)


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_field(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    center: bool = False,
    font_size: int = 40,
    min_font_size: int = 28,
    multiline: bool = False,
    white_bg: bool = False,
) -> None:
    x0, y0, x1, y1 = box
    if white_bg:
        draw.rectangle(box, fill=(255, 255, 255))
    if not text:
        return

    font = _load_font(font_size)
    max_width = max(1, x1 - x0 - 16)
    max_height = max(1, y1 - y0)
    lines = _wrap_text(draw, text, font, max_width) if multiline else [text]
    floor = min(min_font_size, font_size)

    if multiline:
        # Shrink font when wrapped lines exceed the box height.
        while font_size > floor:
            line_height = draw.textbbox((0, 0), "Ag", font=font)[3] + 8
            total_height = line_height * len(lines)
            if total_height <= max_height:
                break
            font_size -= 2
            font = _load_font(font_size)
            lines = _wrap_text(draw, text, font, max_width)
    else:
        # Shrink font if a single line is too wide.
        while len(lines) == 1:
            width = draw.textbbox((0, 0), lines[0], font=font)[2]
            if width <= max_width or font_size <= floor:
                break
            font_size -= 2
            font = _load_font(font_size)

    line_height = draw.textbbox((0, 0), "Ag", font=font)[3] + 8
    total_height = line_height * len(lines)
    y = y0 + max(0, (y1 - y0 - total_height) // 2)

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = x0 + (x1 - x0 - text_w) // 2 if center else x0 + 8
        draw.text((x, y), line, fill=(0, 0, 0), font=font)
        y += line_height


def _draw_bullet_line(
    draw: ImageDraw.ImageDraw,
    size: tuple[int, int],
    cover_box: tuple[int, int, int, int],
    text: str,
    *,
    font_size: int = _FONT_BULLET,
    min_font_size: int = _FONT_MIN_BULLET,
) -> None:
    """Draw a bullet and text for a filled row; skip empty rows."""
    if not text.strip():
        return
    x0, y0, x1, y1 = _map_box(cover_box, size)

    box_x0_preview = cover_box[0]
    text_x = _map_x(box_x0_preview + _PAGE1_BULLET_TEXT_FROM_BOX_X, size)
    bullet_x = _map_x(box_x0_preview + _PAGE1_BULLET_FROM_BOX_X, size)
    bullet_r = max(4, _map_x(_BULLET_RADIUS_PREVIEW, size))
    cy = (y0 + y1) // 2

    draw.ellipse(
        (bullet_x - bullet_r, cy - bullet_r, bullet_x + bullet_r, cy + bullet_r),
        fill=(0, 0, 0),
    )

    floor = min(min_font_size, font_size)
    font = _load_font(font_size)
    max_width = max(1, x1 - text_x - 8)
    while font_size > floor:
        width = draw.textbbox((0, 0), text, font=font)[2]
        if width <= max_width:
            break
        font_size -= 2
        font = _load_font(font_size)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_h = bbox[3] - bbox[1]
    y = cy - text_h // 2 - bbox[1]
    draw.text((text_x, y), text, fill=(0, 0, 0), font=font)


def _extract_page_images() -> list[Image.Image]:
    template_pdf = _template_pdf()

    reader = PdfReader(str(template_pdf))
    images: list[Image.Image] = []
    for page in reader.pages:
        resources = page.get("/Resources")
        if resources is None:
            raise RuntimeError("PDF page is missing image resources.")
        xobjects = resources.get_object().get("/XObject")
        if xobjects is None:
            raise RuntimeError("PDF page has no embedded image to fill.")
        xobjects = xobjects.get_object()
        page_image = None
        for name in xobjects:
            xobj = xobjects[name].get_object()
            if xobj.get("/Subtype") == "/Image":
                page_image = xobj
                break
        if page_image is None:
            raise RuntimeError("Could not find the template page image.")
        data = page_image.get_data()
        width = int(page_image["/Width"])
        height = int(page_image["/Height"])
        # Template pages are RGB FlateDecode bitmaps.
        image = Image.frombytes("RGB", (width, height), data)
        images.append(image)
    if len(images) != 2:
        raise RuntimeError(f"Expected 2 template pages, found {len(images)}.")
    return images


def fill_form_pdf(form_data: dict[str, str]) -> bytes:
    """Return a filled PDF (bytes) based on All About Me Template.pdf."""
    data = normalize_form_data(form_data)
    pages = _extract_page_images()
    size = pages[0].size

    page1 = pages[0].copy()
    draw1 = ImageDraw.Draw(page1)
    _draw_field(
        draw1,
        _map_box(_PAGE1_BOXES["name"], size),
        data["name"],
        center=True,
        font_size=_FONT_NAME,
        min_font_size=_FONT_MIN_NAME,
        white_bg=True,
    )
    for index in range(1, 4):
        key = f"favorite_things_{index}"
        _draw_bullet_line(
            draw1,
            size,
            _PAGE1_BOXES[key],
            data[key],
            font_size=_FONT_BULLET,
            min_font_size=_FONT_MIN_BULLET,
        )
    for index in range(1, 5):
        key = f"reinforcers_{index}"
        _draw_bullet_line(
            draw1,
            size,
            _PAGE1_BOXES[key],
            data[key],
            font_size=_FONT_BULLET,
            min_font_size=_FONT_MIN_BULLET,
        )

    page2 = pages[1].copy()
    draw2 = ImageDraw.Draw(page2)
    for key, font_size, multiline in (
        ("parent_name", _FONT_PAGE2_LINE, False),
        ("parent_phone", _FONT_PAGE2_LINE, False),
        ("parent_email", _FONT_PAGE2_EMAIL, False),
        ("allergies", _FONT_PAGE2_BODY, True),
        ("bathroom_needs", _FONT_PAGE2_BODY, True),
        ("behavioral_management", _FONT_PAGE2_BODY, True),
    ):
        _draw_field(
            draw2,
            _map_box(_PAGE2_BOXES[key], size),
            data[key],
            font_size=font_size,
            min_font_size=_FONT_MIN_PAGE2,
            multiline=multiline,
        )

    reader = PdfReader(str(_template_pdf()))
    page_width = float(reader.pages[0].mediabox.width)
    page_height = float(reader.pages[0].mediabox.height)

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(page_width, page_height))
    for image in (page1, page2):
        pdf.drawImage(
            ImageReader(image),
            0,
            0,
            width=page_width,
            height=page_height,
            preserveAspectRatio=True,
            anchor="c",
        )
        pdf.showPage()
    pdf.save()
    buffer.seek(0)

    writer = PdfWriter()
    for page in PdfReader(buffer).pages:
        writer.add_page(page)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()
