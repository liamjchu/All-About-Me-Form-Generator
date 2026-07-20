"""Fill formTemplate.pdf by drawing text over the page-image placeholders."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Final

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

PROJECT_ROOT: Final = Path(__file__).resolve().parent
TEMPLATE_PDF: Final = PROJECT_ROOT / "formTemplate.pdf"

# Preview calibration used 1190x1541; page images are 1545x2000.
_PREVIEW_SIZE: Final = (1190, 1541)
_IMAGE_SIZE: Final = (1545, 2000)

# Boxes in preview coordinates (x0, y0, x1, y1), top-left origin.
_PAGE1_BOXES: Final[dict[str, tuple[int, int, int, int]]] = {
    "name": (300, 430, 890, 530),
    "favorite_things_1": (155, 770, 750, 830),
    "favorite_things_2": (155, 853, 750, 913),
    "favorite_things_3": (155, 937, 750, 997),
    "reinforcers_1": (395, 1100, 1050, 1160),
    "reinforcers_2": (395, 1182, 1050, 1242),
    "reinforcers_3": (395, 1265, 1050, 1325),
    "reinforcers_4": (395, 1348, 1050, 1408),
}

_PAGE2_BOXES: Final[dict[str, tuple[int, int, int, int]]] = {
    "parent_name": (280, 260, 1000, 320),
    "parent_phone": (290, 345, 1000, 405),
    "parent_email": (280, 430, 1000, 490),
    "allergies": (150, 595, 1030, 710),
    "bathroom_needs": (150, 800, 1030, 880),
    "behavioral_management": (140, 985, 1040, 1140),
}

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
    "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf",
    "/System/Library/Fonts/SFCompactRounded.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
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

    # Sensible defaults for medical-style fields when unknown.
    for key in ("allergies", "bathroom_needs"):
        if not data[key]:
            data[key] = "N/A"

    return data


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


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


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
    multiline: bool = False,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(255, 255, 255))
    if not text:
        return

    font = _load_font(font_size)
    max_width = max(1, x1 - x0 - 16)
    lines = _wrap_text(draw, text, font, max_width) if multiline else [text]

    # Shrink font if a single line is too wide.
    while not multiline and len(lines) == 1:
        width = draw.textbbox((0, 0), lines[0], font=font)[2]
        if width <= max_width or font_size <= 22:
            break
        font_size -= 2
        font = _load_font(font_size)

    line_height = draw.textbbox((0, 0), "Ag", font=font)[3] + 6
    total_height = line_height * len(lines)
    y = y0 + max(0, (y1 - y0 - total_height) // 2)

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = x0 + (x1 - x0 - text_w) // 2 if center else x0 + 8
        draw.text((x, y), line, fill=(0, 0, 0), font=font)
        y += line_height


def _extract_page_images() -> list[Image.Image]:
    if not TEMPLATE_PDF.exists():
        raise RuntimeError(f"PDF template not found: {TEMPLATE_PDF.name}")

    reader = PdfReader(str(TEMPLATE_PDF))
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
    """Return a filled PDF (bytes) based on formTemplate.pdf."""
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
        font_size=64,
    )
    for index in range(1, 4):
        key = f"favorite_things_{index}"
        _draw_field(
            draw1,
            _map_box(_PAGE1_BOXES[key], size),
            data[key],
            font_size=40,
        )
    for index in range(1, 5):
        key = f"reinforcers_{index}"
        _draw_field(
            draw1,
            _map_box(_PAGE1_BOXES[key], size),
            data[key],
            font_size=40,
        )

    page2 = pages[1].copy()
    draw2 = ImageDraw.Draw(page2)
    for key, font_size, multiline in (
        ("parent_name", 40, False),
        ("parent_phone", 40, False),
        ("parent_email", 36, False),
        ("allergies", 40, True),
        ("bathroom_needs", 40, True),
        ("behavioral_management", 40, True),
    ):
        _draw_field(
            draw2,
            _map_box(_PAGE2_BOXES[key], size),
            data[key],
            font_size=font_size,
            multiline=multiline,
        )

    reader = PdfReader(str(TEMPLATE_PDF))
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

    # Re-write through pypdf so the download is a clean 2-page PDF.
    writer = PdfWriter()
    for page in PdfReader(buffer).pages:
        writer.add_page(page)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()
