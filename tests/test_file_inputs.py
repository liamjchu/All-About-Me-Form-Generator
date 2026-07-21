"""Tests for uploaded-file normalization (real PDFs/images, no mocks)."""

from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw

from file_inputs import (
    FooterMark,
    PreparedInput,
    crop_footer_band,
    enhance_form_photo,
    extract_footer_mark_ocr,
    extract_text_from_pdf,
    footers_belong_together,
    group_label,
    group_upload_indices,
    normalize_date_line,
    normalize_image_orientation,
    parse_footer_ocr_text,
    prepare_upload,
)
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
    assert prepared.image_bytes is not None
    assert prepared.image_mime_type == "image/png"
    assert prepared.raw_text is None
    with Image.open(io.BytesIO(prepared.image_bytes)) as image:
        assert image.size == (64, 64)


def test_prepare_upload_jpeg_by_extension_defaults_mime() -> None:
    jpeg = make_jpeg_bytes()
    prepared = prepare_upload(
        file_name="cam.JPG",
        file_bytes=jpeg,
        mime_type=None,
    )
    assert prepared.image_bytes is not None
    assert prepared.image_mime_type == "image/jpeg"
    with Image.open(io.BytesIO(prepared.image_bytes)) as image:
        assert image.format == "JPEG"


def test_prepare_upload_jpeg_keeps_image_mime() -> None:
    jpeg = make_jpeg_bytes()
    prepared = prepare_upload(
        file_name="cam.jpeg",
        file_bytes=jpeg,
        mime_type="image/jpeg",
    )
    assert prepared.image_mime_type == "image/jpeg"


def _lined_document(size: tuple[int, int] = (200, 280)) -> Image.Image:
    """Synthetic form page: light background with dark horizontal text-like lines."""
    image = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(image)
    for y in range(40, size[1] - 20, 18):
        draw.rectangle((24, y, size[0] - 24, y + 4), fill=(30, 30, 30))
    return image


def test_normalize_image_applies_exif_orientation() -> None:
    # Wide pixels tagged as Orientation=6 (rotate 90 CW) should become portrait.
    image = Image.new("RGB", (120, 40), (10, 120, 200))
    exif = image.getexif()
    exif[274] = 6  # Orientation
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)

    fixed_bytes, mime = normalize_image_orientation(
        buffer.getvalue(),
        mime_type="image/jpeg",
    )
    assert mime == "image/jpeg"
    with Image.open(io.BytesIO(fixed_bytes)) as fixed:
        assert fixed.size == (40, 120)


def test_normalize_image_rotates_sideways_document() -> None:
    upright = _lined_document()
    sideways = upright.rotate(90, expand=True)
    buffer = io.BytesIO()
    sideways.save(buffer, format="PNG")

    fixed_bytes, mime = normalize_image_orientation(
        buffer.getvalue(),
        mime_type="image/png",
    )
    assert mime == "image/png"
    with Image.open(io.BytesIO(fixed_bytes)) as fixed:
        # Should be portrait again (taller than wide), matching the source form.
        assert fixed.size[1] > fixed.size[0]


def test_normalize_image_leaves_ambiguous_color_blocks() -> None:
    png = make_png_bytes(size=(80, 40))  # landscape solid color — no text cues
    fixed_bytes, _ = normalize_image_orientation(png, mime_type="image/png")
    with Image.open(io.BytesIO(fixed_bytes)) as fixed:
        assert fixed.size == (80, 40)


def test_enhance_form_photo_downscales_large_pages() -> None:
    huge = Image.new("RGB", (3200, 2400), (200, 200, 200))
    draw = ImageDraw.Draw(huge)
    draw.rectangle((40, 40, 3100, 80), fill=(20, 20, 20))
    enhanced = enhance_form_photo(huge)
    assert max(enhanced.size) == 1400


def test_normalize_image_downscales_large_upload() -> None:
    image = Image.new("RGB", (2400, 3200), (240, 240, 240))
    draw = ImageDraw.Draw(image)
    for y in range(80, 3000, 40):
        draw.rectangle((60, y, 2200, y + 6), fill=(25, 25, 25))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)

    fixed_bytes, mime = normalize_image_orientation(
        buffer.getvalue(),
        mime_type="image/jpeg",
    )
    assert mime == "image/jpeg"
    with Image.open(io.BytesIO(fixed_bytes)) as fixed:
        assert max(fixed.size) == 1400


def test_parse_footer_ocr_text_reads_date_and_page() -> None:
    mark = parse_footer_ocr_text("Jun 30 2026 1:21PM ET          45 of 85")
    assert mark.date_line == "Jun 30 2026 1:21PM ET"
    assert mark.page == 45
    assert mark.total == 85


def test_parse_footer_ocr_text_accepts_slash_page_mark() -> None:
    mark = parse_footer_ocr_text("Jul 1 2026 9:00AM ET 12/40")
    assert mark.page == 12
    assert mark.total == 40


def test_extract_footer_mark_ocr_reads_printed_footer() -> None:
    from PIL import ImageFont

    image = Image.new("RGB", (1100, 90), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 28)
    draw.text((16, 28), "Jun 30 2026 1:21PM ET", fill=(0, 0, 0), font=font)
    draw.text((880, 28), "45 of 85", fill=(0, 0, 0), font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    mark = extract_footer_mark_ocr(buffer.getvalue())
    assert mark.page == 45
    assert mark.total == 85
    assert mark.date_line is not None
    assert "2026" in mark.date_line


def test_crop_footer_band_is_shorter_than_source() -> None:
    png = make_png_bytes(size=(200, 400))
    footer_bytes, mime = crop_footer_band(png, mime_type="image/png")
    assert mime == "image/png"
    with Image.open(io.BytesIO(png)) as source, Image.open(io.BytesIO(footer_bytes)) as footer:
        assert footer.size[1] < source.size[1]
        assert footer.size[0] <= source.size[0]


def test_normalize_date_line_collapses_whitespace() -> None:
    assert normalize_date_line("  Jun 30   2026 1:21PM ET ") == "Jun 30 2026 1:21PM ET"
    assert normalize_date_line("   ") is None


def test_footers_belong_together_consecutive_pages_same_date() -> None:
    date = "Jun 30 2026 1:21PM ET"
    assert footers_belong_together(
        [
            FooterMark(date_line=date, page=45, total=85),
            FooterMark(date_line=date, page=46, total=85),
        ]
    )


def test_footers_belong_together_rejects_page_gap() -> None:
    date = "Jun 30 2026 1:21PM ET"
    assert not footers_belong_together(
        [
            FooterMark(date_line=date, page=45, total=85),
            FooterMark(date_line=date, page=47, total=85),
        ]
    )


def test_footers_belong_together_uses_matching_dates_when_pages_missing() -> None:
    date = "Jun 30 2026 1:21PM ET"
    assert footers_belong_together(
        [
            FooterMark(date_line=date, page=None, total=None),
            FooterMark(date_line=date, page=None, total=None),
        ]
    )
    assert not footers_belong_together(
        [
            FooterMark(date_line=date, page=None, total=None),
            FooterMark(date_line="Jul 1 2026 9:00AM ET", page=None, total=None),
        ]
    )


def test_group_upload_indices_pairs_consecutive_form_pages() -> None:
    date = "Jun 30 2026 1:21PM ET"
    groups = group_upload_indices(
        is_image=[True, True, True, True],
        footers=[
            FooterMark(date_line=date, page=45, total=85),
            FooterMark(date_line=date, page=46, total=85),
            FooterMark(date_line=date, page=47, total=85),
            FooterMark(date_line=date, page=48, total=85),
        ],
        pages_per_form=2,
    )
    assert groups == [[0, 1], [2, 3]]


def test_group_upload_indices_keeps_text_files_solo() -> None:
    groups = group_upload_indices(
        is_image=[False, True, True],
        footers=[
            None,
            FooterMark(date_line="Jun 30 2026 1:21PM ET", page=1, total=2),
            FooterMark(date_line="Jun 30 2026 1:21PM ET", page=2, total=2),
        ],
        pages_per_form=2,
    )
    assert groups == [[0], [1, 2]]


def test_group_label_prefers_page_numbers() -> None:
    assert (
        group_label(
            file_names=["a.jpg", "b.jpg"],
            footers=[
                FooterMark(page=45, total=85),
                FooterMark(page=46, total=85),
            ],
        )
        == "pages-45-46"
    )
