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
# Text is drawn directly on the template (no white fill behind it).
# Favorite things: at most 2 bullets. Reinforcers: at most 3.
_PAGE1_BOXES: Final[dict[str, tuple[int, int, int, int]]] = {
    "name": (250, 420, 940, 545),
    "favorite_things_1": (90, 760, 940, 855),
    "favorite_things_2": (90, 865, 940, 960),
    "favorite_things_3": (90, 970, 940, 1065),  # unused; kept for layout reference
    "reinforcers_1": (340, 1090, 1100, 1190),
    "reinforcers_2": (340, 1200, 1100, 1290),
    "reinforcers_3": (340, 1300, 1100, 1390),
}

_FAVORITE_BULLET_SLOTS: Final = 2
_REINFORCER_BULLET_SLOTS: Final = 3
# Legacy alias used by older call sites / tests that still expect 3 page-1 slots.
_PAGE1_BULLET_SLOTS: Final = _REINFORCER_BULLET_SLOTS

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
    "parent_name",
    "parent_phone",
    "parent_email",
    "allergies",
    "bathroom_needs",
    "behavioral_management",
)

# Diagnoses / disability labels must never land in favorite-things or reinforcer bullets.
_DIAGNOSIS_BLEED: Final = re.compile(
    r"\b(?:"
    r"autism(?:\s+spectrum(?:\s+disorder)?)?|asd|autistic|"
    r"intellectual\s+disabilit(?:y|ies)|"
    r"\bid\b|"
    r"adhd|attention\s+deficit|"
    r"down(?:'?s)?\s+syndrome|"
    r"cerebral\s+palsy|"
    r"developmental\s+delay|"
    r"disabilit(?:y|ies)|diagnos(?:is|es)|"
    r"iep|504\s*plan"
    r")\b",
    flags=re.IGNORECASE,
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


# Header line on intake forms, e.g.:
# "submission ID# 123456 For: Lastname, Firstname | DOB: 1/2/2010 ..."
_FOR_HEADER_RE: Final = re.compile(
    r"(?:submission\s*ID\s*#?\s*\d+\D{0,40})?\bFor:\s*([^|\n]+)",
    flags=re.IGNORECASE,
)


def participant_first_name_from_text(text: str) -> str:
    """Extract participant first name from the intake 'For: Last, First' header only."""
    if not text:
        return ""
    match = _FOR_HEADER_RE.search(text)
    if not match:
        return ""
    return _first_name_only(match.group(1).strip())


def _first_name_only(value: str) -> str:
    """Return a single first name from 'Last, First' or 'First Last' style values."""
    cleaned = (value or "").strip(" \t\r\n,;.-")
    if not cleaned:
        return ""
    if "," in cleaned:
        # "Lastname, Firstname" → Firstname (first token after the comma).
        after = cleaned.split(",", 1)[1].strip()
        return after.split()[0] if after else ""
    # Already a given name, or "First Last" — keep only the first token.
    return cleaned.split()[0]


def normalize_form_data(raw: dict) -> dict[str, str]:
    """Map model JSON (lists or flat keys) into the flat field dict used by the filler."""
    data = empty_form_data()
    if not isinstance(raw, dict):
        return data

    if isinstance(raw.get("name"), str):
        data["name"] = _first_name_only(raw["name"])

    things = raw.get("favorite_things")
    if isinstance(things, list):
        for index, value in enumerate(things[:_FAVORITE_BULLET_SLOTS], start=1):
            data[f"favorite_things_{index}"] = str(value).strip()
    for index in range(1, _FAVORITE_BULLET_SLOTS + 1):
        key = f"favorite_things_{index}"
        if isinstance(raw.get(key), str) and raw[key].strip():
            data[key] = raw[key].strip()
    # Also accept overflow list items so they can be combined into ≤2 slots.
    extra_things: list[str] = []
    if isinstance(things, list) and len(things) > _FAVORITE_BULLET_SLOTS:
        extra_things = [str(value).strip() for value in things[_FAVORITE_BULLET_SLOTS:]]
    # Flat favorite_things_3 from older payloads becomes overflow to combine.
    if isinstance(raw.get("favorite_things_3"), str) and raw["favorite_things_3"].strip():
        extra_things.append(raw["favorite_things_3"].strip())
    _assign_packed_list(
        data,
        prefix="favorite_things",
        count=_FAVORITE_BULLET_SLOTS,
        values=[data[f"favorite_things_{i}"] for i in range(1, _FAVORITE_BULLET_SLOTS + 1)]
        + extra_things,
        sanitize=_sanitize_favorite_thing,
    )
    # Third favorite slot is unused on the template.
    data["favorite_things_3"] = ""

    reinforcers = raw.get("favorite_reinforcers") or raw.get("reinforcers")
    if isinstance(reinforcers, list):
        for index, value in enumerate(reinforcers[:_REINFORCER_BULLET_SLOTS], start=1):
            data[f"reinforcers_{index}"] = str(value).strip()
    for index in range(1, _REINFORCER_BULLET_SLOTS + 1):
        key = f"reinforcers_{index}"
        if isinstance(raw.get(key), str) and raw[key].strip():
            data[key] = raw[key].strip()
    extra_reinforcers: list[str] = []
    if isinstance(reinforcers, list) and len(reinforcers) > _REINFORCER_BULLET_SLOTS:
        extra_reinforcers = [
            str(value).strip() for value in reinforcers[_REINFORCER_BULLET_SLOTS:]
        ]
    # Legacy flat key reinforcers_4 may still arrive from older payloads.
    if isinstance(raw.get("reinforcers_4"), str) and raw["reinforcers_4"].strip():
        extra_reinforcers.append(raw["reinforcers_4"].strip())
    _assign_packed_list(
        data,
        prefix="reinforcers",
        count=_REINFORCER_BULLET_SLOTS,
        values=[data[f"reinforcers_{i}"] for i in range(1, _REINFORCER_BULLET_SLOTS + 1)]
        + extra_reinforcers,
        sanitize=_sanitize_reinforcer,
    )

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

    # Parent/guardian section holds one contact only — never mixed pairs.
    parent_name, parent_phone, parent_email = _sanitize_single_parent_contact(
        data["parent_name"],
        data["parent_phone"],
        data["parent_email"],
    )
    data["parent_name"] = parent_name
    data["parent_phone"] = parent_phone
    data["parent_email"] = parent_email

    # Mobility phrases (e.g. "independent walker") must not land in toileting.
    data["bathroom_needs"] = _sanitize_bathroom_needs(data["bathroom_needs"])
    data["allergies"] = _naturalize_allergy_summary(data["allergies"])
    data["allergies"] = _na_if_blank_or_none(data["allergies"])
    data["bathroom_needs"] = _na_if_blank_or_none(data["bathroom_needs"])
    data["behavioral_management"] = _naturalize_behavioral_summary(
        data["behavioral_management"]
    )

    return data


def _assign_packed_list(
    data: dict[str, str],
    *,
    prefix: str,
    count: int,
    values: list[str],
    sanitize,
) -> None:
    """Keep usable bullets, pack to the front, distribute overflow across slots."""
    packed = [cleaned for value in values if (cleaned := sanitize(value))]
    if len(packed) > count:
        packed = _distribute_items(packed, count)
    for index in range(1, count + 1):
        data[f"{prefix}_{index}"] = packed[index - 1] if index <= len(packed) else ""


def _distribute_items(items: list[str], count: int) -> list[str]:
    """Split items across ``count`` bullets, combining within each (not only the last)."""
    if count <= 0:
        return []
    if len(items) <= count:
        return list(items)
    base, remainder = divmod(len(items), count)
    result: list[str] = []
    index = 0
    for slot in range(count):
        size = base + (1 if slot < remainder else 0)
        chunk = items[index : index + size]
        result.append(", ".join(chunk))
        index += size
    return result


_MULTI_PARENT_SPLIT_RE: Final = re.compile(
    r"\s+(?:and|&)\s+|\s*/\s*|\s*\|\s*|\s*;\s*",
    flags=re.IGNORECASE,
)
_MULTI_CONTACT_SPLIT_RE: Final = re.compile(
    r"\s+(?:and|&|or)\s+|\s*/\s*|\s*\|\s*|\s*;\s*",
    flags=re.IGNORECASE,
)
_PHONE_TOKEN_RE: Final = re.compile(
    r"(?:\+?1[\s\-.]*)?(?:\(?\d{3}\)?[\s\-.]*)?\d{3}[\s\-.]*\d{4}",
)
_EMAIL_TOKEN_RE: Final = re.compile(
    r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}",
    flags=re.IGNORECASE,
)


def _first_parent_name(value: str) -> str:
    """Keep a single guardian name when several are joined in one field."""
    cleaned = (value or "").strip(" \t\r\n,;.-")
    if not cleaned:
        return ""
    # Strip role labels like "Mother:" / "Parent/Guardian 1 -" before splitting.
    cleaned = re.sub(
        r"^(?:parent(?:/guardian)?|guardian|mother|father|mom|dad|"
        r"primary\s+contact)\s*(?:#?\s*\d+)?\s*[:\-–—]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    # "Last, First" is one person — only split when both sides look like full names.
    if "," in cleaned and not re.search(r"\band\b|&|/|\|", cleaned, flags=re.IGNORECASE):
        left, right = (part.strip() for part in cleaned.split(",", 1))
        # Two full names: "Jane Smith, John Doe". One person: "Smith, Jane".
        if left and right and len(left.split()) >= 2 and len(right.split()) >= 2:
            return left
        return cleaned
    parts = [
        part.strip(" \t\r\n,;.-")
        for part in _MULTI_PARENT_SPLIT_RE.split(cleaned)
        if part.strip(" \t\r\n,;.-")
    ]
    if not parts:
        return ""
    first = parts[0]
    # Drop a leftover role label on the chosen piece.
    first = re.sub(
        r"^(?:parent(?:/guardian)?|guardian|mother|father|mom|dad|"
        r"primary\s+contact)\s*(?:#?\s*\d+)?\s*[:\-–—]\s*",
        "",
        first,
        flags=re.IGNORECASE,
    ).strip()
    return first


def _first_phone(value: str) -> str:
    """Keep a single phone number when several are listed."""
    cleaned = (value or "").strip()
    if not cleaned:
        return ""
    phones = _PHONE_TOKEN_RE.findall(cleaned)
    if phones:
        return re.sub(r"\s+", " ", phones[0]).strip(" \t\r\n,;.-")
    parts = [
        part.strip(" \t\r\n,;.-")
        for part in _MULTI_CONTACT_SPLIT_RE.split(cleaned)
        if part.strip(" \t\r\n,;.-")
    ]
    return parts[0] if parts else cleaned


def _first_email(value: str) -> str:
    """Keep a single email when several are listed."""
    cleaned = (value or "").strip()
    if not cleaned:
        return ""
    emails = _EMAIL_TOKEN_RE.findall(cleaned)
    if emails:
        return emails[0].strip(" \t\r\n,;.-")
    parts = [
        part.strip(" \t\r\n,;.-")
        for part in _MULTI_CONTACT_SPLIT_RE.split(cleaned)
        if part.strip(" \t\r\n,;.-")
    ]
    return parts[0] if parts else cleaned


def _sanitize_single_parent_contact(
    name: str, phone: str, email: str
) -> tuple[str, str, str]:
    """Collapse multi-guardian contact blobs into one matched name/phone/email."""
    return _first_parent_name(name), _first_phone(phone), _first_email(email)


def _is_none_like_answer(text: str) -> bool:
    """True for blank / none / n/a style answers, including 'none at the moment'."""
    cleaned = (text or "").strip().strip(" \t\r\n\"'`")
    if not cleaned:
        return True
    return bool(
        re.fullmatch(
            r"(?:"
            r"n/?a|nil|nill|none|no|nothing|empty|not\s+applicable|"
            r"none\s+at\s+(?:the\s+)?(?:moment|time|present)|"
            r"none\s+at\s+this\s+time|"
            r"none\s+currently|none\s+for\s+now|none\s+right\s+now|"
            r"nothing\s+at\s+(?:the\s+)?(?:moment|time|present)|"
            r"nothing\s+at\s+this\s+time|"
            r"nothing\s+currently|nothing\s+for\s+now|"
            r"not\s+at\s+(?:the\s+)?(?:moment|time|present)|"
            r"not\s+at\s+this\s+time|"
            r"no\s+(?:known\s+)?(?:concerns?|issues?|challenges?|problems?|notes?)|"
            r"n/?a\s+at\s+(?:the\s+)?(?:moment|time|present)|"
            r"n/?a\s+at\s+this\s+time|"
            r"currently\s+none|"
            r"-\s*|—|–"
            r")(?:\s*[.!]*)?",
            cleaned,
            flags=re.IGNORECASE,
        )
    )


def _is_placeholder_bullet(text: str) -> bool:
    return _is_none_like_answer(text)


def _looks_like_diagnosis(text: str) -> bool:
    """True when a bullet is a diagnosis/disability label, not an interest/reward."""
    return bool(_DIAGNOSIS_BLEED.search(text))


def _sanitize_list_bullet(text: str) -> str:
    """Drop blank / none-style fillers from page-1 bullet lists."""
    cleaned = (text or "").strip(" \t\r\n,;.-")
    if not cleaned or _is_placeholder_bullet(cleaned):
        return ""
    return cleaned


def _sanitize_favorite_thing(text: str) -> str:
    """Keep strengths/interests; drop placeholders and diagnosis bleed."""
    cleaned = _sanitize_list_bullet(text)
    if not cleaned:
        return ""
    if _looks_like_diagnosis(cleaned):
        return ""
    # Bare independence answers are not favorite things.
    if re.fullmatch(
        r"(?:independent(?:ly)?|independence)(?:\s*[.!]*)?",
        cleaned,
        flags=re.IGNORECASE,
    ):
        return ""
    return cleaned


def _sanitize_reinforcer(text: str) -> str:
    """Keep real reinforcers; drop placeholders and off-field bleed."""
    cleaned = _sanitize_list_bullet(text)
    if not cleaned:
        return ""
    if _looks_like_diagnosis(cleaned):
        return ""
    # Bare independence / mobility answers are not reinforcers.
    if re.fullmatch(
        r"(?:independent(?:ly)?|independence|independent\s+walker|"
        r"walker|ambulat(?:e|es|ion|ory)|gait|"
        r"(?:no\s+)?help\s+needed|does\s+not\s+need\s+help)"
        r"(?:\s*[.!]*)?",
        cleaned,
        flags=re.IGNORECASE,
    ):
        return ""
    # Phrases that clearly belong to bathroom / mobility sections.
    if re.search(
        r"\b(?:walker|ambulat(?:e|es|ion|ory)|gait|restroom|bathroom|toileting)\b",
        cleaned,
        flags=re.IGNORECASE,
    ):
        return ""
    return cleaned


def _na_if_blank_or_none(text: str) -> str:
    """Normalize empty / none-style answers to N/A for medical-style fields."""
    if _is_none_like_answer(text):
        return "N/A"
    return text.strip() if text else "N/A"


def _naturalize_allergy_summary(text: str) -> str:
    """Turn awkward Q/A echoes into short natural statements; keep facts only.

    Type-specific negatives avoid contradictions like listing an allergy and then
    saying "no allergies" when only food allergies were marked none.
    """
    if not text or not text.strip():
        return text
    cleaned = text.strip()
    # Order matters: more specific labels before generic "allergies".
    replacements: tuple[tuple[re.Pattern[str], str], ...] = (
        (
            re.compile(
                r"\bis\s+(?:an?\s+)?epi[\s-]?pen\s+provided\s*[:\-–—]?\s*"
                r"(?:none|no|n/?a|not\s+provided)\b",
                flags=re.IGNORECASE,
            ),
            "no epi pen",
        ),
        (
            re.compile(
                r"\b(?:epi[\s-]?pen\s+provided)\s*[:\-–—]?\s*"
                r"(?:none|no|n/?a|not\s+provided)\b",
                flags=re.IGNORECASE,
            ),
            "no epi pen",
        ),
        (
            re.compile(
                r"\bdoes\s+(?:the\s+)?participant\s+have\s+seizures?\s*[:\-–—]?\s*"
                r"(?:none|no|n/?a)\b",
                flags=re.IGNORECASE,
            ),
            "no seizures",
        ),
        (
            re.compile(
                r"\bseizures?\s*[:\-–—]?\s*(?:none|no|n/?a)\b",
                flags=re.IGNORECASE,
            ),
            "no seizures",
        ),
        (
            re.compile(
                r"\b(?:medications?(?:\s+taken)?(?:\s+during\s+camp(?:\s+hours)?)?)"
                r"\s*[:\-–—]?\s*(?:none|no|n/?a)\b",
                flags=re.IGNORECASE,
            ),
            "no medications during camp",
        ),
        (
            re.compile(
                r"\b(?:participant\s+)?food\s+allergies?(?:/dietary\s+restrictions)?"
                r"(?:\s*/\s*dietary\s+restrictions)?\s*[:\-–—]?\s*"
                r"(?:none|no|n/?a)\b",
                flags=re.IGNORECASE,
            ),
            "no food allergies",
        ),
        (
            re.compile(
                r"\b(?:participant\s+)?allergies?"
                r"(?:\s*\(please\s+include\s+all\))?\s*[:\-–—]?\s*"
                r"(?:none|no|n/?a)\b",
                flags=re.IGNORECASE,
            ),
            "no allergies",
        ),
    )
    for pattern, replacement in replacements:
        cleaned = pattern.sub(replacement, cleaned)

    cleaned = re.sub(
        r"(?i)\b(?:is\s+)?(?:an?\s+)?epi[\s-]?pen\s+provided\s*[:\-–—]\s*",
        "epi pen: ",
        cleaned,
    )
    # Remaining "label: none" → "no <type>" (keeps the type word from the label).
    cleaned = re.sub(
        r"(?i)([^:;.\n]{2,40}?)\s*[:\-–—]\s*(?:none|no|n/?a)\b",
        lambda m: f"no {m.group(1).strip().rstrip('?').strip().lower()}",
        cleaned,
    )
    # Drop a bare "no allergies" when other allergy facts are already present
    # (e.g. "Latex, no allergies" from a mistyped none food field).
    positive_allergy = bool(
        re.search(
            r"(?i)\ballergic\s+to\b|"
            r"\b(?:peanut|tree\s+nuts?|latex|dairy|gluten|eggs?|soy|shellfish|"
            r"pollen|dust|pet\s+dander|bee\s+sting)s?\b|"
            r"\b(?!no\b)\w+\s+allerg(?:y|ies)\b",
            cleaned,
        )
    )
    if positive_allergy:
        cleaned = re.sub(
            r"(?i)(?:^|[,;.]\s*)\bno allergies\b\.?",
            "",
            cleaned,
        )
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s*,\s*,+", ", ", cleaned)
    cleaned = re.sub(r"\s*([,;])\s*", r"\1 ", cleaned)
    return cleaned.strip(" ,;")


# Form section titles the model sometimes pastes into behavioral_management.
_BEHAVIORAL_QA_LABEL: Final = re.compile(
    r"(?i)\b(?:"
    r"participant'?s?\s+(?:areas?\s+that\s+can\s+be\s+challenging|"
    r"behavioral\s+challenges?|challenging\s+areas?|areas?\s+of\s+challenge)|"
    r"areas?\s+that\s+can\s+be\s+challenging|"
    r"challenging\s+areas?|"
    r"behavioral\s+challenges?|"
    r"strategies?\s+that\s+help\s+with\s+challenges?|"
    r"strategies?\s+that\s+help|"
    r"behaviou?ral\s+strategies?|"
    r"behavorial\s+strategies?"
    r")\s*[:\-–—]\s*"
)


def _naturalize_behavioral_summary(text: str) -> str:
    """Strip Q/A labels; blank the field when the answer is none-like."""
    if not text or not text.strip():
        return ""
    cleaned = text.strip()
    # Remove repeated section headings the model echoes from the intake form.
    cleaned = _BEHAVIORAL_QA_LABEL.sub("", cleaned)
    # Prefer sentence breaks over leftover semicolon / pipe scaffolding.
    cleaned = re.sub(r"\s*[|]\s*", ". ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"(?:\.\s*){2,}", ". ", cleaned)
    cleaned = cleaned.strip(" ,;:-")
    if _is_none_like_answer(cleaned):
        return ""
    # "None at the moment. Offer breaks…" style — if it still leads with none, drop.
    cleaned = re.sub(
        r"(?i)^\s*(?:none|n/?a|nothing)(?:\s+at\s+(?:the\s+)?(?:moment|time))?"
        r"(?:\s*[.!,;:—–-]+\s*|\s+)",
        "",
        cleaned,
        count=1,
    ).strip(" ,;:-")
    if not cleaned or _is_none_like_answer(cleaned):
        return ""
    if cleaned and cleaned[-1] not in ".!?" and len(cleaned.split()) >= 8:
        cleaned += "."
    return cleaned


# Labels used to read behavioral answers back out of OCR / text sources.
_BEHAVIORAL_SOURCE_LABELS: Final[tuple[str, ...]] = (
    r"participant'?s?\s+behavioral\s+challenges?",
    r"participant'?s?\s+areas?\s+that\s+can\s+be\s+challenging",
    r"areas?\s+that\s+can\s+be\s+challenging",
    r"strategies?\s+that\s+help(?:\s+with\s+challenges?)?",
    r"behaviou?ral\s+strategies?",
    r"behavorial\s+strategies?",
)

_FAVORITE_SOURCE_LABELS: Final[tuple[str, ...]] = (
    r"participant'?s?\s+strengths?\s+and\s+favorite\s+interests?",
    r"strengths?\s+and\s+favorite\s+interests?",
)

_REINFORCER_SOURCE_LABELS: Final[tuple[str, ...]] = (
    r"favorite\s+reinforcers?",
)


def _collect_labeled_answers(text: str, label_patterns: tuple[str, ...]) -> list[str]:
    """Return answers that follow any of the given form labels in ``text``."""
    answers: list[str] = []
    for label in label_patterns:
        pattern = re.compile(
            rf"(?im)\b(?:{label})\s*[:\-–—]?\s*([^\n]+)"
        )
        for match in pattern.finditer(text):
            value = match.group(1).strip().strip(" \t\"'`")
            if value:
                answers.append(value)
    return answers


def apply_none_source_guards(form_data: dict[str, str], source_text: str) -> dict[str, str]:
    """Blank fields when the intake text clearly answered none / n/a.

    Prevents the model from inventing content for sections that the form marked
    as none (e.g. "Participant's behavioral challenges: None at the moment").
    """
    data = dict(form_data)
    text = (source_text or "").strip()
    if not text:
        return data

    behavioral_answers = _collect_labeled_answers(text, _BEHAVIORAL_SOURCE_LABELS)
    real_behavioral = [a for a in behavioral_answers if not _is_none_like_answer(a)]
    if behavioral_answers and not real_behavioral:
        data["behavioral_management"] = ""

    favorite_answers = _collect_labeled_answers(text, _FAVORITE_SOURCE_LABELS)
    if favorite_answers and all(_is_none_like_answer(a) for a in favorite_answers):
        data["favorite_things_1"] = ""
        data["favorite_things_2"] = ""
        data["favorite_things_3"] = ""

    reinforcer_answers = _collect_labeled_answers(text, _REINFORCER_SOURCE_LABELS)
    if reinforcer_answers and all(_is_none_like_answer(a) for a in reinforcer_answers):
        data["reinforcers_1"] = ""
        data["reinforcers_2"] = ""
        data["reinforcers_3"] = ""

    return data


# Back-compat alias used by older tests/imports.
def _naturalize_medical_summary(text: str) -> str:
    return _naturalize_allergy_summary(text)


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
) -> None:
    x0, y0, x1, y1 = box
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
    """Draw a bullet and text for a filled row; skip empty rows.

    Draws directly on the template art (no white rectangle behind text).
    Text is shrunk, then truncated if needed, so it stays inside the box.
    """
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

    # Still too wide at the floor size — truncate so we never spill past the box.
    display = text
    if draw.textbbox((0, 0), display, font=font)[2] > max_width:
        ellipsis = "…"
        while display and draw.textbbox((0, 0), display + ellipsis, font=font)[2] > max_width:
            display = display[:-1]
        display = (display.rstrip(" ,;:-") + ellipsis) if display else ellipsis

    bbox = draw.textbbox((0, 0), display, font=font)
    text_h = bbox[3] - bbox[1]
    y = cy - text_h // 2 - bbox[1]
    draw.text((text_x, y), display, fill=(0, 0, 0), font=font)


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


def merge_profiles_pdf(profiles: list[dict]) -> bytes:
    """Concatenate every filled template into one multi-page PDF."""
    writer = PdfWriter()
    for profile in profiles:
        reader = PdfReader(io.BytesIO(profile["pdf_bytes"]))
        for page in reader.pages:
            writer.add_page(page)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


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
    )
    for index in range(1, _FAVORITE_BULLET_SLOTS + 1):
        key = f"favorite_things_{index}"
        _draw_bullet_line(
            draw1,
            size,
            _PAGE1_BOXES[key],
            data[key],
            font_size=_FONT_BULLET,
            min_font_size=_FONT_MIN_BULLET,
        )
    for index in range(1, _REINFORCER_BULLET_SLOTS + 1):
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
