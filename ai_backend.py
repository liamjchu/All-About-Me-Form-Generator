"""Ollama-powered conversion of participant details into All About Me profiles."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Final, Sequence

from PIL import Image

from file_inputs import FooterMark, extract_footer_mark_ocr, normalize_date_line
from pdf_filler import (
    apply_none_source_guards,
    fill_form_pdf,
    normalize_form_data,
    participant_first_name_from_text,
)

PROJECT_ROOT: Final = Path(__file__).resolve().parent

# Local Ollama native API. No cloud API key required.
DEFAULT_BASE_URL: Final = "http://127.0.0.1:11434"
DEFAULT_MODEL: Final = "llama3.2:latest"
DEFAULT_VISION_MODEL: Final = "qwen2.5vl:7b"
# Keep context modest — the default 128k window on qwen2.5vl is far larger
# than one/two form photos need and slows every token.
# 4k context is enough for one/two form photos + the system prompt.
DEFAULT_NUM_CTX: Final = 4096
DEFAULT_NUM_PREDICT: Final = 500
VISION_JPEG_QUALITY: Final = 75

SYSTEM_PROMPT: Final = """You extract participant facts for an All About Me PDF form.
Return ONLY valid JSON (no markdown fences, no commentary) with this shape:
{
  "name": "",
  "favorite_things": ["", ""],
  "favorite_reinforcers": ["", "", ""],
  "parent_name": "",
  "parent_phone": "",
  "parent_email": "",
  "allergies": "",
  "bathroom_needs": "",
  "behavioral_management": ""
}

CRITICAL — no hallucination:
- Copy facts ONLY from the designated SOURCE labels below on THIS form.
- If a source answer is blank, illegible, "none", "none at the moment",
  "nothing at this time", "n/a", "no", or similar, that field has NO usable
  content — leave the JSON field empty ("" / ["",""]) or use "N/A" only where
  noted. Do NOT invent a substitute answer.
- NEVER invent hobbies, interests, reinforcers, names, contacts, allergies,
  bathroom needs, strategies, behavioral tips, or any other values.
- NEVER use example values from this prompt. NEVER fill gaps with guesses,
  stereotypes, or "typical" answers.
- Leaving a field blank is always better than making something up.

SOURCE → TEMPLATE mapping (use ONLY these input-form labels; ignore other sections):
- name: ONLY the participant first name from the top-left header line that looks
  like: "submission ID# nnnnnn For: Lastname, Firstname | DOB: m/dd/yyyy ...".
  Use ONLY the Firstname after "For:" (the given name after the comma).
  Do NOT use last name. Do NOT use parent/guardian names. Do NOT pull a name
  from any other section of the form (signature, contact, body answers, etc.).
  If that For: header cannot be read, leave name as "".
- favorite_things: ONLY the answer written under
  "Participant's strengths and favorite interests".
  Return at most 2 short list items. If there are many interests, summarize or
  keep the most notable ones, and combine related items onto both lines
  (e.g. "a, b" and "c, d") so they fit in 2 bullets — do not dump overflow
  onto only the last bullet. Unused slots must be "".
  If that source is blank / "none" / "n/a" / "no" / illegible, return ["", ""].
  NEVER copy diagnoses, disabilities, medical labels, or other sections
  (autism, ASD, ID, intellectual disability, ADHD, Down syndrome, etc.).
  Never pad with "none", "n/a", "no", or "independent".
- favorite_reinforcers: ONLY the answer written under "Favorite reinforcers"
  (rewards / motivators written on the form).
  Split into up to 3 short list items. Combine onto fewer lines if needed,
  distributing across bullets rather than only the last one.
  Unused slots must be "". If that source is blank / "none" / "n/a" / "no" /
  illegible, return ["", "", ""].
  Never invent items. Never copy words from other sections (bathroom, strengths,
  mobility, independence, diagnoses, disabilities). Never pad with "none",
  "n/a", "no", or "independent".
- parent_name / parent_phone / parent_email: parent/guardian contact fields only.
  Single-line values. Empty string if that contact line is blank.
- allergies: ONLY combine these source areas into one concise natural summary:
  1) "Medications, if taken during camp hours"
  2) "Does participant have seizures?"
  3) "Participant allergies(Please include all):"
  4) "Participant food allergies/dietary restrictions:"
  5) Any epi-pen / emergency medication question on the form
  Write flowing short statements (e.g. "no epi pen", "no seizures",
  "allergic to peanuts", "no food allergies"), NOT question/answer echoes like
  "Is epi pen provided: none". When one allergy type is none but another has
  content, say the type explicitly ("no food allergies") — never a bare
  "no allergies" that contradicts listed allergies. Never invent details.
  If all are blank/none/no, return exactly "N/A".
- bathroom_needs: ONLY "Does the participant need help in the restroom?".
  If blank/none/no help needed, return exactly "N/A".
  Toileting only — never mobility wording (walker, gait, ambulation).
- behavioral_management: ONLY combine answers from these source areas (any that
  appear on the form; accept near-matching label wording):
  1) "Participant's areas that can be challenging"
  2) "Strategies that help with challenges"
  3) "Behavioral strategies" / "Behavioral Strategies" / "Behavorial Strategies"
  Also accept near-matches like "Participant's behavioral challenges".
  If EVERY present source above is blank / "none" / "none at the moment" /
  "n/a" / "no" / illegible, return exactly "" — do NOT invent challenges or
  strategies.
  When there IS real written content (not none-like), rewrite into a brief
  natural summary (about 1–3 short sentences). Keep only key challenges and
  helpful strategies. Do NOT paste section titles, headings, or "label: value"
  / question/answer scaffolding (never "Participant's behavioral challenges:
  ..."). Paraphrase into flowing prose — be consistent across forms.

Rules:
- Use only facts from the mapped source areas above. Never invent or borrow from
  other fields. Diagnoses / disabilities never belong in favorite_things or
  favorite_reinforcers — leave those lists blank rather than substitute.
- Printed and handwritten answers both count. Read carefully through glare, shadow,
  or slight blur when the text is still legible.
- If a printed label is hard to read but the filled answer clearly sits in that
  field's usual place on the form, still extract it into the matching JSON key.
- Light typo cleanup is allowed; do not change meaning.
- Partial extraction is fine: fill only fields whose source text you can actually
  read. Do not invent values for unreadable or blank fields.
- Page 1 list items stay short (a few words each).
- Page 2 body fields may be short sentences when the source supports it.
- When multiple images are provided, they are consecutive pages of ONE participant
  form — merge facts across pages into a single JSON object.
"""

FOOTER_PROMPT: Final = """Read only the printed footer in this image crop.
Bottom-left usually has a timestamp like "Jun 30 2026 1:21PM ET".
Bottom-right usually has a page mark like "45 of 85" or "45/85".
Return ONLY JSON with this shape:
{"date_line": "Jun 30 2026 1:21PM ET", "page": 45, "total": 85}
Copy date_line exactly as printed. Use null for any field you cannot read.
"""


def _env_or_dotenv(name: str, default: str | None = None) -> str | None:
    """Return a setting from the environment or this project's .env file."""
    value = os.getenv(name)
    if value:
        return value.strip()

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, raw_value = line.partition("=")
            if key.strip() == name and separator:
                parsed = raw_value.strip().strip("'\"")
                if parsed:
                    return parsed

    return default


def _base_url() -> str:
    # Accept either the native root or a leftover OpenAI-compatible /v1 URL.
    url = (_env_or_dotenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL).rstrip(
        "/"
    )
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def _text_model() -> str:
    return _env_or_dotenv("OLLAMA_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL


def _vision_model() -> str:
    return (
        _env_or_dotenv("OLLAMA_VISION_MODEL", DEFAULT_VISION_MODEL)
        or DEFAULT_VISION_MODEL
    )


def _chat(
    model: str,
    messages: list[dict[str, object]],
    *,
    num_predict: int = DEFAULT_NUM_PREDICT,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> str:
    """Call Ollama's /api/chat endpoint and return the assistant text."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }
    request = urllib.request.Request(
        f"{_base_url()}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not reach Ollama at {_base_url()}. Is it running "
            f"(`ollama serve`)? Details: {error}"
        ) from error
    except TimeoutError as error:
        raise RuntimeError(
            f"Ollama timed out for model '{model}'. Try a smaller model or "
            "fewer/simpler uploads."
        ) from error

    content = (body.get("message") or {}).get("content", "").strip()
    if not content:
        raise RuntimeError(
            f"The model '{model}' returned an empty response. Confirm it is "
            f"pulled with `ollama pull {model}`."
        )
    return content


def _parse_json_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise RuntimeError(
                "The model did not return usable JSON for the form."
            ) from error
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise RuntimeError("The model JSON was not an object.")
    return parsed


def _parse_form_json(raw: str) -> dict[str, str]:
    return normalize_form_data(_parse_json_object(raw))


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        match = re.search(r"\d+", cleaned)
        if match:
            return int(match.group(0))
    return None


def _image_to_jpeg_b64(image_bytes: bytes) -> str:
    """Re-encode as compact JPEG so vision requests stay small/fast."""
    with Image.open(io.BytesIO(image_bytes)) as opened:
        image = opened.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=VISION_JPEG_QUALITY, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _footer_vision_fallback_enabled() -> bool:
    """Opt-in only — vision footer reads add minutes per image on laptop GPUs."""
    raw = (_env_or_dotenv("FOOTER_VISION_FALLBACK", "0") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def extract_footer_mark(
    footer_image_bytes: bytes,
    *,
    image_mime_type: str = "image/png",
) -> FooterMark:
    """Read bottom-left date and bottom-right page mark from a footer crop.

    Uses local Tesseract OCR (seconds). Vision fallback is off by default
    because a failed OCR on a 30-image batch would otherwise add ~30 slow
    Ollama calls before generation even starts. Set FOOTER_VISION_FALLBACK=1
    to re-enable it.
    """
    if not footer_image_bytes:
        return FooterMark()
    if not image_mime_type.startswith("image/"):
        raise ValueError("image_mime_type must be an image MIME type.")

    ocr_mark = extract_footer_mark_ocr(footer_image_bytes)
    if ocr_mark.page is not None or ocr_mark.date_line:
        return ocr_mark

    if not _footer_vision_fallback_enabled():
        return FooterMark()

    raw = _chat(
        _vision_model(),
        [
            {"role": "system", "content": FOOTER_PROMPT},
            {
                "role": "user",
                "content": "Return the footer JSON only.",
                "images": [_image_to_jpeg_b64(footer_image_bytes)],
            },
        ],
        num_predict=80,
        num_ctx=2048,
    )
    try:
        parsed = _parse_json_object(raw)
    except (RuntimeError, json.JSONDecodeError):
        return FooterMark()

    date_raw = parsed.get("date_line")
    date_line = normalize_date_line(date_raw if isinstance(date_raw, str) else None)
    return FooterMark(
        date_line=date_line,
        page=_optional_int(parsed.get("page")),
        total=_optional_int(parsed.get("total")),
    )


def form_data_to_markdown(form_data: dict[str, str]) -> str:
    """Build a readable All About Me Markdown profile from extracted fields."""
    things = [
        form_data.get(f"favorite_things_{index}", "").strip()
        for index in range(1, 3)
        if form_data.get(f"favorite_things_{index}", "").strip()
    ]
    reinforcers = [
        form_data.get(f"reinforcers_{index}", "").strip()
        for index in range(1, 4)
        if form_data.get(f"reinforcers_{index}", "").strip()
    ]

    def bullets(items: list[str]) -> str:
        if not items:
            return "- "
        return "\n".join(f"- {item}" for item in items)

    return "\n".join(
        [
            "# All About Me",
            "",
            f"**Name:** {form_data.get('name', '')}",
            "",
            "## My Favorite Things",
            bullets(things),
            "",
            "## Favorite Reinforcers",
            bullets(reinforcers),
            "",
            "## Parent/Guardian",
            f"- **Name:** {form_data.get('parent_name', '')}",
            f"- **Phone:** {form_data.get('parent_phone', '')}",
            f"- **Email:** {form_data.get('parent_email', '')}",
            "",
            "## Allergies/Medical Needs",
            form_data.get("allergies", "") or "N/A",
            "",
            "## Bathroom Needs",
            form_data.get("bathroom_needs", "") or "N/A",
            "",
            "## Behavioral Management",
            form_data.get("behavioral_management", ""),
            "",
        ]
    )


def extract_form_data(
    raw_text: str | None = None,
    *,
    image_bytes: bytes | None = None,
    image_mime_type: str = "image/png",
    images: Sequence[tuple[bytes, str]] | None = None,
) -> dict[str, str]:
    """Run Ollama extraction and return normalized form fields.

    ``images`` is a sequence of ``(image_bytes, mime_type)`` for multi-page
    photo uploads. ``image_bytes`` remains a single-image convenience.
    """
    text = raw_text.strip() if raw_text else ""
    image_list: list[tuple[bytes, str]] = []
    if images:
        image_list.extend(images)
    elif image_bytes:
        image_list.append((image_bytes, image_mime_type))

    if not text and not image_list:
        raise ValueError("Provide raw_text, image_bytes, or images.")
    for _, mime in image_list:
        if not mime.startswith("image/"):
            raise ValueError("image_mime_type must be an image MIME type.")

    page_note = ""
    if len(image_list) > 1:
        page_note = (
            f" There are {len(image_list)} page images for one participant; "
            "merge facts across all of them."
        )

    if image_list:
        prompt_text = (
            "These are photo(s) of a filled participant intake form. "
            "Read printed and handwritten answers carefully, then map them "
            "into the All About Me JSON using the SOURCE → TEMPLATE rules. "
            "Copy only text that actually appears in each designated source "
            "slot. If a slot is blank, n/a, none, 'none at the moment', or "
            "unreadable, leave that JSON field empty (or N/A where the rules "
            "say). Do not invent or guess any values. Return JSON only."
            f"{page_note}"
        )
        if text:
            prompt_text += f"\n\nADDITIONAL TEXT SOURCE:\n{text}"
    else:
        prompt_text = (
            "Map the labeled input-form answers into the All About Me JSON "
            "using only the SOURCE → TEMPLATE rules from the system prompt. "
            "Copy only facts present in the designated source slots. "
            "Do not invent or guess. Return JSON only.\n\n"
            f"TEXT SOURCE:\n{text or '[No text source was supplied.]'}"
        )

    user_message: dict[str, object] = {"role": "user", "content": prompt_text}
    if image_list:
        # Ollama vision models expect raw base64 image strings (no data: URL).
        user_message["images"] = [_image_to_jpeg_b64(data) for data, _ in image_list]
        model = _vision_model()
    else:
        model = _text_model()

    model_text = _chat(
        model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            user_message,
        ],
        num_ctx=DEFAULT_NUM_CTX if image_list else 4096,
        num_predict=DEFAULT_NUM_PREDICT,
    )
    form_data = _parse_form_json(model_text)
    # Prefer the intake "For: Last, First" header when it appears in text sources.
    header_first = participant_first_name_from_text(text) if text else ""
    if header_first:
        form_data["name"] = header_first
    # If the form text says none / n/a for a section, never keep invented content.
    if text:
        form_data = apply_none_source_guards(form_data, text)
    if not form_data.get("name") and not any(
        form_data.get(key)
        for key in form_data
        if key != "name" and form_data.get(key) not in ("", "N/A")
    ):
        raise RuntimeError("The model could not find usable participant details.")
    return form_data


def generate_all_about_me_profile(
    raw_text: str | None = None,
    *,
    image_bytes: bytes | None = None,
    image_mime_type: str = "image/png",
    images: Sequence[tuple[bytes, str]] | None = None,
) -> tuple[str, bytes]:
    """Return Markdown preview text and a filled formTemplate.pdf.

    Talks to a local Ollama server (default ``http://127.0.0.1:11434``).
    Text uploads use ``OLLAMA_MODEL`` (default ``llama3.2:latest``).
    Image uploads use ``OLLAMA_VISION_MODEL`` (default ``qwen2.5vl:7b``).
    """
    form_data = extract_form_data(
        raw_text,
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
        images=images,
    )
    return form_data_to_markdown(form_data), fill_form_pdf(form_data)


def generate_all_about_me_pdf(
    raw_text: str | None = None,
    *,
    image_bytes: bytes | None = None,
    image_mime_type: str = "image/png",
    images: Sequence[tuple[bytes, str]] | None = None,
) -> bytes:
    """Return only the filled PDF (convenience wrapper)."""
    _, pdf_bytes = generate_all_about_me_profile(
        raw_text,
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
        images=images,
    )
    return pdf_bytes
