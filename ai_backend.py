"""Ollama-powered conversion of participant details into All About Me profiles."""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Final

from pdf_filler import fill_form_pdf, normalize_form_data

PROJECT_ROOT: Final = Path(__file__).resolve().parent

# Local Ollama native API. No cloud API key required.
DEFAULT_BASE_URL: Final = "http://127.0.0.1:11434"
DEFAULT_MODEL: Final = "llama3.2:latest"
DEFAULT_VISION_MODEL: Final = "llava:7b"

SYSTEM_PROMPT: Final = """You extract participant facts for an All About Me PDF form.
Return ONLY valid JSON (no markdown fences, no commentary) with this shape:
{
  "name": "",
  "favorite_things": ["", "", ""],
  "favorite_reinforcers": ["", "", "", ""],
  "parent_name": "",
  "parent_phone": "",
  "parent_email": "",
  "allergies": "",
  "bathroom_needs": "",
  "behavioral_management": ""
}

SOURCE → TEMPLATE mapping (use ONLY these input-form labels; ignore other sections):
- name: participant name fields only.
- favorite_things: ONLY "Participant's strengths and favorite interests".
  Split into up to 3 short list items. Empty strings if none.
- favorite_reinforcers: ONLY "Favorite reinforcers".
  Split into up to 4 short list items. Empty strings if none.
- parent_name / parent_phone / parent_email: parent/guardian contact fields only.
  Single-line values.
- allergies: ONLY combine these four source areas into one concise summary:
  1) "Medications, if taken during camp hours"
  2) "Does participant have seizures?"
  3) "Participant allergies(Please include all):"
  4) "Participant food allergies/dietary restrictions:"
  If all are blank/none/no, return exactly "N/A".
- bathroom_needs: ONLY "Does the participant need help in the restroom?".
  If blank/none/no help needed, return exactly "N/A".
  Toileting only — never mobility wording (walker, gait, ambulation).
- behavioral_management: ONLY combine:
  1) "Participant's areas that can be challenging"
  2) "Strategies that help with challenges"
  Concise paragraph (roughly 1–4 sentences). Empty string if neither has content.

Rules:
- Use only facts from the mapped source areas above. Never invent or borrow from other fields.
- Light typo cleanup is allowed; do not change meaning.
- Page 1 list items stay short (a few words each).
- Page 2 body fields may be short sentences when the source supports it.
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


def _chat(model: str, messages: list[dict[str, object]]) -> str:
    """Call Ollama's /api/chat endpoint and return the assistant text."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_predict": 900},
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


def _parse_form_json(raw: str) -> dict[str, str]:
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
    return normalize_form_data(parsed)


def form_data_to_markdown(form_data: dict[str, str]) -> str:
    """Build a readable All About Me Markdown profile from extracted fields."""
    things = [
        form_data.get(f"favorite_things_{index}", "").strip()
        for index in range(1, 4)
        if form_data.get(f"favorite_things_{index}", "").strip()
    ]
    reinforcers = [
        form_data.get(f"reinforcers_{index}", "").strip()
        for index in range(1, 5)
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
) -> dict[str, str]:
    """Run Ollama extraction and return normalized form fields."""
    text = raw_text.strip() if raw_text else ""
    has_image = bool(image_bytes)
    if not text and not has_image:
        raise ValueError("Provide raw_text, image_bytes, or both.")
    if has_image and not image_mime_type.startswith("image/"):
        raise ValueError("image_mime_type must be an image MIME type.")

    prompt_text = (
        "Map the labeled input-form answers into the All About Me JSON "
        "using only the SOURCE → TEMPLATE rules from the system prompt. "
        "Return JSON only.\n\n"
        f"TEXT SOURCE:\n{text or '[No text source was supplied.]'}"
    )

    user_message: dict[str, object] = {"role": "user", "content": prompt_text}
    if has_image:
        # Ollama vision models expect raw base64 image strings (no data: URL).
        user_message["images"] = [base64.b64encode(image_bytes).decode("ascii")]
        model = _vision_model()
    else:
        model = _text_model()

    model_text = _chat(
        model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            user_message,
        ],
    )
    form_data = _parse_form_json(model_text)
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
) -> tuple[str, bytes]:
    """Return Markdown preview text and a filled formTemplate.pdf.

    Talks to a local Ollama server (default ``http://127.0.0.1:11434``).
    Text uploads use ``OLLAMA_MODEL`` (default ``llama3.2:latest``).
    Image uploads use ``OLLAMA_VISION_MODEL`` (default ``llava:7b``).
    """
    form_data = extract_form_data(
        raw_text,
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
    )
    return form_data_to_markdown(form_data), fill_form_pdf(form_data)


def generate_all_about_me_pdf(
    raw_text: str | None = None,
    *,
    image_bytes: bytes | None = None,
    image_mime_type: str = "image/png",
) -> bytes:
    """Return only the filled PDF (convenience wrapper)."""
    _, pdf_bytes = generate_all_about_me_profile(
        raw_text,
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
    )
    return pdf_bytes
