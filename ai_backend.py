"""Ollama-powered conversion of participant details into All About Me profiles."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Final

from pdf_filler import (
    apply_none_source_guards,
    fill_form_pdf,
    normalize_form_data,
    participant_first_name_from_text,
)
from source_minimize import minimize_source_text_for_llm

PROJECT_ROOT: Final = Path(__file__).resolve().parent

# Local Ollama native API. No cloud API key required.
DEFAULT_BASE_URL: Final = "http://127.0.0.1:11434"
DEFAULT_MODEL: Final = "llama3.2:latest"
DEFAULT_NUM_CTX: Final = 4096
DEFAULT_NUM_PREDICT: Final = 500

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
  Use EXACTLY ONE parent/guardian — never two names, never "A and B", "A & B",
  "A / B", or comma-joined pairs. Prefer Parent/Guardian 1, Primary Contact, or
  the first fully listed guardian when several appear.
  parent_phone and parent_email MUST belong to that SAME chosen person only —
  do not mix one guardian's name with another guardian's phone or email.
  One phone number and one email address max. Empty string if that contact
  line is blank.
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
- Light typo cleanup is allowed; do not change meaning.
- Partial extraction is fine: fill only fields whose source text you can actually
  read. Do not invent values for unreadable or blank fields.
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


def _is_loopback_host(host: str) -> bool:
    """True when host is localhost / loopback (IPv4 or IPv6)."""
    normalized = (host or "").strip().lower().strip("[]")
    if normalized in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _require_loopback_base_url(url: str) -> str:
    """Reject non-local Ollama endpoints so intake text cannot leave this machine."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError(
            f"OLLAMA_BASE_URL must be http(s); got scheme {parsed.scheme!r}."
        )
    host = parsed.hostname or ""
    if not _is_loopback_host(host):
        raise RuntimeError(
            "OLLAMA_BASE_URL must point at a loopback address "
            f"(127.0.0.1 / localhost / ::1), not {host!r}. "
            "This keeps participant text on this machine."
        )
    return url


def _base_url() -> str:
    # Accept either the native root or a leftover OpenAI-compatible /v1 URL.
    url = (_env_or_dotenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL).rstrip(
        "/"
    )
    if url.endswith("/v1"):
        url = url[:-3]
    return _require_loopback_base_url(url)


def _text_model() -> str:
    return _env_or_dotenv("OLLAMA_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL


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


def extract_form_data(raw_text: str | None = None) -> dict[str, str]:
    """Run Ollama extraction on PDF text and return normalized form fields."""
    text = raw_text.strip() if raw_text else ""
    if not text:
        raise ValueError("Provide raw_text extracted from a PDF.")

    # Full text stays local for name override / none-guards; Ollama only sees
    # mapped sections with participant last name / DOB / submission IDs redacted.
    llm_text = minimize_source_text_for_llm(text)
    prompt_text = (
        "Map the labeled input-form answers into the All About Me JSON "
        "using only the SOURCE → TEMPLATE rules from the system prompt. "
        "Copy only facts present in the designated source slots. "
        "Do not invent or guess. Return JSON only.\n\n"
        f"TEXT SOURCE:\n{llm_text}"
    )

    model_text = _chat(
        _text_model(),
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        num_ctx=DEFAULT_NUM_CTX,
        num_predict=DEFAULT_NUM_PREDICT,
    )
    form_data = _parse_form_json(model_text)
    # Prefer the intake "For: Last, First" header when it appears in text sources.
    header_first = participant_first_name_from_text(text)
    if header_first:
        form_data["name"] = header_first
    # If the form text says none / n/a for a section, never keep invented content.
    form_data = apply_none_source_guards(form_data, text)
    if not form_data.get("name") and not any(
        form_data.get(key)
        for key in form_data
        if key != "name" and form_data.get(key) not in ("", "N/A")
    ):
        raise RuntimeError("The model could not find usable participant details.")
    return form_data


def generate_all_about_me_profile(raw_text: str | None = None) -> tuple[str, bytes]:
    """Return Markdown preview text and a filled formTemplate.pdf.

    Talks to a local Ollama server (default ``http://127.0.0.1:11434``).
    PDF text uploads use ``OLLAMA_MODEL`` (default ``llama3.2:latest``).
    """
    form_data = extract_form_data(raw_text)
    return form_data_to_markdown(form_data), fill_form_pdf(form_data)


def generate_all_about_me_pdf(raw_text: str | None = None) -> bytes:
    """Return only the filled PDF (convenience wrapper)."""
    _, pdf_bytes = generate_all_about_me_profile(raw_text)
    return pdf_bytes
