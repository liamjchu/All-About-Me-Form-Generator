"""OpenAI-powered conversion of participant details into an All About Me form."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Final

from openai import OpenAI

MODEL: Final = "gpt-4o-mini"
PROJECT_ROOT: Final = Path(__file__).resolve().parent
TEMPLATE_PATH: Final = PROJECT_ROOT / "all_about_me_template.md"

SYSTEM_PROMPT: Final = """You turn participant information into an All About Me form.
Return only a completed Markdown document based on the supplied template.
Keep every heading, label, and section in the same order as the template.
Use only facts explicitly provided in the source text or visible in the image.
Never infer, embellish, or make up a fact. Leave a value blank when it was not
provided; use N/A only when the field is clearly inapplicable. Preserve the
friendly, simple wording of the template and do not wrap the result in a code
fence or add commentary.
"""


def _get_api_key() -> str:
    """Return the API key from the environment or this project's .env file."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            name, separator, value = line.partition("=")
            if name.strip() == "OPENAI_API_KEY" and separator:
                api_key = value.strip().strip("'\"")
                if api_key:
                    return api_key

    raise RuntimeError(
        "OPENAI_API_KEY is required. Add it to this project's .env file before "
        "generating profiles."
    )


def _read_template() -> str:
    if not TEMPLATE_PATH.exists():
        raise RuntimeError(f"Markdown template not found: {TEMPLATE_PATH.name}")
    return TEMPLATE_PATH.read_text(encoding="utf-8").strip()


def generate_all_about_me_markdown(
    raw_text: str | None = None,
    *,
    image_bytes: bytes | None = None,
    image_mime_type: str = "image/png",
) -> str:
    """Return the root Markdown template filled from text and/or a text image.

    An ``OPENAI_API_KEY`` must be present in the process environment or in the
    project-root ``.env`` file. Image input is sent directly to GPT-4o mini,
    which reads the text in the image before filling the form.
    """
    text = raw_text.strip() if raw_text else ""
    has_image = bool(image_bytes)
    if not text and not has_image:
        raise ValueError("Provide raw_text, image_bytes, or both.")
    if has_image and not image_mime_type.startswith("image/"):
        raise ValueError("image_mime_type must be an image MIME type.")

    content: list[dict[str, str]] = [
        {
            "type": "input_text",
            "text": (
                "Fill this Markdown template using the following participant "
                f"information.\n\nTEMPLATE:\n{_read_template()}\n\n"
                f"TEXT SOURCE:\n{text or '[No text source was supplied.]'}"
            ),
        }
    ]
    if has_image:
        encoded_image = base64.b64encode(image_bytes).decode("ascii")
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{image_mime_type};base64,{encoded_image}",
            }
        )

    response = OpenAI(api_key=_get_api_key()).responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=[{"role": "user", "content": content}],
        max_output_tokens=1200,
    )
    markdown = response.output_text.strip()
    if not markdown:
        raise RuntimeError("The model returned an empty profile.")
    return markdown
