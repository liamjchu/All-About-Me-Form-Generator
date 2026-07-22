"""Reduce intake text to mapped All About Me source sections before LLM calls."""

from __future__ import annotations

import re
from typing import Final

from pdf_filler import participant_first_name_from_text

# Labels whose answers the model is allowed to see (SYSTEM_PROMPT sources only).
_SECTION_LABEL_PATTERNS: Final[tuple[str, ...]] = (
    r"participant'?s?\s+strengths?\s+and\s+favorite\s+interests?",
    r"strengths?\s+and\s+favorite\s+interests?",
    r"favorite\s+interests?",
    r"favorite\s+reinforcers?",
    r"parent/?\s*guardian(?:\s*\d+)?",
    r"primary\s+contact",
    r"(?:parent|guardian)\s*(?:name|phone|email|cell(?:ular)?(?:\s*phone)?)",
    r"(?:home|cell|mobile|work)\s*phone(?:\s*number)?",
    r"(?:phone|email|e-mail)(?:\s*(?:number|address))?\s*:",
    r"medications?(?:\s*,?\s*if\s+taken\s+during\s+camp\s+hours)?",
    r"does\s+(?:the\s+)?participant\s+have\s+seizures?",
    r"participant\s+allergies(?:\s*\([^)]*\))?",
    r"(?:participant\s+)?food\s+allergies?/dietary\s+restrictions?",
    r"(?:is\s+)?epi[\s-]?pen(?:\s+provided)?",
    r"emergency\s+medication",
    r"does\s+(?:the\s+)?participant\s+need\s+help\s+in\s+the\s+restroom",
    r"help\s+in\s+the\s+restroom",
    r"bathroom\s+needs?",
    r"participant'?s?\s+behavioral\s+challenges?",
    r"participant'?s?\s+areas?\s+that\s+can\s+be\s+challenging",
    r"areas?\s+that\s+can\s+be\s+challenging",
    r"strategies?\s+that\s+help(?:\s+with\s+challenges?)?",
    r"behaviou?ral\s+strategies?",
    r"behavorial\s+strategies?",
)

_COMBINED_LABEL_RE: Final = re.compile(
    # Leading word boundary only — some labels end with ":" (not a word char).
    rf"(?im)\b(?:{'|'.join(_SECTION_LABEL_PATTERNS)})"
)

_DOB_RE: Final = re.compile(r"\bDOB\s*:\s*[^\n|]+", flags=re.IGNORECASE)
_FOR_FULL_RE: Final = re.compile(
    r"\bFor:\s*[^|\n]+",
    flags=re.IGNORECASE,
)
_SUBMISSION_ID_RE: Final = re.compile(
    r"\bsubmission\s*ID\s*#?\s*\d+\b",
    flags=re.IGNORECASE,
)
# Loose "Name:" lines from informal fixtures — keep first name only.
_NAME_LINE_RE: Final = re.compile(
    r"(?im)^\s*name\s*:\s*(.+)$",
)


def _redact_identifiers(chunk: str, *, first_name: str) -> str:
    """Strip DOB, submission IDs, and full For: headers from a kept chunk."""
    cleaned = _DOB_RE.sub("DOB: [redacted]", chunk)
    cleaned = _SUBMISSION_ID_RE.sub("submission ID# [redacted]", cleaned)
    if first_name:
        cleaned = _FOR_FULL_RE.sub(f"For: [redacted], {first_name}", cleaned)
    else:
        cleaned = _FOR_FULL_RE.sub("For: [redacted]", cleaned)
    return cleaned.strip()


_NEW_FIELD_LINE_RE: Final = re.compile(
    r"^[A-Za-z][^:\n]{0,80}:\s*\S",
)


def _clip_section(text: str, start: int, hard_end: int) -> str:
    """Keep a label line plus continuation lines; stop at blank or new Label: value."""
    chunk = text[start:hard_end]
    lines = chunk.splitlines()
    if not lines:
        return ""
    kept: list[str] = [lines[0]]
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            break
        if _NEW_FIELD_LINE_RE.match(stripped):
            break
        kept.append(line)
        if len(kept) >= 8:
            break
    return "\n".join(kept).strip()


def _section_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans from each mapped label through the next label."""
    matches = list(_COMBINED_LABEL_RE.finditer(text))
    if not matches:
        return []
    spans: list[tuple[int, int]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        end = min(end, start + 800)
        spans.append((start, end))
    return spans


def minimize_source_text_for_llm(text: str) -> str:
    """Keep only mapped source sections; redact participant last name / DOB / IDs.

    Full intake text is still used locally for header-name override and none-guards.
    This string is what gets sent to Ollama.
    """
    source = (text or "").strip()
    if not source:
        return ""

    first_name = participant_first_name_from_text(source)
    parts: list[str] = []

    if first_name:
        parts.append(f"Participant first name (from For: header): {first_name}")
    else:
        name_match = _NAME_LINE_RE.search(source)
        if name_match:
            # Prefer first token only so last names are not sent when avoidable.
            raw_name = name_match.group(1).strip()
            given = raw_name.split(",", 1)[-1].strip().split()[0] if raw_name else ""
            if given:
                parts.append(f"Participant first name: {given}")

    for start, end in _section_spans(source):
        chunk = _redact_identifiers(
            _clip_section(source, start, end),
            first_name=first_name,
        )
        if chunk and chunk not in parts:
            parts.append(chunk)

    if not parts:
        # No mapped labels — send nothing rather than the full PDF text.
        return (
            "(no mapped All About Me source sections found in the upload; "
            "leave all JSON fields empty)"
        )

    return "\n\n".join(parts)
