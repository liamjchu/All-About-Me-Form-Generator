"""Rough remaining-time estimates for the Generate Profiles run."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

# Planned seconds per work unit (local Ollama on a typical laptop).
PREP_IMAGE_SECONDS: float = 1.5
PREP_TEXT_SECONDS: float = 0.4
GEN_VISION_SECONDS: float = 55.0
GEN_TEXT_SECONDS: float = 8.0


def is_image_upload(file_name: str, mime_type: str | None) -> bool:
    name = file_name.lower()
    mime = (mime_type or "").lower()
    return mime.startswith("image/") or name.endswith((".png", ".jpg", ".jpeg"))


def format_remaining(seconds: float | None) -> str:
    """Human-readable remaining-time label."""
    if seconds is None:
        return "Estimating time remaining…"
    whole = max(0, int(round(seconds)))
    if whole <= 1:
        return "About 1 second remaining"
    if whole < 60:
        return f"About {whole} seconds remaining"
    minutes, secs = divmod(whole, 60)
    if minutes < 60:
        if secs == 0:
            return f"About {minutes} min remaining"
        return f"About {minutes} min {secs}s remaining"
    hours, minutes = divmod(minutes, 60)
    if minutes == 0:
        return f"About {hours} hr remaining"
    return f"About {hours} hr {minutes} min remaining"


def estimate_prep_seconds(*, image_count: int, text_count: int) -> float:
    return image_count * PREP_IMAGE_SECONDS + text_count * PREP_TEXT_SECONDS


def estimate_gen_seconds(*, vision_groups: int, text_groups: int) -> float:
    return vision_groups * GEN_VISION_SECONDS + text_groups * GEN_TEXT_SECONDS


def estimate_batch_seconds(
    *,
    file_names: Sequence[str],
    mime_types: Sequence[str | None],
    pages_per_form: int,
) -> float:
    """Initial estimate before any work finishes."""
    images = sum(
        1 for name, mime in zip(file_names, mime_types) if is_image_upload(name, mime)
    )
    texts = len(file_names) - images
    prep = estimate_prep_seconds(image_count=images, text_count=texts)
    pages = max(1, pages_per_form)
    vision_groups = (images + pages - 1) // pages if images else 0
    return prep + estimate_gen_seconds(vision_groups=vision_groups, text_groups=texts)


@dataclass
class EtaTracker:
    """Tracks planned work units and derives remaining time from elapsed rate."""

    total_units: float
    completed_units: float = 0.0
    started_at: float = field(default_factory=time.monotonic)

    def add_completed(self, units: float) -> None:
        if units > 0:
            self.completed_units += units

    def set_total(self, total_units: float) -> None:
        """Replace the planned total (e.g. after grouping is known)."""
        self.total_units = max(total_units, self.completed_units)

    def remaining_seconds(self) -> float:
        elapsed = time.monotonic() - self.started_at
        left = max(0.0, self.total_units - self.completed_units)
        if self.completed_units <= 0.05:
            return max(0.0, self.total_units - elapsed)
        rate = elapsed / self.completed_units
        return left * rate

    def label(self, status: str) -> str:
        return f"{format_remaining(self.remaining_seconds())} · {status}"
