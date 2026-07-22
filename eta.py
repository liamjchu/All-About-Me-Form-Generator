"""Rough remaining-time estimates for the Generate Profiles run."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

# Planned seconds per work unit (local Ollama on a typical laptop).
PREP_SECONDS: float = 0.4
GEN_SECONDS: float = 8.0

T = TypeVar("T")


def run_with_heartbeat(
    work: Callable[[], T],
    *,
    on_tick: Callable[[float, float], None],
    expected_seconds: float,
    poll_seconds: float = 0.05,
) -> T:
    """Run blocking work off-thread and report progress while it runs.

    ``on_tick(elapsed_seconds, fraction_within_slice)`` is called periodically
    so a UI can keep updating during a long local model call. ``fraction``
    eases toward 0.92 based on ``expected_seconds``, then jumps to done when
    the worker finishes.
    """
    box: dict[str, object] = {}

    def target() -> None:
        try:
            box["result"] = work()
        except BaseException as error:  # noqa: BLE001 - re-raised below
            box["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    started = time.monotonic()
    while thread.is_alive():
        elapsed = time.monotonic() - started
        if expected_seconds > 0:
            within = min(0.92, elapsed / expected_seconds)
        else:
            within = 0.5
        on_tick(elapsed, within)
        thread.join(timeout=poll_seconds)

    error = box.get("error")
    if error is not None:
        assert isinstance(error, BaseException)
        raise error
    return box["result"]  # type: ignore[return-value]


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


def estimate_batch_seconds(*, file_count: int) -> float:
    """Initial estimate before any work finishes."""
    count = max(0, file_count)
    return count * (PREP_SECONDS + GEN_SECONDS)


@dataclass
class EtaTracker:
    """Tracks planned work units and derives remaining time from elapsed rate."""

    total_units: float
    completed_units: float = 0.0
    started_at: float = field(default_factory=time.monotonic)
    # Fractional credit for the unit currently running (heartbeat), not yet
    # committed via add_completed. Without this, long blocked work makes
    # elapsed/completed climb and the ETA increases every second.
    provisional_units: float = 0.0

    def add_completed(self, units: float) -> None:
        if units > 0:
            self.completed_units += units
        self.provisional_units = 0.0

    def set_total(self, total_units: float) -> None:
        """Replace the planned total (e.g. after grouping is known)."""
        self.total_units = max(total_units, self.completed_units)

    def set_provisional(self, units: float) -> None:
        """Credit in-flight work so remaining time can shrink mid-step."""
        self.provisional_units = max(0.0, units)

    def remaining_seconds(self) -> float:
        elapsed = time.monotonic() - self.started_at
        done = min(
            self.total_units,
            self.completed_units + self.provisional_units,
        )
        left = max(0.0, self.total_units - done)
        if done <= 0.05:
            return max(0.0, self.total_units - elapsed)
        rate = elapsed / done
        return left * rate

    def label(self, status: str) -> str:
        return f"{format_remaining(self.remaining_seconds())} · {status}"
