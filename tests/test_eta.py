"""Tests for remaining-time estimates."""

from __future__ import annotations

import time

from eta import (
    GEN_TEXT_SECONDS,
    GEN_VISION_SECONDS,
    PREP_IMAGE_SECONDS,
    EtaTracker,
    estimate_batch_seconds,
    format_remaining,
    is_image_upload,
    run_with_heartbeat,
)


def test_format_remaining_seconds_and_minutes() -> None:
    assert format_remaining(None) == "Estimating time remaining…"
    assert format_remaining(1) == "About 1 second remaining"
    assert format_remaining(12) == "About 12 seconds remaining"
    assert format_remaining(60) == "About 1 min remaining"
    assert format_remaining(75) == "About 1 min 15s remaining"
    assert format_remaining(3600) == "About 1 hr remaining"
    assert format_remaining(3660) == "About 1 hr 1 min remaining"


def test_is_image_upload_by_mime_or_extension() -> None:
    assert is_image_upload("page.jpg", "image/jpeg")
    assert is_image_upload("scan.PNG", None)
    assert not is_image_upload("notes.txt", "text/plain")


def test_estimate_batch_seconds_weights_photos_heavier() -> None:
    from eta import PREP_TEXT_SECONDS

    text_only = estimate_batch_seconds(
        file_names=["a.txt"],
        mime_types=["text/plain"],
        pages_per_form=2,
    )
    photos = estimate_batch_seconds(
        file_names=["a.jpg", "b.jpg"],
        mime_types=["image/jpeg", "image/jpeg"],
        pages_per_form=2,
    )
    assert photos > text_only
    assert photos == 2 * PREP_IMAGE_SECONDS + GEN_VISION_SECONDS
    assert text_only == PREP_TEXT_SECONDS + GEN_TEXT_SECONDS


def test_eta_tracker_scales_with_observed_rate() -> None:
    tracker = EtaTracker(total_units=100.0, started_at=time.monotonic() - 10.0)
    tracker.add_completed(20.0)
    # 10s for 20 units => 0.5s/unit; 80 left => ~40s
    remaining = tracker.remaining_seconds()
    assert 35 <= remaining <= 45
    assert "remaining" in tracker.label("Working…")


def test_eta_tracker_provisional_keeps_remaining_from_climbing() -> None:
    """Long blocked work must not inflate ETA every tick via elapsed/completed."""
    started = time.monotonic() - 5.0
    tracker = EtaTracker(total_units=50.0, started_at=started)
    tracker.add_completed(5.0)  # prep done; gen slice still running
    early = tracker.remaining_seconds()

    tracker.started_at = time.monotonic() - 25.0
    without_provisional = tracker.remaining_seconds()
    assert without_provisional > early

    tracker.set_provisional(20.0)  # ~20s into a 45s vision call
    with_provisional = tracker.remaining_seconds()
    assert with_provisional < without_provisional
    assert with_provisional < early


def test_run_with_heartbeat_ticks_while_waiting() -> None:
    ticks: list[tuple[float, float]] = []

    def work() -> str:
        time.sleep(0.12)
        return "ok"

    result = run_with_heartbeat(
        work,
        on_tick=lambda elapsed, within: ticks.append((elapsed, within)),
        expected_seconds=0.2,
        poll_seconds=0.04,
    )
    assert result == "ok"
    assert len(ticks) >= 1
    assert all(0.0 <= within <= 0.92 for _, within in ticks)


def test_run_with_heartbeat_rethrows_worker_errors() -> None:
    def work() -> None:
        raise ValueError("boom")

    try:
        run_with_heartbeat(
            work,
            on_tick=lambda *_: None,
            expected_seconds=1.0,
            poll_seconds=0.02,
        )
    except ValueError as error:
        assert "boom" in str(error)
    else:
        raise AssertionError("expected ValueError")
