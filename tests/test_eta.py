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
