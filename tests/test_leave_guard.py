"""Tests for generation leave/stop confirmation helpers."""

from __future__ import annotations

from leave_guard import _CONFIRM_MESSAGE, set_generation_leave_guard


def test_leave_guard_injects_active_script(monkeypatch) -> None:
    captured: list[tuple[str, int]] = []

    def fake_html(script: str, height: int = 0) -> None:
        captured.append((script, height))

    monkeypatch.setattr("leave_guard.components.html", fake_html)
    monkeypatch.setattr("leave_guard.time.sleep", lambda _seconds: None)
    set_generation_leave_guard(True, flush=True)

    assert len(captured) == 1
    script, _height = captured[0]
    assert "aam-leave-guard-boot" in script
    assert "__aamGenerating" in script
    assert "stopThenReplay" in script
    assert "findStopButton" in script
    assert "HANDLER_VERSION" in script
    assert _CONFIRM_MESSAGE in script
    assert "stFileUploader" in script
    assert "true" in script


def test_leave_guard_injects_inactive_script(monkeypatch) -> None:
    captured: list[str] = []
    monkeypatch.setattr(
        "leave_guard.components.html",
        lambda script, height=0: captured.append(script),
    )
    set_generation_leave_guard(False)

    assert captured
    assert "aam-leave-guard-boot" in captured[0]
    # Inactive payload sets ACTIVE/__aamGenerating to false.
    assert "false" in captured[0]


def test_leave_guard_flush_sleeps_only_when_enabling(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("leave_guard.components.html", lambda script, height=0: None)
    monkeypatch.setattr("leave_guard.time.sleep", lambda seconds: sleeps.append(seconds))

    set_generation_leave_guard(False, flush=True)
    assert sleeps == []

    set_generation_leave_guard(True, flush=True)
    assert len(sleeps) == 1
    assert sleeps[0] > 0
