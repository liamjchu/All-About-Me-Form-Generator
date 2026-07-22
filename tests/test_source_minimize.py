"""Tests for LLM source minimization / PII redaction."""

from __future__ import annotations

from source_minimize import minimize_source_text_for_llm


def test_minimize_keeps_mapped_sections_and_redacts_header_pii() -> None:
    source = (
        "submission ID# 998877 For: Rivera, Alex | DOB: 5/6/2014 | Camp intake\n"
        "Diagnosis: autism spectrum disorder\n"
        "Participant's strengths and favorite interests: swimming, dogs\n"
        "Favorite reinforcers: stickers\n"
        "Parent/Guardian 1: Sam Rivera\n"
        "Phone: 555-0100\n"
        "Email: sam@example.com\n"
        "Does the participant need help in the restroom? No\n"
        "Participant's behavioral challenges: None at the moment\n"
        "Unrelated emergency contact SSN: 123-45-6789\n"
    )
    minimized = minimize_source_text_for_llm(source)
    assert "Alex" in minimized
    assert "swimming" in minimized
    assert "stickers" in minimized
    assert "Sam Rivera" in minimized
    assert "555-0100" in minimized
    assert "None at the moment" in minimized
    assert "Rivera, Alex" not in minimized
    assert "5/6/2014" not in minimized
    assert "998877" not in minimized
    assert "autism" not in minimized.lower()
    assert "123-45-6789" not in minimized


def test_minimize_returns_placeholder_when_no_mapped_sections() -> None:
    minimized = minimize_source_text_for_llm(
        "Random notes about the weather and a diagnosis of ADHD."
    )
    assert "no mapped" in minimized.lower()
    assert "ADHD" not in minimized
