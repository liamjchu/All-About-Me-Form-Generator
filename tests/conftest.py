"""Shared fixtures for genuine (mostly mock-free) tests."""

from __future__ import annotations

import pytest

from tests.helpers import start_ollama_server


@pytest.fixture
def ollama_server():
    """Start a real local HTTP server that speaks enough of Ollama's chat API."""
    server, thread, base_url = start_ollama_server()
    yield base_url
    server.shutdown()
    thread.join(timeout=2)


@pytest.fixture
def sample_form_data() -> dict[str, str]:
    return {
        "name": "Jordan Lee",
        "favorite_things_1": "painting",
        "favorite_things_2": "soccer",
        "favorite_things_3": "cats",
        "reinforcers_1": "praise",
        "reinforcers_2": "snack",
        "reinforcers_3": "iPad time",
        "reinforcers_4": "walks",
        "parent_name": "Taylor Lee",
        "parent_phone": "555-0199",
        "parent_email": "taylor.lee@example.com",
        "allergies": "Latex",
        "bathroom_needs": "Needs assistance with fasteners",
        "behavioral_management": "Use a calm voice and offer two choices.",
    }
