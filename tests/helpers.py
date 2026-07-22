"""Builders and local servers for real PDF/HTTP fixtures."""

from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from pypdf import PdfWriter
from reportlab.pdfgen import canvas


def make_text_pdf(text: str, pages: int = 1) -> bytes:
    """Build a real multi-page PDF with extractable text via ReportLab."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(612, 792))
    for index in range(pages):
        pdf.setFont("Helvetica", 14)
        pdf.drawString(72, 720, f"{text} (page {index + 1})")
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def make_empty_pdf() -> bytes:
    """Build a PDF with a blank page and no extractable text."""
    buffer = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(buffer)
    return buffer.getvalue()


class FakeOllamaHandler(BaseHTTPRequestHandler):
    """Minimal Ollama /api/chat stand-in that returns configured JSON content."""

    response_content: str = "{}"
    status_code: int = 200
    last_payload: dict | None = None

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            FakeOllamaHandler.last_payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            FakeOllamaHandler.last_payload = None

        body = {
            "model": (FakeOllamaHandler.last_payload or {}).get("model", ""),
            "message": {
                "role": "assistant",
                "content": FakeOllamaHandler.response_content,
            },
            "done": True,
        }
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(FakeOllamaHandler.status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        if FakeOllamaHandler.status_code < 400:
            self.wfile.write(encoded)


def start_ollama_server() -> tuple[HTTPServer, threading.Thread, str]:
    """Start FakeOllamaHandler on an ephemeral port; caller must shut it down."""
    FakeOllamaHandler.response_content = json.dumps(
        {
            "name": "Alex",
            "favorite_things": ["swimming", "music", "dogs"],
            "favorite_reinforcers": ["stickers", "breaks", "high five"],
            "parent_name": "Sam Rivera",
            "parent_phone": "555-0100",
            "parent_email": "sam@example.com",
            "allergies": "Peanuts",
            "bathroom_needs": "Needs verbal reminders",
            "behavioral_management": "Offer choices when frustrated.",
        }
    )
    FakeOllamaHandler.status_code = 200
    FakeOllamaHandler.last_payload = None

    server = HTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}"
