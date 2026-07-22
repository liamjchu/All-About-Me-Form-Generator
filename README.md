# All About Me Form Generator

Convert participant information PDFs into filled All About Me PDF profiles using a
local [Ollama](https://ollama.com) model (no OpenAI cloud key).

## Setup

1. Install and start [Ollama](https://ollama.com/download), then pull the text
   model this app expects:
   ```bash
   ollama pull llama3.2:latest
   ```
   Confirm the server is up:
   ```bash
   curl http://127.0.0.1:11434/api/tags
   ```

2. Create a virtual environment, then install dependencies:
   ```bash
   /opt/homebrew/bin/python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

3. Optional: create a `.env` in the project root to override defaults:
   ```text
   OLLAMA_BASE_URL=http://127.0.0.1:11434
   OLLAMA_MODEL=llama3.2:latest
   ```
   `.env` is ignored by Git. No API key is required.

4. Start the app (reachable on your LAN when bound to `0.0.0.0`):
   ```bash
   streamlit run app.py --server.address 0.0.0.0
   ```
   Open **http://localhost:8501** on this computer. Do not use `http://0.0.0.0:8501`
   in the browser — that address only means “listen on all interfaces.”
   Other devices on the same Wi‑Fi can use `http://YOUR_LAN_IP:8501`
   (find it with `ipconfig getifaddr en0`).

Only PDF uploads are accepted. Text is extracted from each PDF, then
`llama3.2:latest` maps it into fields for `formTemplate.pdf`. Scanned
image-only PDFs (no selectable text) are not supported.
