# All About Me Form Generator

Convert participant information PDFs into filled All About Me PDF profiles using a
local [Ollama](https://ollama.com) model (no cloud API key).

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
   `OLLAMA_BASE_URL` must be a loopback address (`127.0.0.1` / `localhost` /
   `::1`); remote hosts are rejected so intake text stays on this machine.

4. Start the app (localhost only by default — see `.streamlit/config.toml`):
   ```bash
   streamlit run app.py
   ```
   Open **http://localhost:8501** on this computer.

### Optional: HTTPS via reverse proxy

Keep Streamlit on `127.0.0.1`, then put TLS in front with Caddy:

```bash
caddy run --config deploy/Caddyfile
```

Edit `deploy/Caddyfile` to use your hostname. Traffic to Streamlit stays on
loopback; browsers talk HTTPS to Caddy.

### Optional: LAN access

Only if you intentionally need other devices on the same network:

```bash
streamlit run app.py --server.address 0.0.0.0
```

Prefer a VPN or SSH tunnel over opening the port on untrusted Wi‑Fi. Other
devices can use `http://YOUR_LAN_IP:8501` (find it with
`ipconfig getifaddr en0` on macOS). Do not open `http://0.0.0.0:8501` in the
browser — that address only means “listen on all interfaces.”

## Upload limits

Only PDF uploads are accepted, max **10 MB per file**. Text is extracted from
each PDF; only mapped All About Me source sections (with participant last name /
DOB / submission IDs redacted) are sent to the local model. Scanned image-only
PDFs (no selectable text) are not supported. Use **Wipe profiles from this
session** (or download, which also clears) to drop generated profiles from
memory.
