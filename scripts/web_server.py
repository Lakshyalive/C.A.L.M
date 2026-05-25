"""
scripts/web_server.py
FastAPI + WebSocket bridge between therapy_voice.py and the browser UI.
"""

import asyncio
import json
import threading
import time
import webbrowser

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()

# ── Shared state ──────────────────────────────────────────

# All currently-connected browser clients
connected_clients: list = []

# The server's event loop (set once server starts)
_server_loop: asyncio.AbstractEventLoop = None

# Messages buffered before the very first client connects.
# Replayed to the first connection so the opening message is never missed.
_message_buffer: list = []

# Messages replayed to every NEW client (refresh / reconnect) after startup.
# Set by therapy_voice.py after speaking the opening greeting.
_welcome_messages: list = []

# Whether any browser has ever connected this session
_ever_connected: bool = False

# ── Interrupt event ────────────────────────────────────────
# Set by the browser (user clicks the mic button during speaking/thinking).
# Cleared by therapy_voice.py at the top of each listen cycle.
interrupt_event = threading.Event()

# Set by the browser when the user clicks the mic button while we're listening.
# Signals voice_input.record_until_silence() to stop immediately and transcribe.
stop_listening_event = threading.Event()


# ── Public helpers ─────────────────────────────────────────

def set_welcome_messages(messages: list):
    """
    Store messages that will be replayed to every new WebSocket client.
    Call this after the opening message has been spoken.
    """
    global _welcome_messages
    _welcome_messages = list(messages)


# ── WebSocket endpoint ─────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global _ever_connected

    await websocket.accept()

    if not _ever_connected:
        # First-ever browser connection: flush all buffered pre-connection messages
        _ever_connected = True
        for msg in _message_buffer:
            try:
                await websocket.send_json(msg)
            except Exception:
                pass
    else:
        # Subsequent connection (page refresh / reconnect): replay welcome messages
        for msg in _welcome_messages:
            try:
                await websocket.send_json(msg)
            except Exception:
                pass

    connected_clients.append(websocket)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
                _handle_client_message(data)
            except Exception:
                pass
    except Exception:
        pass
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


def _handle_client_message(data: dict):
    """Process messages the browser sends us (e.g. interrupt requests)."""
    action = data.get("action")
    if action == "interrupt":
        interrupt_event.set()
    elif action == "done_listening":
        stop_listening_event.set()


# ── Broadcast ──────────────────────────────────────────────

async def _broadcast(message: dict):
    dead = []
    for client in connected_clients:
        try:
            await client.send_json(message)
        except Exception:
            dead.append(client)
    for c in dead:
        if c in connected_clients:
            connected_clients.remove(c)


def send_to_ui(message: dict):
    """
    Thread-safe broadcast to all connected browser clients.
    Buffers the message if no client has connected yet — it will be flushed
    when the first browser connects.
    """
    if _server_loop is None:
        return
    if not _ever_connected:
        _message_buffer.append(message)
        return
    asyncio.run_coroutine_threadsafe(_broadcast(message), _server_loop)


# ── Serve HTML ─────────────────────────────────────────────

@app.get("/")
async def root():
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>static/index.html not found</h1>", status_code=404)


# ── Start server ───────────────────────────────────────────

def start_ui_server(host="127.0.0.1", port=8000):
    """Start the FastAPI server in a background daemon thread."""

    def run():
        global _server_loop
        _server_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_server_loop)
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        _server_loop.run_until_complete(server.serve())

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    time.sleep(1.5)                              # wait for server to bind
    webbrowser.open(f"http://{host}:{port}")     # auto-open browser
    print(f"      UI → http://{host}:{port}")