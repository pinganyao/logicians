"""Native desktop window for the live performance UI."""

from __future__ import annotations

import socket
import threading
import time


def wait_for_server(host: str, port: int, *, timeout: float = 15.0) -> None:
    """Block until the local HTTP server accepts connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.3):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"Server did not start on {host}:{port} within {timeout:.0f}s")


def run_desktop_window(host: str, port: int) -> None:
    """Start uvicorn in a background thread and open a native app window."""
    import uvicorn
    import webview

    config = uvicorn.Config(
        "logicians.server:app",
        host=host,
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    wait_for_server(host, port)
    url = f"http://{host}:{port}"

    webview.create_window(
        "The Logicians",
        url,
        width=460,
        height=820,
        min_size=(400, 640),
        resizable=True,
        text_select=True,
    )
    webview.start(debug=False)
