"""launch() must never silently double-bind a busy port (Windows
SO_REUSEADDR lets two servers share one port and the OLD one keeps the
traffic — a stale viewer then hijacks the browser)."""
import socket
import threading

from pyorps.gui.app import _resolve_port


def test_free_port_is_kept():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    assert _resolve_port("127.0.0.1", free) == free


def test_busy_port_is_skipped():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    busy = server.getsockname()[1]
    accepter = threading.Thread(target=lambda: server.accept(), daemon=True)
    accepter.start()
    try:
        chosen = _resolve_port("127.0.0.1", busy)
        assert chosen != busy
        assert chosen > busy
    finally:
        server.close()
