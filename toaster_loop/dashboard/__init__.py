"""Phase 5 dashboard: loopback-only control API + static web UI.

`app.py` holds the transport-agnostic request handler (pure JSON in/out, unit
testable without a socket); `server.py` wraps it in a stdlib HTTP server bound
to 127.0.0.1 only. No third-party web framework, no external CDN assets —
the UI is fully offline-capable.
"""

from .app import DashboardApp, ApiResponse  # noqa: F401
