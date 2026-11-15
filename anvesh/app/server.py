"""HTTP API + dashboard host (standard library only; no web-framework dependency).

  /api/v1/...   JSON API, versioned. Errors are {"error": {"code", "message", "details"}}.
  /             the dashboard (static files from `anvesh/app/static` ONLY -- never the repo root).

AUTH. `Authorization: Bearer <token>` or the HttpOnly, SameSite=Strict `anvesh_session` cookie (so <img>/<video>
tags for live view and evidence work). State-changing requests authenticated by cookie must carry
`X-Requested-With: anvesh` (cross-site forms cannot set it) and, if an Origin header is present, it must match
the Host. Every route declares the permission it needs; the role matrix lives in `anvesh.app.auth`.

THREADING. One thread per request (ThreadingHTTPServer). Request handlers never open a camera or load a model:
those happen on session threads (`anvesh.app.runner`), so one dead camera cannot block the API.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import re
import shutil
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from anvesh.app import cameras as cams
from anvesh.app.accounts import AccountError
from anvesh.app.auth import AuthError
from anvesh.app.cases import CaseError
from anvesh.app.capabilities import CAPABILITIES, all_category_status, category_status
from anvesh.app.reports import build_case_package, case_report_html
from anvesh.app.retention import RetentionError
from anvesh.app.runner import TERMINAL_STATES, SessionManager
from anvesh.app.timeutil import now_iso

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"
API = "/api/v1"
VIDEO_EXT = cams.VIDEO_EXT
COOKIE = "anvesh_session"
PASSWORD_CHANGE_ALLOWED = ("/auth/me", "/auth/password", "/auth/logout", "/health", "/ready")
MAX_JSON_BYTES = 1 << 20
MAX_FIELD_CHARS = 20000  # longest free-text field (notes, reasons) accepted from a client
DEFAULT_REQUEST_TIMEOUT_S = 60.0


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details=None) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details


def content_length(h) -> int:
    """The request's Content-Length as a non-negative int. A negative or non-numeric value is refused (and the connection
    closed, because the body framing can no longer be trusted); `rfile.read(-1)` would otherwise block the handler thread
    until the client hangs up."""
    raw = h.headers.get("Content-Length")
    if raw is None or raw.strip() == "":
        return 0
    try:
        n = int(raw.strip())
    except ValueError:
        n = -1
    if n < 0:
        h.close_connection = True
        raise ApiError(400, "bad_length", "Content-Length must be a non-negative integer")
    return n


class Request:
    def __init__(self, handler, method, path, query, params) -> None:
        self.h, self.method, self.path, self.query, self.params = handler, method, path, query, params
        self.user = None
        self.token = None
        self._json = None

    def q(self, name, default=None):
        v = self.query.get(name)
        return v[0] if v else default

    def qint(self, name, default, lo=0, hi=1000):
        try:
            v = int(self.q(name, default))
        except ValueError:
            raise ApiError(422, "bad_query", f"{name} must be an integer") from None
        return max(lo, min(hi, v))

    @property
    def body(self) -> dict:
        if self._json is None:
            n = content_length(self.h)
            if n > MAX_JSON_BYTES:
                raise ApiError(413, "too_large", "JSON body too large")  # the connection is closed by dispatch()
            raw = self.h.rfile.read(n) if n else b""
            self.h._body_read = True
            if not raw:
                self._json = {}
            else:
                try:
                    self._json = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError, RecursionError):
                    raise ApiError(400, "bad_json", "request body is not valid JSON") from None
            if not isinstance(self._json, dict):
                raise ApiError(400, "bad_json", "request body must be a JSON object")
        return self._json

    def need(self, name: str, typ=str, maxlen: int = MAX_FIELD_CHARS):
        v = self.body.get(name)
        if v is None or (typ is str and not isinstance(v, str)) or (typ is str and not v.strip()) or not isinstance(v, typ)                 or (typ is not bool and isinstance(v, bool)):
            raise ApiError(422, "validation", f"'{name}' is required ({typ.__name__})")
        if isinstance(v, str) and len(v) > maxlen:
            raise ApiError(422, "validation", f"'{name}' is too long (max {maxlen} characters)")
        return v

    def opt(self, name: str, typ=str, default=None, maxlen: int = MAX_FIELD_CHARS):
        """An optional body field: absent or null -> `default`; present with the wrong type -> 422 (never coerced by truthiness)."""
        v = self.body.get(name)
        if v is None:
            return default
        if not isinstance(v, typ) or (typ is not bool and isinstance(v, bool)):
            raise ApiError(422, "validation", f"'{name}' must be of type {typ.__name__}")
        if isinstance(v, str) and len(v) > maxlen:
            raise ApiError(422, "validation", f"'{name}' is too long (max {maxlen} characters)")
        return v

    def require(self, permission: str) -> None:
        if self.user is None:
            raise ApiError(401, "unauthenticated", "sign in required")
        if not self.user.can(permission):
            raise ApiError(403, "forbidden", f"your role ({self.user.role}) lacks the '{permission}' permission")


class _QuietThreadingServer(ThreadingHTTPServer):
    """A browser tab that closes mid-request (or mid-stream) is normal, not an error: a vanished peer is logged at debug level only.
    Anything else is logged with its traceback through `logging` instead of being printed to stderr."""

    def handle_error(self, request, client_address):
        import sys

        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, TimeoutError)):
            log.debug("client %s went away: %s", client_address[0], type(exc).__name__)
        else:
            log.exception("unhandled error serving %s", client_address[0])


class AppServer:
    def __init__(self, services, sessions: SessionManager = None, host: str = None, port: int = None,
                 request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> None:
        self.s = services
        self.request_timeout_s = request_timeout_s
        self.sessions = sessions or SessionManager(services)
        self.host = host or services.config.host
        self.port = services.config.port if port is None else port
        self.routes: list = []
        self.demo_login = None  # (username, password) enables POST /auth/demo on a loopback server; set by scripts/run_demo.py --one-click
        self.simple_home = False  # serve the simple officer screen at "/" (the full dashboard stays at /advanced)
        self._register_routes()
        self._httpd = None
        self._thread = None

    # -- routing -----------------------------------------------------------------

    def route(self, method: str, pattern: str, handler, public: bool = False) -> None:
        regex = re.compile("^" + re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", pattern) + "$")
        self.routes.append((method, regex, handler, public))

    def _register_routes(self) -> None:
        from anvesh.app import routes
        routes.register(self)

    def start(self) -> int:
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "ANVESH"
            timeout = server.request_timeout_s  # drops slow-loris / half-open connections instead of holding a thread forever

            def log_message(self, fmt, *args):  # request logs go through logging, never with query strings or tokens
                log.debug("%s %s", self.command, self.path.split("?")[0])

            def do_GET(self):
                server.dispatch(self, "GET")

            def do_POST(self):
                server.dispatch(self, "POST")

            def do_PATCH(self):
                server.dispatch(self, "PATCH")

            def do_PUT(self):
                server.dispatch(self, "PUT")

        self._httpd = _QuietThreadingServer((self.host, self.port), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="anvesh-http", daemon=True)
        self._thread.start()
        self.s.alerts.start_worker()
        return self.port

    def stop(self) -> None:
        self.sessions.shutdown()
        self.s.alerts.stop_worker()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()

    # -- dispatch ------------------------------------------------------------------

    def dispatch(self, h, method: str) -> None:
        h._body_read = False
        if h.headers.get("Transfer-Encoding"):
            # Bodies are framed by Content-Length only. Accepting a chunked body we do not decode would leave its bytes on a
            # keep-alive connection to be parsed as the next request (request smuggling), so refuse and close.
            h.close_connection = True
            h._body_read = True
            return self._error(h, 411, "length_required", "Transfer-Encoding is not supported; send Content-Length", headers={"Connection": "close"})
        try:
            self._dispatch(h, method)
        finally:
            self._drain_body(h)

    @staticmethod
    def _drain_body(h) -> None:
        """A handler (or an early error) that did not read the request body would leave those bytes on the
        keep-alive connection, where they would corrupt the NEXT request line. Discard them, or close the
        connection if the body is too large to be worth reading."""
        if getattr(h, "_body_read", True):
            return
        try:
            n = content_length(h)
        except ApiError:
            return
        if n <= 0:
            return
        if n > MAX_JSON_BYTES:
            h.close_connection = True
            return
        try:
            h.rfile.read(n)
        except OSError:
            h.close_connection = True

    def _dispatch(self, h, method: str) -> None:
        parsed = urlparse(h.path)
        path = unquote(parsed.path)
        try:
            if not path.startswith(API):
                return self._static(h, method, path)
            for m, regex, handler, public in self.routes:
                if m != method:
                    continue
                match = regex.match(path[len(API):])
                if match is None:
                    continue
                req = Request(h, method, path, parse_qs(parsed.query), match.groupdict())
                self._authenticate(req)
                if not public and req.user is None:
                    raise ApiError(401, "unauthenticated", "sign in required")
                if req.user is not None and req.user.must_change_password and path[len(API):] not in PASSWORD_CHANGE_ALLOWED:
                    # A temporary password (set by an administrator) may only be used to choose a real one.
                    raise ApiError(403, "password_change_required", "you must change your temporary password before doing anything else")
                if method != "GET" and not public:
                    self._csrf(req)
                result = handler(req)
                if result is not None:
                    self._send_json(h, *result) if isinstance(result, tuple) else self._send_json(h, 200, result)
                return
            if any(regex.match(path[len(API):]) for _m, regex, _h, _p in self.routes):
                raise ApiError(405, "method_not_allowed", f"{method} not allowed here")
            raise ApiError(404, "not_found", "no such endpoint")
        except ApiError as e:
            self._error(h, e.status, e.code, e.message, e.details)
        except CaseError as e:
            self._error(h, e.status, "case_rule", str(e), e.errors or None)
        except RetentionError as e:
            self._error(h, e.status, "retention", str(e))
        except AuthError as e:
            self._error(h, e.status, "auth", str(e))
        except AccountError as e:
            self._error(h, e.status, "account_rule", str(e))
        except cams.ConfigError as e:
            self._error(h, 422, "validation", str(e))
        except LookupError as e:
            self._error(h, 404, "not_found", str(e.args[0]) if e.args else "not found")
        except PermissionError as e:
            self._error(h, 403, "forbidden", str(e))
        except ValueError as e:
            self._error(h, 422, "validation", str(e))
        except (BrokenPipeError, ConnectionResetError, TimeoutError):  # the peer vanished or stalled mid-body: nothing to answer
            h.close_connection = True
        except Exception:  # noqa: BLE001
            log.exception("unhandled error")
            self._error(h, 500, "internal", "internal error")

    def _authenticate(self, req: Request) -> None:
        auth = req.h.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else None
        via_cookie = False
        if not token:
            for part in (req.h.headers.get("Cookie") or "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == COOKIE:
                    token, via_cookie = v, True
        req.token = token
        req.via_cookie = via_cookie
        req.user = self.s.auth.authenticate(token) if token else None

    def _csrf(self, req: Request) -> None:
        if not getattr(req, "via_cookie", False):
            return
        if req.h.headers.get("X-Requested-With") != "anvesh":
            raise ApiError(403, "csrf", "missing X-Requested-With header")
        origin = req.h.headers.get("Origin")
        if origin and urlparse(origin).netloc != req.h.headers.get("Host"):
            raise ApiError(403, "csrf", "cross-origin request refused")

    # -- responses -----------------------------------------------------------------

    @staticmethod
    def _headers(h, extra: dict = None) -> None:
        h.send_header("X-Content-Type-Options", "nosniff")
        h.send_header("X-Frame-Options", "DENY")
        h.send_header("Referrer-Policy", "no-referrer")
        h.send_header("Cache-Control", "no-store")
        h.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
                                                 "style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'")
        for k, v in (extra or {}).items():
            h.send_header(k, v)

    def _send_json(self, h, status: int, obj, headers: dict = None) -> None:
        data = json.dumps(obj, default=str).encode("utf-8")
        h.send_response(status)
        h.send_header("Content-Type", "application/json; charset=utf-8")
        h.send_header("Content-Length", str(len(data)))
        self._headers(h, headers)
        h.end_headers()
        h.wfile.write(data)

    def _error(self, h, status, code, message, details=None, headers: dict = None) -> None:
        try:
            self._send_json(h, status, {"error": {"code": code, "message": message, "details": details}}, headers)
        except OSError:
            pass

    def send_bytes(self, h, data: bytes, mime: str, status: int = 200, headers: dict = None) -> None:
        h.send_response(status)
        h.send_header("Content-Type", mime)
        h.send_header("Content-Length", str(len(data)))
        self._headers(h, headers)
        h.end_headers()
        h.wfile.write(data)

    def send_file(self, h, path: Path, mime: str = None, download_name: str = None) -> None:
        """Static/evidence file with HTTP Range support (video seeking)."""
        size = path.stat().st_size
        mime = mime or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        start, end, status = 0, size - 1, 200
        rng = h.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
            if not m or (m.group(1) == "" and m.group(2) == ""):
                h.send_response(416)
                h.send_header("Content-Range", f"bytes */{size}")
                h.send_header("Content-Length", "0")
                h.end_headers()
                return
            if m.group(1) == "":
                start = max(0, size - int(m.group(2)))
            else:
                start = int(m.group(1))
                if m.group(2):
                    end = min(end, int(m.group(2)))
            if start > end or start >= size:
                h.send_response(416)
                h.send_header("Content-Range", f"bytes */{size}")
                h.send_header("Content-Length", "0")
                h.end_headers()
                return
            status = 206
        length = end - start + 1
        h.send_response(status)
        h.send_header("Content-Type", mime)
        h.send_header("Accept-Ranges", "bytes")
        h.send_header("Content-Length", str(length))
        if status == 206:
            h.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        extra = {"Content-Disposition": f'attachment; filename="{download_name}"'} if download_name else {}
        self._headers(h, extra)
        h.end_headers()
        with path.open("rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(1 << 16, remaining))
                if not chunk:
                    break
                h.wfile.write(chunk)
                remaining -= len(chunk)

    def _static(self, h, method: str, path: str) -> None:
        if method != "GET":
            return self._error(h, 405, "method_not_allowed", "static files are read-only")
        if path in ("", "/"):
            rel = "officer.html" if self.simple_home else "index.html"
        elif path == "/advanced":
            rel = "index.html"
        elif path == "/officer":
            rel = "officer.html"
        else:
            rel = path.lstrip("/")
        target = (STATIC_DIR / rel).resolve()
        try:
            target.relative_to(STATIC_DIR.resolve())
        except ValueError:
            return self._error(h, 404, "not_found", "not found")
        if not target.is_file():
            return self._error(h, 404, "not_found", "not found")
        self.send_file(h, target)

    # -- helpers used by routes ------------------------------------------------------

    def save_upload(self, req: Request) -> dict:
        import cv2

        name = Path(req.q("filename", "upload.mp4")).name
        ext = Path(name).suffix.lower()
        if ext not in VIDEO_EXT:
            raise ApiError(422, "validation", f"unsupported file type {ext!r}; allowed: {sorted(VIDEO_EXT)}")
        n = content_length(req.h)
        if n <= 0:
            raise ApiError(411, "length_required", "Content-Length is required")
        if n > self.s.config.max_upload_bytes:
            raise ApiError(413, "too_large", f"upload exceeds {self.s.config.max_upload_bytes} bytes")
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        dest = self.s.config.uploads_dir / f"{uuid.uuid4().hex[:8]}-{safe}"
        remaining = n
        req.h._body_read = True
        with dest.open("wb") as fh:
            while remaining > 0:
                chunk = req.h.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                fh.write(chunk)
                remaining -= len(chunk)
        if remaining:
            dest.unlink(missing_ok=True)
            raise ApiError(400, "truncated", "upload ended early")
        cap = cv2.VideoCapture(str(dest))
        ok = cap.isOpened()
        fps, frames = (cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else (0, 0)
        cap.release()
        if not ok or fps <= 0:
            dest.unlink(missing_ok=True)
            raise ApiError(422, "unreadable_video", "the file could not be opened as a video")
        self.s.audit.append(req.user.username, "upload.video", "upload", dest.name, {"bytes": n, "name": name}, actor_role=req.user.role)
        return {"path": str(dest), "name": dest.name, "bytes": n, "fps": fps, "frames": int(frames),
                "duration_s": round(frames / fps, 2) if fps else None}


def serve(services, host=None, port=None, tracker_factory=None) -> AppServer:
    server = AppServer(services, SessionManager(services, tracker_factory=tracker_factory), host=host, port=port)
    server.start()
    return server
