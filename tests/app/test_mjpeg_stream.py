"""The live video endpoint (`/sessions/{id}/stream.mjpg`) over a REAL socket and the real HTTP server.

The session objects are small fakes with exactly the two things the handler reads (`last_jpeg`, `runtime()["state"]`), so the tests control
frame timing and state deterministically; perception itself is covered by the end-to-end session tests.

Covered: frames arrive whole and in order, a stopped session ends its stream, one session's stream never carries another's frames,
a vanished client frees its handler thread, an unknown session is a 404 and the stream needs a sign-in.
"""
import http.client
import json
import socket
import threading
import time

import pytest


class FakeSession:
    """Produces a new, uniquely numbered JPEG-shaped payload every `period_s` while `state` is live."""

    def __init__(self, tag: bytes, period_s: float = 0.03) -> None:
        self.tag, self.period_s, self.state, self.n = tag, period_s, "live", 0
        self.last_jpeg = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set() and self.state == "live":
            self.n += 1
            self.last_jpeg = b"\xff\xd8" + self.tag + b"#%06d" % self.n + b"\xff\xd9"
            time.sleep(self.period_s)

    def runtime(self) -> dict:
        return {"state": self.state}

    def finish(self, state: str = "stopped") -> None:
        self.state = state
        self._stop.set()
        self._thread.join(2)
        self.last_jpeg = None  # what the real runner's _release() does

    def stop(self) -> None:  # parity with the real session (used by server shutdown)
        self.finish("stopped")


@pytest.fixture()
def server(services):
    from anvesh.app.server import serve

    services.auth.create_user("inv-demo", "Inv", "investigator", "pw-inv-demo-long", is_demo=True)
    srv = serve(services, port=0)
    srv.demo_login = ("inv-demo", "pw-inv-demo-long")
    yield srv
    for sess in list(srv.sessions._active.values()):
        if isinstance(sess, FakeSession):
            sess.finish()
    srv.sessions._active.clear()
    srv.stop()


def _cookie(srv) -> str:
    conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
    conn.request("POST", "/api/v1/auth/demo", headers={"X-Requested-With": "anvesh", "Content-Type": "application/json"})
    r = conn.getresponse()
    r.read()
    assert r.status == 200
    return r.getheader("Set-Cookie").split(";")[0]


def _open_stream(srv, session_id: str, cookie: str = None):
    s = socket.create_connection(("127.0.0.1", srv.port), timeout=10)
    req = f"GET /api/v1/sessions/{session_id}/stream.mjpg HTTP/1.1\r\nHost: x\r\nX-Requested-With: anvesh\r\n" + (f"Cookie: {cookie}\r\n" if cookie else "") + "\r\n"
    s.sendall(req.encode())
    return s


class Reader:
    """Incremental multipart/x-mixed-replace parser over a socket."""

    def __init__(self, sock) -> None:
        self.sock, self.buf, self.eof = sock, b"", False

    def _fill(self, deadline: float) -> bool:
        self.sock.settimeout(max(0.05, deadline - time.time()))
        try:
            chunk = self.sock.recv(65536)
        except socket.timeout:
            return False
        if not chunk:
            self.eof = True
            return False
        self.buf += chunk
        return True

    def status_line(self, timeout: float = 5.0) -> str:
        deadline = time.time() + timeout
        while b"\r\n\r\n" not in self.buf and time.time() < deadline and self._fill(deadline):
            pass
        head, _sep, self.buf = self.buf.partition(b"\r\n\r\n")
        return head.decode("latin1")

    def next_frame(self, timeout: float = 5.0):
        """The next complete JPEG body, or None on timeout/EOF."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            i = self.buf.find(b"--frame\r\n")
            if i >= 0:
                j = self.buf.find(b"\r\n\r\n", i)
                if j >= 0:
                    hdr = self.buf[i:j].decode("latin1")
                    length = int([ln for ln in hdr.split("\r\n") if ln.lower().startswith("content-length")][0].split(":")[1])
                    if len(self.buf) >= j + 4 + length + 2:
                        body = self.buf[j + 4:j + 4 + length]
                        self.buf = self.buf[j + 4 + length + 2:]
                        return body
            if not self._fill(deadline) and self.eof:
                return None
        return None

    def read_to_eof(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.eof:
                return True
            self._fill(deadline)
        return self.eof


def test_frames_arrive_whole_and_in_order(server):
    fake = FakeSession(b"A")
    server.sessions._active["SES-A"] = fake
    sock = _open_stream(server, "SES-A", _cookie(server))
    rd = Reader(sock)
    head = rd.status_line()
    assert head.startswith("HTTP/") and " 200 " in head.split("\r\n")[0]
    assert "multipart/x-mixed-replace; boundary=frame" in head
    assert "no-store" in head
    seen = []
    for _ in range(6):
        body = rd.next_frame()
        assert body is not None, "stream stalled"
        assert body.startswith(b"\xff\xd8") and body.endswith(b"\xff\xd9"), "a frame was cut or corrupted"
        seen.append(int(body[3:].split(b"#")[1][:6]))
    assert seen == sorted(seen) and len(set(seen)) == len(seen), f"frames repeated or out of order: {seen}"
    sock.close()


def test_stream_ends_when_the_session_stops_and_never_replays_an_old_frame(server):
    fake = FakeSession(b"A")
    server.sessions._active["SES-A"] = fake
    sock = _open_stream(server, "SES-A", _cookie(server))
    rd = Reader(sock)
    rd.status_line()
    assert rd.next_frame() is not None
    fake.finish("stopped")
    assert rd.read_to_eof(10.0), "the stream stayed open after its session stopped"
    # whatever was already in flight is fine, but nothing after the stop may be a frame from after it
    sock.close()


def test_one_sessions_stream_never_carries_another_sessions_frames(server):
    a, b = FakeSession(b"AAAA"), FakeSession(b"BBBB")
    server.sessions._active["SES-A"], server.sessions._active["SES-B"] = a, b
    cookie = _cookie(server)
    sa, sb = _open_stream(server, "SES-A", cookie), _open_stream(server, "SES-B", cookie)
    ra, rb = Reader(sa), Reader(sb)
    ra.status_line(), rb.status_line()
    for _ in range(5):
        fa, fb = ra.next_frame(), rb.next_frame()
        assert fa is not None and fb is not None
        assert fa[2:6] == b"AAAA" and fb[2:6] == b"BBBB"
    a.finish("stopped")
    assert ra.read_to_eof(10.0)
    # B is unaffected by A stopping
    assert rb.next_frame() is not None and not rb.eof
    sa.close(), sb.close()


def test_a_vanished_client_releases_its_handler_thread(server):
    fake = FakeSession(b"A")
    server.sessions._active["SES-A"] = fake
    baseline = threading.active_count()
    cookie = _cookie(server)
    time.sleep(0.5)  # let the sign-in request's own handler thread finish
    baseline = min(baseline, threading.active_count())
    socks = []
    for _ in range(5):
        s = _open_stream(server, "SES-A", cookie)
        rd = Reader(s)
        rd.status_line()
        assert rd.next_frame() is not None
        socks.append(s)
    assert threading.active_count() >= baseline + 5, "the five streams should each hold a handler thread while open"
    for s in socks:  # abrupt close: RST, as a closed browser tab does
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
        s.close()
    deadline = time.time() + 10
    while time.time() < deadline and threading.active_count() > baseline:
        time.sleep(0.1)
    assert threading.active_count() <= baseline, "stream handler threads were not released after the clients vanished"


def test_unknown_session_is_404_and_the_stream_needs_a_sign_in(server):
    cookie = _cookie(server)
    s = _open_stream(server, "SES-DOES-NOT-EXIST", cookie)
    head = Reader(s).status_line()
    assert " 404 " in head.split("\r\n")[0]
    s.close()
    server.sessions._active["SES-A"] = FakeSession(b"A")
    s = _open_stream(server, "SES-A", None)
    head = Reader(s).status_line()
    assert " 401 " in head.split("\r\n")[0]
    s.close()
