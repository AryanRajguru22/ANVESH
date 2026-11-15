"""A local HTTP stub (127.0.0.1 only) for adapter tests -- never a real external service."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class StubServer:
    """`respond(path, headers, body) -> (status, dict|str|bytes, delay_s)`."""

    def __init__(self, respond):
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n)
                outer.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                status, payload, delay = respond(self.path, dict(self.headers), body)
                if delay:
                    time.sleep(delay)
                data = payload if isinstance(payload, (bytes, str)) else json.dumps(payload)
                data = data.encode() if isinstance(data, str) else data
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass  # the client gave up (timeout test)

            def log_message(self, *args):
                pass

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self._srv.server_address[1]}/hook"
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def close(self):
        self._srv.shutdown()
        self._srv.server_close()
