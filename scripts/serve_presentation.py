"""Serve the repository root over HTTP with HTTP Range-request support --
needed for the live dashboard's <video> element (seeking/scrubbing relies
on the browser being able to request byte ranges of the video file).

Python's stdlib `http.server` (used ad hoc via `python -m http.server`)
does NOT support Range requests: it always returns the full file with a
plain 200, ignoring any `Range` header. Browsers still cope for playback
that just runs from the start, but seeking/scrubbing a large external
video (the live dashboard's demo.mp4 / analyzed clip) is unreliable
without proper 206 Partial Content support, and some browsers/OSes leave
a `<video>` element sitting at readyState=HAVE_NOTHING for a long time
waiting on a range it never gets. This server fixes that with ~stdlib-only
code (no extra dependency, no internet access needed to run it).

Usage (from the repository root):

    python scripts/serve_presentation.py [port]   # default port 8000

Then open, e.g.:
    http://localhost:8000/ui/live/anvesh_dashboard_live.html
    http://localhost:8000/ui/live/anvesh_live_dashboard.html
"""

from __future__ import annotations

import http.server
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def send_head(self):
        path = self.translate_path(self.path)
        if not Path(path).is_file():
            return super().send_head()

        range_header = self.headers.get("Range")
        file_size = Path(path).stat().st_size
        if not range_header:
            f = open(path, "rb")
            self.send_response(200)
            self.send_header("Content-type", self.guess_type(path))
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return f

        match = re.match(r"bytes=(\d*)-(\d*)", range_header)
        if not match:
            self.send_error(416, "Invalid Range header")
            return None
        start_s, end_s = match.groups()
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
        end = min(end, file_size - 1)
        if start > end or start >= file_size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.end_headers()
            return None

        length = end - start + 1
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-type", self.guess_type(path))
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._range = (start, length)
        return f

    def copyfile(self, source, outputfile):
        if hasattr(self, "_range"):
            start, length = self._range
            remaining = length
            chunk_size = 1 << 20
            while remaining > 0:
                chunk = source.read(min(chunk_size, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        else:
            super().copyfile(source, outputfile)


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = http.server.ThreadingHTTPServer(("localhost", port), RangeRequestHandler)
    print(f"[serve] Serving {REPO_ROOT} at http://localhost:{port}/ (Range requests supported, Ctrl+C to stop)")
    print(f"[serve]   Full real-video live dashboard: http://localhost:{port}/ui/live/anvesh_dashboard_live.html")
    print(f"[serve]   Judge-video live dashboard:      http://localhost:{port}/ui/live/anvesh_live_dashboard.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
