"""Save the current annotated frame of the newest session of a demo server: python demo/tools/snap.py --port 8761 --data-dir demo/data-test --out file.jpg"""
import argparse
import http.cookiejar
import json
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8761)
ap.add_argument("--data-dir", default="demo/data-test")
ap.add_argument("--out", required=True)
a = ap.parse_args()
creds = {}
for line in (Path(a.data_dir) / "initial_credentials.txt").read_text(encoding="utf-8").splitlines():
    p = line.split("\t")
    if len(p) == 3 and not line.startswith("#"):
        creds[p[0]] = p[2]
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
base = f"http://127.0.0.1:{a.port}/api/v1"


def call(method, path, body=None, raw=False):
    req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", "X-Requested-With": "anvesh"})
    d = op.open(req, timeout=30).read()
    return d if raw else json.loads(d)


call("POST", "/auth/login", {"username": "investigator", "password": creds["investigator"]})
sid = call("GET", "/sessions?limit=1")["items"][0]["session_id"]
Path(a.out).write_bytes(call("GET", f"/sessions/{sid}/snapshot.jpg", raw=True))
print("saved", a.out, sid)
