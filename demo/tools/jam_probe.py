"""Start a session through the API on a (fast-mode) demo server and log what the officer layer reports while the video runs.
Saves the annotated frame whenever the jam verdict changes. For checking the jam-cause candidate on real footage.

  scripts/run_demo.py --reset --fast --seconds 330 --port 8761 --data-dir demo/data-test --no-simulated
  python demo/tools/jam_probe.py --port 8761 --data-dir demo/data-test --camera demo-motorway --out demo/runs/jam_probe
"""
import argparse
import http.cookiejar
import json
import time
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8761)
ap.add_argument("--data-dir", default="demo/data-test")
ap.add_argument("--camera", default="demo-motorway")
ap.add_argument("--out", default="demo/runs/jam_probe")
ap.add_argument("--every", type=float, default=3.0)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
base = f"http://127.0.0.1:{a.port}/api/v1"
creds = {}
for line in (Path(a.data_dir) / "initial_credentials.txt").read_text(encoding="utf-8").splitlines():
    p = line.split("\t")
    if len(p) == 3 and not line.startswith("#"):
        creds[p[0]] = p[2]
jar = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def call(method, path, body=None, raw=False):
    req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", "X-Requested-With": "anvesh"})
    with op.open(req, timeout=30) as r:
        data = r.read()
    return data if raw else json.loads(data)


call("POST", "/auth/login", {"username": "investigator", "password": creds["investigator"]})
sid = call("POST", f"/cameras/{a.camera}/sessions")["session_id"]
print("session", sid)
last_key, n = None, 0
while True:
    time.sleep(a.every)
    try:
        st = call("GET", f"/sessions/{sid}/officer")
    except Exception as e:  # noqa: BLE001
        print("poll error", e)
        break
    jam = st["jam"]
    key = (jam.get("active"), jam.get("cause_track"), jam.get("rating"))
    line = f"t={st['t']:7.1f} state={st['state']:9s} fps={st['fps']:5.1f} veh={st['vehicles']:2d} jam={jam.get('active')} n={jam.get('vehicles')} cause={jam.get('cause_label')} {jam.get('rating')} first={jam.get('stopped_first')} corr={jam.get('order_correlation')}"
    print(line, flush=True)
    if key != last_key:
        last_key = key
        try:
            (out / f"jam_{n:02d}_t{int(st['t'])}.jpg").write_bytes(call("GET", f"/sessions/{sid}/snapshot.jpg", raw=True))
            n += 1
        except Exception as e:  # noqa: BLE001
            print("snapshot error", e)
    if st["state"] in ("completed", "stopped", "failed"):
        break
print("alerts:")
for al in st["alerts"][::-1]:
    print("  ", al["t"], al["kind"], "|", al["detail"])
print("counts", st["alert_counts"])
