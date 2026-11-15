"""Browser-level lifecycle check of the officer screen against a RUNNING demo (headless Chrome via Playwright, system Python).

It drives the real page and the real server (no mocks): start the real motorway session -> frames arrive and the video grows -> Pause shows the
paused state -> Resume -> Stop shows the stopped state and the server really ends the session -> switch sources 8 times quickly (more than the
6 connections a browser allows per host, so a leaked video stream would freeze the page) and confirm polling still answers and exactly one session
is running at the end.

  python demo/tools/ui_lifecycle.py --base http://127.0.0.1:8760

Exit code 1 if any step fails. Needs a demo started with scripts/run_demo.py --one-click.
"""
import argparse
import json
import sys
import time

from playwright.sync_api import sync_playwright

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def api(pg, method, path):
    return pg.evaluate(
        """async ([m, p]) => { const r = await fetch('/api/v1' + p, {method: m, headers: {'X-Requested-With': 'anvesh', 'Content-Type': 'application/json'}, credentials: 'same-origin'});
        let b = null; try { b = await r.json(); } catch (e) {} return {status: r.status, body: b}; }""", [method, path])


def wait_for(pg, js, timeout=40):
    end = time.time() + timeout
    while time.time() < end:
        try:
            v = pg.evaluate(js)
        except Exception:  # noqa: BLE001 - page navigating
            v = None
        if v:
            return v
        time.sleep(0.5)
    return None


def main(base):
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(base + "/officer", wait_until="domcontentloaded")
        pg.wait_for_selector(".card[data-cam], #live:not([hidden])", timeout=20000)
        # the page may attach to a session already running; stop everything first so the test starts clean
        for c in (api(pg, "GET", "/cameras")["body"] or {}).get("items", []):
            if c.get("active_session"):
                api(pg, "POST", f"/sessions/{c['active_session']['session_id']}/stop")
        time.sleep(1.5)
        pg.goto(base + "/officer", wait_until="domcontentloaded")
        pg.click('.card[data-cam="demo-motorway"]', timeout=15000)

        ok = wait_for(pg, "(() => { const v = document.getElementById('video'); return v && v.naturalWidth > 0; })()", 90)
        check("video frames arrive in the page (real stream)", ok)
        pill = wait_for(pg, "(() => { const p = document.querySelector('#statusline .pill-state'); return p && p.textContent.trim() === 'LIVE' ? 'LIVE' : null; })()", 30)
        check("status line shows LIVE with camera and session id", pill == "LIVE" and pg.inner_text("#statusline").count("SES-") == 1, pg.inner_text("#statusline").replace("\n", " | "))
        sid = pg.inner_text("#statusline .sl-id").strip()
        t1 = wait_for(pg, "(() => { const t = document.getElementById('clock').textContent; return t.startsWith('00:0') && t > '00:02' ? t : null; })()", 30)
        check("video clock advances while processing", t1, str(t1))

        pg.click("#pauseBtn")
        paused = wait_for(pg, "(() => document.querySelector('#statusline .pill-state').textContent.trim() === 'PAUSED' && !document.getElementById('stageMsg').hidden)()", 15)
        check("pause shows PAUSED and a message over the video", paused, pg.inner_text("#stageMsg") if paused else "")
        st = api(pg, "GET", f"/sessions/{sid}/officer")["body"]
        check("server reports the session paused", st and st["state"] == "paused", st and st["state"])
        pg.click("#pauseBtn")
        resumed = wait_for(pg, "(() => document.querySelector('#statusline .pill-state').textContent.trim() === 'LIVE')()", 15)
        check("resume returns to LIVE", resumed)

        pg.click("#stopBtn")
        stopped = wait_for(pg, "(() => !document.getElementById('endCard').hidden && document.getElementById('endTitle').textContent === 'Stopped')()", 20)
        check("stop shows the Stopped end card (not 'Finished')", stopped, pg.inner_text("#endCard").replace("\n", " ") if stopped else "")
        st = api(pg, "GET", f"/sessions/{sid}/officer")["body"]
        check("server reports the session stopped", st and st["state"] == "stopped", st and st["state"])
        pg.evaluate(f"""fetch('/api/v1/sessions/{sid}/stream.mjpg', {{headers: {{'X-Requested-With': 'anvesh'}}, credentials: 'same-origin'}}).then(r => r.arrayBuffer().then(b => {{ window.__sr = r.status; window.__len = b.byteLength; }}))""")
        time.sleep(4)
        check("a stopped session's stream request runs to its end (it is not an endless feed)", pg.evaluate("window.__sr !== undefined"), f"status {pg.evaluate('window.__sr')}, {pg.evaluate('window.__len')} bytes after the stop")

        # rapid source switching: more than 6 switches with video streams open must not exhaust the browser's connections
        pg.click("#againBtn")
        time.sleep(2)
        names = ["Test: jam from a breakdown", "Test: wrong-way driver", "Test: accident", "Real motorway CCTV"] * 2
        for n in names:
            pg.click(f'.src:has-text("{n}")', timeout=15000)
            time.sleep(1.2)
        time.sleep(3)
        answered = pg.evaluate("""fetch('/api/v1/cameras', {headers: {'X-Requested-With': 'anvesh'}, credentials: 'same-origin'}).then(r => r.status)""")
        check("after 8 quick source switches the server still answers", answered == 200, str(answered))
        cams = (api(pg, "GET", "/cameras")["body"] or {}).get("items", [])
        running = [c["camera_id"] for c in cams if c.get("active_session") and c["active_session"]["state"] in ("starting", "connecting", "live", "stalled", "reconnecting", "paused")]
        check("only the selected source is processing (one session at a time)", running == ["demo-motorway"], str(running))
        check("no JavaScript error was raised", not errs, "; ".join(errs)[:200])
        api(pg, "POST", f"/sessions/{[c for c in cams if c['camera_id'] == 'demo-motorway'][0]['active_session']['session_id']}/stop")
        b.close()
    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8760")
    sys.exit(main(ap.parse_args().base))
