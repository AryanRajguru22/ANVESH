"""Captures the officer-screen screenshots used by the panel guide from a RUNNING demo (headless Chrome via Playwright, system Python).

    python docs/viva_assets/capture_screens.py --base http://127.0.0.1:8760

Writes docs/viva_assets/screens/officer_chooser.png, officer_live.png, officer_jam_sim.png. The live shot contains frames of demo.mp4 (licence
unverified), so the screens and the PDFs built from them are git-ignored."""
import argparse, time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / "screens"


def wait_video(pg, timeout):
    """Polls with evaluate (the page's CSP forbids the string eval that wait_for_function uses)."""
    end = time.time() + timeout
    while time.time() < end:
        if pg.evaluate("(() => { const v = document.getElementById('video'); return !!(v && v.naturalWidth > 0); })()"):
            return
        time.sleep(1)
    raise SystemExit("no video frame arrived")


def api(pg, m, p):
    return pg.evaluate("""async ([m, p]) => { const r = await fetch('/api/v1' + p, {method: m, headers: {'X-Requested-With': 'anvesh', 'Content-Type': 'application/json'}, credentials: 'same-origin'}); try { return await r.json(); } catch (e) { return null; } }""", [m, p])


ap = argparse.ArgumentParser(); ap.add_argument("--base", default="http://127.0.0.1:8760"); a = ap.parse_args()
OUT.mkdir(exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.goto(a.base + "/officer", wait_until="domcontentloaded"); time.sleep(3)
    for c in (api(pg, "GET", "/cameras") or {}).get("items", []):
        if c.get("active_session"):
            api(pg, "POST", f"/sessions/{c['active_session']['session_id']}/stop")
    time.sleep(2)
    pg.goto(a.base + "/officer", wait_until="domcontentloaded"); pg.wait_for_selector(".card[data-cam]", timeout=20000); time.sleep(1)
    pg.screenshot(path=str(OUT / "officer_chooser.png"))
    pg.click('.card[data-cam="demo-motorway"]'); wait_video(pg, 120); time.sleep(25)
    pg.screenshot(path=str(OUT / "officer_live.png"))
    pg.click('.src:has-text("Test: jam from a breakdown")'); wait_video(pg, 60); time.sleep(9)
    pg.screenshot(path=str(OUT / "officer_jam_sim.png"))
    time.sleep(1)
    b.close()
print("saved to", OUT)
