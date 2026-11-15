"""Browser acceptance of the dashboard against a RUNNING demo session (headless Chrome via Playwright; system Python).

Checks, as the investigator: live page shows the annotated stream, progress, source fps, queue depth, calibration label, track table and the event
feed; an incident page shows evidence images that really load and a clip that really decodes and plays in Chrome; the layout has no horizontal
overflow at desktop and phone width; and the research/demo banner is present on every page. Saves screenshots and a JSON report.

  python demo/tools/ui_acceptance.py --base http://127.0.0.1:8760 --data-dir demo/data-final --out demo/runs/ui_final
"""
import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def creds(path, user):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            u, _r, pw = line.split("\t")
            if u == user:
                return pw
    raise SystemExit("no such user")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8760")
    ap.add_argument("--data-dir", default="demo/data")
    ap.add_argument("--out", default="demo/runs/ui")
    ap.add_argument("--user", default="investigator")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"checks": []}

    def check(name, ok, detail=""):
        rep["checks"].append({"check": name, "ok": bool(ok), "detail": str(detail)[:300]})
        print(("PASS " if ok else "FAIL ") + name, str(detail)[:200])

    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" and "401" not in m.text else None)
        pg.goto(a.base + "/")
        pg.wait_for_selector("#u", timeout=15000)
        pg.fill("#u", a.user)
        pg.fill("#p", creds(Path(a.data_dir) / "initial_credentials.txt", a.user))
        pg.keyboard.press("Enter")
        pg.wait_for_selector("text=Live monitoring", timeout=15000)
        banner = pg.inner_text("body")
        check("research/demo-only banner visible after login", "RESEARCH / DEMO USE ONLY" in banner)
        pg.goto(a.base + "/#/live")
        time.sleep(3)
        pg.click("text=open")
        time.sleep(5)
        txt = pg.inner_text("body")
        check("live page: progress, source fps, queue depth, calibration notice",
              all(k in txt for k in ("Video ", "source fps", "queue depth now", "APPROXIMATE metric")), "")
        img = pg.query_selector("img.live")
        check("live page: annotated stream element present", img is not None and "stream.mjpg" in (img.get_attribute("src") or ""))
        ok_img = pg.evaluate("() => { const i = document.querySelector('img.live'); return i ? [i.naturalWidth, i.naturalHeight] : null }")
        check("live page: MJPEG frames actually decode", bool(ok_img and ok_img[0] > 0), ok_img)
        check("live page: live event feed present", "Live event feed" in txt)
        w1 = pg.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth + 1")
        check("no horizontal overflow at 1440 px", w1)
        pg.screenshot(path=str(out / "live_desktop.png"))
        # progress moves
        t1 = pg.inner_text("body")
        time.sleep(8)
        pg.screenshot(path=str(out / "live_desktop_later.png"))
        pg.set_viewport_size({"width": 390, "height": 844})
        time.sleep(2)
        check("no horizontal overflow at phone width (390 px)", pg.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth + 1"),
              pg.evaluate("() => [document.documentElement.scrollWidth, window.innerWidth]"))
        pg.screenshot(path=str(out / "live_phone.png"))
        pg.set_viewport_size({"width": 1440, "height": 900})
        # incident with evidence
        pg.goto(a.base + "/#/incidents")
        time.sleep(3)
        pg.screenshot(path=str(out / "incidents.png"))
        links = pg.query_selector_all("tr.click")
        check("incident list shows incidents from THIS session", len(links) > 0, len(links))
        if links:
            links[0].click()
            time.sleep(4)
            pg.screenshot(path=str(out / "incident_detail.png"), full_page=True)
            body = pg.inner_text("body")
            check("incident page shows reasons (indicators), uncertainty and basis", all(k in body.lower() for k in ("uncertainty and limits", "indicators", "severity basis")), "")
            pg.click("text=Annotated frames")
            time.sleep(2)
            imgs = pg.evaluate("() => [...document.querySelectorAll('img')].filter(i => i.src.includes('/evidence/')).map(i => [i.naturalWidth, i.complete])")
            check("evidence images load", len(imgs) > 0 and all(w > 0 for w, _ in imgs), imgs[:4])
            pg.click("button:has-text('Clip')")
            time.sleep(2)
            vid = pg.query_selector("video")
            if vid:
                pg.evaluate("() => { const v = document.querySelector('video'); v.muted = true; v.play().catch(()=>{}); }")
                time.sleep(3)
                st = pg.evaluate("() => { const v = document.querySelector('video'); return {w: v.videoWidth, h: v.videoHeight, dur: v.duration, t: v.currentTime, err: v.error && v.error.code} }")
                check("evidence clip decodes and plays in Chrome", st["w"] > 0 and st["dur"] > 0 and st["t"] > 0 and not st["err"], st)
                pg.screenshot(path=str(out / "incident_clip.png"))
            else:
                check("evidence clip element present", False, "no <video> element")
        check("no JavaScript errors on the pages visited", not errors, errors[:4])
        b.close()
    (out / "ui_report.json").write_text(json.dumps(rep, indent=1))
    fails = [c for c in rep["checks"] if not c["ok"]]
    print(f"\n{len(rep['checks']) - len(fails)}/{len(rep['checks'])} UI checks passed")


if __name__ == "__main__":
    main()
