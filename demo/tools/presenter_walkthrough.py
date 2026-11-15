"""Clicks through the presenter path from demo/README.md in headless Chrome (no API shortcuts), screenshotting each step.
Run against `scripts/run_demo.py --reset` (default demo/data). Roles are switched by signing out and in.

  python demo/tools/presenter_walkthrough.py --data-dir demo/data --out demo/runs/walkthrough
"""
import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def creds(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            u, _r, pw = line.split("\t")
            out[u] = pw
    return out


def audit_step(login, pg, a, shot, log):
    """The audit trail (newest first): the administrator filters the sandbox hand-off entry and verifies the hash chain."""
    login("admin")
    pg.goto(a.base + "/#/history")
    time.sleep(2)
    pg.get_by_role("button", name="Audit log").click()
    time.sleep(2)
    shot("14_audit")
    box = pg.get_by_placeholder("action")
    box.fill("handoff.result")
    box.press("Enter")
    time.sleep(2)
    log("audit log shows the sandbox hand-off of this walkthrough (reference TEST-CHALLAN-...)", "handoff.result" in pg.inner_text("body") and "TEST-CHALLAN" in pg.inner_text("body"))
    shot("14b_audit_filtered")
    pg.get_by_role("button", name="Verify hash chain").click()
    try:
        pg.wait_for_selector("text=Audit chain intact", timeout=8000)
        log("audit hash chain verifies", True)
    except Exception as e:  # noqa: BLE001
        log("audit hash chain verifies", False, str(e)[:120])
    shot("15_audit_verified")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8760")
    ap.add_argument("--data-dir", default="demo/data")
    ap.add_argument("--out", default="demo/runs/walkthrough")
    ap.add_argument("--keep-real-running", action="store_true",
                    help="leave the real-video session running (the full 20-minute acceptance run); default stops it after a few seconds to keep the walkthrough short")
    ap.add_argument("--audit-only", action="store_true", help="only run the audit-trail step (against a data dir that already holds a walkthrough)")
    ap.add_argument("--viewport", default="1920x1080", help="WIDTHxHEIGHT of the browser window (a presentation screen)")
    a = ap.parse_args()
    vw, vh = (int(x) for x in a.viewport.split("x"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pw = creds(Path(a.data_dir) / "initial_credentials.txt")
    steps = []

    def log(step, ok, detail=""):
        steps.append({"step": step, "ok": bool(ok), "detail": str(detail)[:250]})
        print(("OK   " if ok else "FAIL ") + step, str(detail)[:160], flush=True)

    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page(viewport={"width": vw, "height": vh})
        js_errors = []
        pg.on("pageerror", lambda e: js_errors.append(str(e)))
        pg.on("console", lambda m: js_errors.append(m.text) if m.type == "error" and "401" not in m.text else None)
        pg.on("dialog", lambda d: d.accept())

        def shot(n):
            pg.screenshot(path=str(out / f"{n}.png"), full_page=True)

        def login(user):
            pg.goto(a.base + "/")
            time.sleep(1)
            if pg.query_selector("text=Sign out"):
                pg.click("text=Sign out")
                time.sleep(1)
            pg.wait_for_selector("#u", timeout=15000)
            pg.fill("#u", user)
            pg.fill("#p", pw[user])
            pg.keyboard.press("Enter")
            pg.wait_for_selector("text=Live monitoring", timeout=15000)

        def click_text(label, exact=False, timeout=8000):
            loc = pg.get_by_role("button", name=label, exact=exact)
            loc.first.click(timeout=timeout)
            time.sleep(1.5)

        if a.audit_only:
            audit_step(login, pg, a, shot, log)
            b.close()
            Path(out / "walkthrough_report.json").write_text(json.dumps(steps, indent=1))
            return
        # 0: a fresh demo shows nothing from earlier sessions
        login("investigator")
        pg.goto(a.base + "/#/incidents")
        time.sleep(2)
        log("fresh start: the incident list is empty (no stale incidents)", "No incidents match." in pg.inner_text("body"))
        shot("00_fresh_incidents")
        # 1-2: investigator starts the real video from the Cameras page
        pg.goto(a.base + "/#/cameras")
        time.sleep(2)
        shot("01_cameras")
        names = pg.inner_text("body")
        log("cameras page lists the demo camera and SIMULATED cameras", "Demo motorway" in names and "SIMULATED" in names)
        card = pg.locator(".card", has_text="Demo motorway").first
        card.get_by_role("button", name="Start").click()
        time.sleep(6)
        log("Start on the real video opens a running session", "Open live view" in pg.inner_text("body") or "#/live/" in pg.url, pg.url)
        pg.goto(a.base + "/#/live")
        time.sleep(2)
        pg.click("text=open")
        time.sleep(6)
        shot("02_live_real_video")
        t = pg.inner_text("body")
        log("live page shows the video being processed (progress, fps)", "Video " in t and "processing fps" in t)
        if not a.keep_real_running:  # stop it again to keep the walkthrough short
            pg.goto(a.base + "/#/cameras")
            time.sleep(2)
            pg.locator(".card", has_text="Demo motorway").first.get_by_role("button", name="Stop").click()
            time.sleep(3)
        # 5b: simulated queue (not a collision) and collision
        for cam in ("sim-queue", "sim-collision", "sim-wrong-way"):
            pg.goto(a.base + "/#/cameras")
            time.sleep(2)
            pg.locator(".card", has_text=cam).first.get_by_role("button", name="Start").click()
            time.sleep(1)
        time.sleep(50)
        pg.goto(a.base + "/#/incidents")
        time.sleep(3)
        shot("03_incidents")
        body = pg.inner_text("body")
        log("simulated collision candidate appears, labelled candidate + SIMULATED", "possible collision (candidate)" in body and "SIMULATED" in body)
        n_coll = pg.locator("tr.click", has_text="possible collision").count()
        log("exactly one collision candidate (the queue scenario produced none)", n_coll == 1, n_coll)
        n_ww = pg.locator("tr.click", has_text="wrong-way").count()
        log("the SIMULATED wrong-way camera produced one wrong-way candidate, labelled SIMULATED",
            n_ww == 1 and "SIMULATED" in pg.locator("tr.click", has_text="wrong-way").first.inner_text(), n_ww)
        n_q = pg.locator("tr.click", has_text="sim-queue").filter(has_text="queue member").count()
        log("the SIMULATED queue camera's stopped vehicles are marked queue members, none is a collision", n_q >= 3 and pg.locator("tr.click", has_text="sim-queue").filter(has_text="collision").count() == 0, n_q)
        # open a stopped-vehicle incident from the simulated collision camera and create a case
        rows = pg.locator("tr.click", has_text="stopped vehicle")
        log("stopped-vehicle incidents available for a case", rows.count() > 0, rows.count())
        rows.first.click()
        time.sleep(4)
        shot("04_incident_detail")
        pg.get_by_role("button", name="Open violation case from this incident").click()
        time.sleep(3)
        shot("05_case_created")
        case_url = pg.url
        log("case created", "#/cases/" in pg.url or "CASE-" in pg.inner_text("body"), pg.url)
        case_path = case_url.split("#", 1)[1]

        def act(label, note="reviewed in the presenter walkthrough"):
            """Click an action button, fill the note modal and Confirm."""
            pg.get_by_role("button", name=label).first.click(timeout=8000)
            time.sleep(1)
            ta = pg.query_selector(".modal textarea, [role=dialog] textarea, textarea")
            if ta:
                ta.fill(note)
            pg.get_by_role("button", name="Confirm").first.click(timeout=8000)
            time.sleep(2.5)

        def goto_case(user):
            login(user)
            pg.goto(a.base + "/#" + case_path)
            time.sleep(3)

        def step(name, fn):
            try:
                fn()
                log(name, True)
            except Exception as e:  # noqa: BLE001
                btns = [x.inner_text() for x in pg.query_selector_all("button")][:30]
                log(name, False, f"{str(e)[:120]} | buttons: {btns}")

        step("investigator: Submit for evidence verification", lambda: act("Submit for evidence verification"))
        shot("06_submitted")
        goto_case("reviewer")
        step("reviewer: Verify evidence", lambda: act("Verify evidence"))
        shot("07_evidence_verified")
        # investigator enters the plate on the incident page
        login("investigator")
        pg.goto(a.base + "/#" + case_path)
        time.sleep(3)
        pg.locator("a[href^='#/incidents/INC-']").first.click()
        time.sleep(3)

        def plate():
            pg.get_by_role("button", name="Enter plate manually").click()
            time.sleep(1)
            shot("08_plate_form")
            pg.get_by_placeholder("MH12AB1234").fill("MH12AB1234")
            pg.get_by_role("button", name="Save").first.click()
            time.sleep(2)
            assert "MH12AB1234" in pg.inner_text("body")
        step("investigator: Enter plate manually (DEMO PLACEHOLDER)", plate)
        shot("09_plate_entered")
        goto_case("reviewer2")
        step("reviewer2: Verify plate (a different reviewer)", lambda: act("Verify plate"))
        shot("10_plate_verified")
        goto_case("officer")
        import re
        step("officer: Registry lookup (SANDBOX)", lambda: pg.get_by_role("button", name=re.compile("Look up registration", re.I)).first.click() or time.sleep(3))
        shot("11_registry_sandbox")
        log("registry result is labelled SANDBOX", "SANDBOX" in pg.inner_text("body"))
        step("officer: Approve for submission", lambda: act("Approve for submission"))
        shot("12_approved")

        def handoff():
            pg.get_by_role("button", name="Hand off").first.click()
            time.sleep(1)
        btns = [x.inner_text() for x in pg.query_selector_all("button")]
        log("hand-off buttons offered", any("Submit to SANDBOX" in b for b in btns), btns)
        step("officer: hand-off to the SANDBOX adapter", lambda: (pg.get_by_role("button", name="Submit to SANDBOX simulator").first.click(), time.sleep(1.5),
                                                                pg.get_by_role("button", name="Confirm").first.click() if pg.query_selector("text=Confirm") else None, time.sleep(3)))
        shot("13_handoff")
        t = pg.inner_text("body")
        log("case reaches challan reference received (TEST- reference)", "TEST-CHALLAN" in t, "")
        audit_step(login, pg, a, shot, log)
        log("no JavaScript errors during the whole walkthrough", not js_errors, js_errors[:3])
        b.close()
    Path(out / "walkthrough_report.json").write_text(json.dumps(steps, indent=1))


if __name__ == "__main__":
    main()
