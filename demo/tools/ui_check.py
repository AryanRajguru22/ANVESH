"""Drives the real dashboard in headless Chrome (Playwright, system Chrome) and saves screenshots. Used for browser verification
because the Claude-in-Chrome extension is not always connected.

  python demo/tools/ui_check.py --base http://127.0.0.1:8760 --data-dir demo/data-run1 --user investigator --route "#/live" --out demo/runs/shots
"""
import argparse, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

def creds(path, user):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            u, _r, pw = line.split("\t")
            if u == user: return pw
    raise SystemExit("no such user")

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8760"); ap.add_argument("--data-dir", default="demo/data")
ap.add_argument("--user", default="investigator"); ap.add_argument("--route", action="append")
ap.add_argument("--out", default="demo/runs/shots"); ap.add_argument("--wait", type=float, default=3.0)
ap.add_argument("--w", type=int, default=1440); ap.add_argument("--h", type=int, default=900)
a = ap.parse_args(); a.route = a.route or ["#/live"]
Path(a.out).mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport={"width": a.w, "height": a.h})
    logs = []
    pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
    pg.goto(a.base + "/")
    pg.wait_for_selector("input", timeout=10000)
    inputs = pg.query_selector_all("input")
    print("inputs:", [(i.get_attribute("name"), i.get_attribute("type"), i.get_attribute("id")) for i in inputs])
    pg.fill("input[type=text], input:not([type])", a.user) if pg.query_selector("input[type=text], input:not([type])") else None
    pg.fill("input[type=password]", creds(Path(a.data_dir) / "initial_credentials.txt", a.user))
    pg.keyboard.press("Enter"); time.sleep(2)
    print("after login url:", pg.url, "| title:", pg.title())
    pg.screenshot(path=f"{a.out}/after_login.png")
    for i, r in enumerate(a.route):
        pg.goto(a.base + "/" + r if r.startswith("#") else a.base + r); time.sleep(a.wait)
        name = r.strip("#/").replace("/", "_") or "home"
        pg.screenshot(path=f"{a.out}/{i:02d}_{name}.png", full_page=False)
        print("shot", r)
    print("console issues:", logs[:20])
    b.close()
