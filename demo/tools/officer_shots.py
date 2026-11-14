"""Drive the simple officer screen in headless Chrome (Playwright) and save screenshots - the one-click flow an officer would use.

  python demo/tools/officer_shots.py --card "Test: wrong-way" --waits 6 14 --out demo/runs/officer
  python demo/tools/officer_shots.py --base http://127.0.0.1:8760 --card "Real motorway" --waits 20 60

Needs the server started with `scripts/run_demo.py --one-click` (so no sign-in step). Prints the console errors and the tile/alert text it saw.
"""
import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8760")
ap.add_argument("--card", default="Real motorway")
ap.add_argument("--waits", type=float, nargs="+", default=[8.0])
ap.add_argument("--out", default="demo/runs/officer")
ap.add_argument("--w", type=int, default=1440)
ap.add_argument("--h", type=int, default=900)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
slug = a.card.lower().replace(" ", "_").replace(":", "")

with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport={"width": a.w, "height": a.h})
    logs = []
    pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
    pg.goto(a.base + "/")
    pg.wait_for_selector("#cards .card, #live:not([hidden])", timeout=15000)
    pg.screenshot(path=str(out / f"{slug}_0_chooser.png"))
    if pg.query_selector("#cards .card"):
        pg.get_by_role("button", name=a.card).first.click()
    last = 0.0
    for i, w in enumerate(a.waits, 1):
        time.sleep(max(0.0, w - last))
        last = w
        pg.screenshot(path=str(out / f"{slug}_{i}_{int(w)}s.png"))
        tiles = pg.eval_on_selector_all("#tiles .tile", "els => els.map(e => e.className.replace('tile ','') + ' | ' + e.querySelector('.tt').textContent + ' | ' + e.querySelector('.tv').textContent + ' | ' + e.querySelector('.td').textContent)")
        alerts = pg.eval_on_selector_all("#alerts .alert", "els => els.map(e => e.textContent.replace(/\\s+/g,' ').trim())")
        print(f"--- {w:.0f}s tiles:")
        for t in tiles:
            print("   ", t)
        print(f"--- {w:.0f}s alerts ({len(alerts)}):")
        for t in alerts[:8]:
            print("   ", t)
    print("console issues:", json.dumps(logs[:15], indent=1))
    b.close()
