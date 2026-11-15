"""Page-by-page layout check at the sizes a presentation is likely to use, against a RUNNING demo (headless Chrome via Playwright; system Python).

For every page and every size it checks: the RESEARCH / DEMO banner is present, nothing makes the page scroll sideways, no <img> is broken,
no element is clipped by the right edge of the window, and no JavaScript error was logged. It saves one screenshot per page and size.

  python demo/tools/ui_resolutions.py --base http://127.0.0.1:8760 --data-dir demo/data --out demo/runs/ui_resolutions
"""
import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

SIZES = [(1920, 1080), (1366, 768), (1280, 720), (1024, 768)]
PAGES = [("overview", "#/overview"), ("cameras", "#/cameras"), ("live list", "#/live"), ("incidents", "#/incidents"), ("congestion", "#/congestion"),
         ("cases", "#/cases"), ("history", "#/history")]

JS_AUDIT = """() => {
  const W = window.innerWidth, out = {sw: document.documentElement.scrollWidth, W, broken: [], clipped: []};
  for (const i of document.images) if (i.complete && i.naturalWidth === 0 && i.src) out.broken.push(i.src.slice(-60));
  for (const e of document.querySelectorAll('body *')) {
    const r = e.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    if (r.right > W + 2 && !e.closest('.table-wrap') && getComputedStyle(e).position !== 'fixed') {
      out.clipped.push(e.tagName + '.' + (e.className || '').toString().slice(0, 30));
      if (out.clipped.length > 5) break;
    }
  }
  return out;
}"""


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
    ap.add_argument("--out", default="demo/runs/ui_resolutions")
    ap.add_argument("--user", default="admin")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page(viewport={"width": 1920, "height": 1080})
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" and "401" not in m.text else None)
        pg.goto(a.base + "/")
        pg.wait_for_selector("#u", timeout=15000)
        pg.fill("#u", a.user)
        pg.fill("#p", creds(Path(a.data_dir) / "initial_credentials.txt", a.user))
        pg.keyboard.press("Enter")
        pg.wait_for_selector("text=Live monitoring", timeout=15000)
        # the detail pages need a real id
        pg.goto(a.base + "/#/incidents")
        time.sleep(2)
        first = pg.query_selector("tr.click")
        pages = list(PAGES)
        if first:
            first.click()
            time.sleep(3)
            pages.append(("incident detail", "#" + pg.url.split("#", 1)[1]))
        pg.goto(a.base + "/#/live")
        time.sleep(2)
        link = pg.query_selector("a:has-text('open')")
        if link:
            link.click()
            time.sleep(4)
            pages.append(("live session", "#" + pg.url.split("#", 1)[1]))
        for w, h in SIZES:
            pg.set_viewport_size({"width": w, "height": h})
            for name, route in pages:
                pg.goto(a.base + "/" + route)
                time.sleep(2.5)
                body = pg.inner_text("body")
                audit = pg.evaluate(JS_AUDIT)
                ok = ("RESEARCH / DEMO USE ONLY" in body and audit["sw"] <= audit["W"] + 1 and not audit["broken"] and not audit["clipped"])
                rows.append({"size": f"{w}x{h}", "page": name, "ok": ok, "banner": "RESEARCH / DEMO USE ONLY" in body, "scrollWidth": audit["sw"],
                             "innerWidth": audit["W"], "broken_images": audit["broken"], "clipped": audit["clipped"]})
                print(("PASS " if ok else "FAIL ") + f"{w}x{h} {name}", "" if ok else json.dumps(rows[-1]), flush=True)
                pg.screenshot(path=str(out / f"{w}x{h}_{name.replace(' ', '_')}.png"))
        b.close()
    res = {"pages_checked": len(rows), "failures": [r for r in rows if not r["ok"]], "js_errors": errors[:10], "rows": rows}
    (out / "ui_resolutions.json").write_text(json.dumps(res, indent=1))
    print(f"\n{len(rows) - len(res['failures'])}/{len(rows)} page-size checks passed; JS errors: {len(errors)}")


if __name__ == "__main__":
    main()
