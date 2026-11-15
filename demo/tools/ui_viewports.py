"""Responsive check of the officer screen (/officer) and the full dashboard live view (/advanced#/live) against a RUNNING demo.

For each viewport (320, 375, 390, 768, 1024, 1440 px wide) it opens the page in headless Chrome (Playwright, system Python), starts the real
motorway session on the officer screen, and checks: no sideways scrolling (STRICT: scrollWidth must equal the window width), no element sticking out
of the right edge, the video element is on screen and at least `--min-video-w` px wide, and no JavaScript error. It saves one screenshot per size.

  python demo/tools/ui_viewports.py --base http://127.0.0.1:8760 --out demo/runs/ui_viewports

Exit code 1 if any check fails. Needs a demo started with scripts/run_demo.py --one-click (loopback sign-in).
"""
import argparse
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

SIZES = [(320, 640), (375, 812), (390, 844), (768, 1024), (1024, 768), (1440, 900)]
AUDIT = """() => {
  const W = innerWidth, o = {sw: document.documentElement.scrollWidth, W, clip: [], video: null};
  for (const e of document.querySelectorAll('body *')) {
    const r = e.getBoundingClientRect();
    if (!r.width || !r.height) continue;
    if (r.right > W + 0.5 && !e.closest('.table-wrap, .srcbar') && getComputedStyle(e).position !== 'fixed') {
      o.clip.push(e.tagName + '.' + String(e.className).slice(0, 25)); if (o.clip.length > 6) break;
    }
  }
  const v = document.querySelector('#video, .lv-media');
  if (v) { const r = v.getBoundingClientRect(); o.video = {top: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height), onscreen: r.top < innerHeight * 0.75 && r.bottom > 0}; }
  return o;
}"""


def run(base, out, min_video_w):
    out.mkdir(parents=True, exist_ok=True)
    rows, failed = [], False
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        for name, path, click in (("officer", "/officer", '.card[data-cam="demo-motorway"]'), ("advanced", "/officer", None)):
            for w, h in SIZES:
                pg = b.new_page(viewport={"width": w, "height": h})
                errs = []
                pg.on("pageerror", lambda e: errs.append(str(e)))
                pg.goto(base + "/officer", wait_until="domcontentloaded")
                time.sleep(2)
                if click:
                    try:
                        pg.click(click, timeout=4000)  # first size starts the session; later sizes attach to the running one
                    except Exception:  # noqa: BLE001 - already attached to a running session
                        pass
                if name == "advanced":
                    pg.goto(base + "/advanced#/live", wait_until="domcontentloaded")
                time.sleep(8)
                a = pg.evaluate(AUDIT)
                a.update(page=name, size=[w, h], errs=errs)
                ok = a["sw"] <= a["W"] and not a["clip"] and not errs and a["video"] is not None and a["video"]["onscreen"] and a["video"]["w"] >= min(min_video_w, w - 24)
                a["ok"] = ok
                failed |= not ok
                rows.append(a)
                pg.screenshot(path=str(out / f"{name}_{w}x{h}.png"))
                pg.close()
        b.close()
    (out / "report.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for r in rows:
        v = r["video"] or {}
        print(f"{r['page']:9s} {r['size'][0]:>4}x{r['size'][1]:<4} overflow={r['sw'] - r['W']:>2} clipped={len(r['clip'])} video={v.get('w')}x{v.get('h')} top={v.get('top')} errors={len(r['errs'])}  {'OK' if r['ok'] else 'FAIL'}")
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8760")
    ap.add_argument("--out", default="demo/runs/ui_viewports")
    ap.add_argument("--min-video-w", type=int, default=280)
    a = ap.parse_args()
    sys.exit(run(a.base, Path(a.out), a.min_video_w))
