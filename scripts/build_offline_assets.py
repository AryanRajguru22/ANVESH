"""Vendor the live dashboard's external dependencies (Tailwind CSS, Google
Fonts, Material Symbols) into ui/live/vendor/ so the LIVE presentation
artifact (scripts/analyze_video.py's output) needs no internet access at
presentation time.

This is a one-time (or "re-run when the template's markup/config changes")
build step, run with internet access available -- NOT at presentation time.
It never touches the protected static demo (ui/anvesh_dashboard.html,
ui/data/anvesh_demo_data.json), which keeps depending on the CDN exactly as
before (out of scope -- it is frozen).

Produces:
  ui/live/vendor/tailwind.css   -- statically compiled (no runtime JIT/CDN),
                                    scanned against the real template markup
                                    so every utility class actually used
                                    (including arbitrary-value classes like
                                    `text-[16px]`) is included.
  ui/live/vendor/fonts.css      -- @font-face rules pointing at local files
  ui/live/vendor/fonts/*.woff2  -- the actual font files, downloaded once
                                    from Google Fonts

Usage (from the repository root, with internet access and Node available):

    python scripts/build_offline_assets.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = REPO_ROOT / "ui" / "anvesh_dashboard.template.html"
VENDOR_DIR = REPO_ROOT / "ui" / "live" / "vendor"
FONTS_DIR = VENDOR_DIR / "fonts"

# Mirrors the inline `tailwind.config` object in
# ui/anvesh_dashboard.template.html exactly (colors, fontFamily, fontSize,
# spacing). Keep these in sync if that object ever changes.
TAILWIND_CONFIG_JS = """/** Mirrors the inline tailwind.config in ui/anvesh_dashboard.template.html. */
module.exports = {
  darkMode: "class",
  content: ["../../anvesh_dashboard.template.html"],
  theme: {
    extend: {
      colors: {
        "surface": "#0c141f", "surface-dim": "#0c141f", "surface-bright": "#323a46",
        "surface-container-lowest": "#070f19", "surface-container-low": "#141c27",
        "surface-container": "#18202b", "surface-container-high": "#232a36",
        "surface-container-highest": "#2d3541", "surface-variant": "#2d3541",
        "on-surface": "#dbe3f3", "on-surface-variant": "#bdc8d1",
        "inverse-surface": "#dbe3f3", "inverse-on-surface": "#29313d",
        "outline": "#87929a", "outline-variant": "#3e484f",
        "primary": "#8ed5ff", "on-primary": "#00354a", "primary-container": "#38bdf8", "on-primary-container": "#004965",
        "secondary": "#4edea3", "on-secondary": "#003824", "secondary-container": "#00a572", "on-secondary-container": "#00311f",
        "tertiary": "#ffc174", "on-tertiary": "#472a00", "tertiary-container": "#f59e0b", "on-tertiary-container": "#613b00",
        "error": "#ef4444", "on-error": "#070d14", "error-container": "#93000a", "on-error-container": "#ffdad6",
        "background": "#0c141f", "on-background": "#dbe3f3"
      },
      fontFamily: {
        sans: ["Hanken Grotesk", "-apple-system", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"]
      },
      fontSize: {
        "body-sm": ["11px", { lineHeight: "16px", letterSpacing: "0.01em" }],
        "label-caps": ["9px", { lineHeight: "12px", letterSpacing: "0.08em", fontWeight: "700" }],
        "body-md": ["13px", { lineHeight: "18px" }],
        "data-lg": ["16px", { lineHeight: "20px", letterSpacing: "-0.02em", fontWeight: "600" }],
        "headline-lg": ["18px", { lineHeight: "24px", letterSpacing: "0.04em", fontWeight: "700" }],
        "data-md": ["12px", { lineHeight: "16px", letterSpacing: "-0.01em", fontWeight: "500" }],
        "data-sm": ["10px", { lineHeight: "14px" }],
        "headline-sm": ["13px", { lineHeight: "18px", letterSpacing: "0.02em", fontWeight: "600" }],
        "headline-md": ["15px", { lineHeight: "20px", letterSpacing: "0.02em", fontWeight: "600" }],
        "label-sm": ["11px", { lineHeight: "14px", letterSpacing: "0.02em", fontWeight: "500" }]
      },
      spacing: { "2xs": "2px", xs: "4px", sm: "8px", md: "12px", lg: "16px", xl: "24px" }
    }
  }
};
"""

TAILWIND_INPUT_CSS = "@tailwind base;\n@tailwind components;\n@tailwind utilities;\n"

# The exact two Google Fonts CSS2 URLs already used by the template's <link>
# tags -- unchanged, so the downloaded files match what the template expects.
FONT_CSS_URLS = [
    "https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap",
    "https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap",
]

# A modern-browser UA is required to get woff2 (not woff/ttf) URLs back from
# Google Fonts' UA-sniffed CSS endpoint.
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

_URL_RE = re.compile(r"url\((https://fonts\.gstatic\.com/[^)]+)\)")


def build_tailwind_css() -> None:
    config_path = VENDOR_DIR / "tailwind.config.js"
    input_path = VENDOR_DIR / "_tailwind_input.css"
    output_path = VENDOR_DIR / "tailwind.css"
    config_path.write_text(TAILWIND_CONFIG_JS, encoding="utf-8")
    input_path.write_text(TAILWIND_INPUT_CSS, encoding="utf-8")
    print(f"[offline-assets] Compiling Tailwind CSS from {TEMPLATE_PATH.name} markup...")
    subprocess.run(
        [
            "npx", "--yes", "tailwindcss@3",
            "-i", str(input_path), "-o", str(output_path),
            "-c", str(config_path), "--minify",
        ],
        cwd=str(VENDOR_DIR), check=True, shell=(sys.platform == "win32"),
    )
    input_path.unlink()
    print(f"[offline-assets] Wrote {output_path} ({output_path.stat().st_size} bytes)")


def _fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def build_fonts() -> None:
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    combined_css_parts = []
    seen_filenames = set()
    for css_url in FONT_CSS_URLS:
        print(f"[offline-assets] Fetching {css_url}")
        css_text = _fetch_bytes(css_url).decode("utf-8")
        rewritten = css_text
        for font_url in _URL_RE.findall(css_text):
            filename = font_url.rsplit("/", 1)[-1]
            local_path = FONTS_DIR / filename
            if filename not in seen_filenames:
                seen_filenames.add(filename)
                if not local_path.is_file():
                    font_bytes = _fetch_bytes(font_url)
                    local_path.write_bytes(font_bytes)
                    print(f"[offline-assets]   -> fonts/{filename} ({len(font_bytes)} bytes)")
            rewritten = rewritten.replace(font_url, f"fonts/{filename}")
        combined_css_parts.append(rewritten)
    fonts_css_path = VENDOR_DIR / "fonts.css"
    fonts_css_path.write_text("\n".join(combined_css_parts), encoding="utf-8")
    print(f"[offline-assets] Wrote {fonts_css_path} ({fonts_css_path.stat().st_size} bytes), {len(seen_filenames)} font files")


def main() -> int:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    build_tailwind_css()
    build_fonts()
    print("[offline-assets] Done. ui/live/vendor/ is self-contained -- no CDN needed at presentation time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
