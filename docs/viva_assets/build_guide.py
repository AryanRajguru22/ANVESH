"""Builds docs/ANVESH_PANEL_PRESENTATION_GUIDE.pdf/.docx (helper script, run with system Python)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import content_guide1, content_guide2, content_guide3
from docbuilder import build_pdf, build_docx
OUT = Path(__file__).resolve().parents[1]
blocks = content_guide1.blocks + content_guide2.blocks + content_guide3.blocks
meta = ["Project: ANVESH - Intelligent Road Event Investigation & Traffic Intelligence", "Presentation: 2 November 2026 (updated 15 November 2026 for the officer screen)   |   Runbook: demo/README.md (13 steps)",
        "Companion document: Complete Technical Viva Handbook", "Demo launch: .venv\\Scripts\\python.exe scripts\\run_demo.py --reset   ->   http://127.0.0.1:8760/"]
note = "RESEARCH / DEMONSTRATION ONLY. Every registry and challan result in the demo is SIMULATED. The demo video's origin and licence are unverified (classroom use only)."
title, sub = "Panel Presentation and Live Demonstration Guide", "Your speaking script, click-path and answers for the evaluation panel"
foot = "ANVESH - Panel Presentation Guide"
build_pdf(blocks, OUT / "ANVESH_PANEL_PRESENTATION_GUIDE.pdf", title, sub, meta, foot, cover_note=note)
build_docx(blocks, OUT / "ANVESH_PANEL_PRESENTATION_GUIDE.docx", title, sub, meta, foot, cover_note=note)
print("built")
