"""Builds docs/ANVESH_COMPLETE_TECHNICAL_VIVA_HANDBOOK.pdf/.docx from content_hb*.py (helper script, run with system Python)."""
import importlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from docbuilder import build_pdf, build_docx
OUT = Path(__file__).resolve().parents[1]
blocks, glossary = [], []
for name in ("content_hb1", "content_hb2", "content_hb3", "content_hb4", "content_hb5", "content_hb6", "content_hb7", "content_hb8"):
    try:
        m = importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name == name:
            continue
        raise
    blocks += m.blocks
    glossary += getattr(m, "glossary", [])
meta = ["Project: ANVESH - Intelligent Road Event Investigation & Traffic Intelligence", "Repository: D:\ANVESH   |   Presentation: 2 November 2026",
        "Based on: source code, Session 6 and Session 7 reports, demo kit, installed Ultralytics 8.4.132 source", "Version: finalization (Session 9, 15 November 2026): officer screen, streaming lifecycle, analytical-validity and benchmark chapters added; see docs/ANVESH_FINALIZATION_REPORT.md"]
note = "RESEARCH / DEMONSTRATION ONLY. Every registry / challan result shown by the demo is SIMULATED. Items the repository does not establish are marked \"Not established in the current implementation\"."
title, sub = "Complete Technical Viva Handbook", "Understand, demonstrate and defend ANVESH"
foot = "ANVESH - Complete Technical Viva Handbook"
build_pdf(blocks, OUT / "ANVESH_COMPLETE_TECHNICAL_VIVA_HANDBOOK.pdf", title, sub, meta, foot, glossary=glossary or None, cover_note=note)
build_docx(blocks, OUT / "ANVESH_COMPLETE_TECHNICAL_VIVA_HANDBOOK.docx", title, sub, meta, foot, glossary=glossary or None, cover_note=note)
print("built")
