"""Builds docs/ANVESH_PRESENTATION_AND_TECHNICAL_VIVA_GUIDE.pdf and .md from content_viva1..4.py (one source for both).

    python docs/viva_assets/make_viva_figs.py      # diagrams (synthetic drawings, no video frames)
    python docs/viva_assets/build_viva_guide.py    # PDF + Markdown (system Python with reportlab and matplotlib)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import content_viva1, content_viva2, content_viva3, content_viva4  # noqa: E402
import docbuilder  # noqa: E402
from docbuilder import build_pdf  # noqa: E402

docbuilder.COVER_HEADER = "ANVESH  |  Vehicle Behaviour Analysis and Traffic Forensics System"

OUT = HERE.parent
blocks = content_viva1.blocks + content_viva2.blocks + content_viva3.blocks + content_viva4.blocks
TITLE, SUB = "ANVESH - Presentation and Technical Viva Guide", "Understand, present and defend the project from the basics"
META = ["Project: ANVESH - Vehicle Behaviour Analysis and Traffic Forensics System",
        "Based on the final repository (branch officer-screen, commit d0ec041), source code, finalization report and benchmark files",
        "Written for the project presentation / viva; research and demonstration system"]
NOTE = ("Every number comes from the code, the reports or the benchmark files in this repository. Items that were not measured are marked \"not measured\" / \"not recorded\". "
        "Worked examples are labelled illustrative.")


def md_text(t):
    return t


def to_markdown(blocks):
    out = [f"# {TITLE}", "", f"*{SUB}*", "", *[f"- {m}" for m in META], "", f"> {NOTE}", ""]
    for b in blocks:
        k = b[0]
        if k == "h1":
            out += ["", f"## {b[1]}", ""]
        elif k == "h2":
            out += ["", f"### {b[1]}", ""]
        elif k == "h3":
            out += ["", f"#### {b[1]}", ""]
        elif k == "p":
            out += [b[1], ""]
        elif k == "bul":
            out += [f"- {x}" for x in b[1]] + [""]
        elif k == "num":
            out += [f"{i}. {x}" for i, x in enumerate(b[1], 1)] + [""]
        elif k == "table":
            _k, header, rows, _w, cap = b
            if cap:
                out += [f"**{cap}**", ""]
            out += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            out += ["| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in r) + " |" for r in rows] + [""]
        elif k == "fig":
            name = Path(b[1]).name
            out += [f"![{b[3] or name}](viva_assets/viva_figs/{name})", "", f"*{b[3] or ''}*", ""]
        elif k == "formula":
            out += ["$$", b[1], "$$", "", f"*{b[2] or ''}*", ""]
        elif k == "callout":
            title = b[3] or b[1].title()
            out += [f"> **{title}:** {b[2]}", ""]
        elif k == "code":
            out += ["```", b[1], "```", ""]
        elif k == "qa":
            for q, a in b[1]:
                out += [f"**{q}**", "", a, ""]
        elif k in ("pb", "space"):
            continue
    return "\n".join(out)


if __name__ == "__main__":
    build_pdf(blocks, OUT / "ANVESH_PRESENTATION_AND_TECHNICAL_VIVA_GUIDE.pdf", TITLE, SUB, META, "ANVESH - Presentation and Technical Viva Guide", toc_depth=2, cover_note=NOTE)
    (OUT / "ANVESH_PRESENTATION_AND_TECHNICAL_VIVA_GUIDE.md").write_text(to_markdown(blocks), encoding="utf-8")
    print("built; questions:", content_viva4.n[0])
