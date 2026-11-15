"""Tiny document engine: ONE content description -> a reportlab PDF and a python-docx DOCX.

Block constructors (used by content_*.py):
    H1/H2/H3(text)  P(text)  BUL([..])  NUM([..])  TABLE(header, rows, widths, caption)  FIG(path, width, caption)
    FORMULA(tex, caption)  CALLOUT(kind, text, title)  CODE(text)  QA([(q, a), ..])  PB()  SPACE(mm)
Inline markup inside text: **bold**  `code`.  Everything else is plain text (escaped automatically).
Helper script for the ANVESH viva handbook / panel guide. Not part of the application.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "figures" / "_formulas"
CACHE.mkdir(parents=True, exist_ok=True)

NAVY, TEAL, GREY, LIGHT = "#1F3A5F", "#2A7F9E", "#5A6572", "#EEF2F6"
CALLOUT_STYLE = {  # kind -> (background, border, label)
    "note": ("#EAF3FA", "#2A7F9E", "Note"),
    "warn": ("#FFF4E0", "#D98E04", "Be careful"),
    "limit": ("#FCEDED", "#B5413B", "Limitation"),
    "say": ("#EAF6EC", "#3C8D4B", "What to say"),
    "tip": ("#F1EEF9", "#6B52AE", "Tip"),
    "def": ("#F3F5F7", "#5A6572", "In simple words"),
}


# ---------------------------------------------------------------- block constructors
COVER_HEADER = "ANVESH  |  Intelligent Road Event Investigation & Traffic Intelligence"  # overridable by a build script


def H1(t): return ("h1", t)
def H2(t): return ("h2", t)
def H3(t): return ("h3", t)
def P(t): return ("p", t)
def BUL(items): return ("bul", list(items))
def NUM(items): return ("num", list(items))
def TABLE(header, rows, widths=None, caption=None): return ("table", header, rows, widths, caption)
def FIG(path, width=0.9, caption=None): return ("fig", str(path), width, caption)
def FORMULA(tex, caption=None): return ("formula", tex, caption)
def CALLOUT(kind, text, title=None): return ("callout", kind, text, title)
def CODE(t): return ("code", t)
def QA(pairs): return ("qa", list(pairs))
def PB(): return ("pb",)
def SPACE(mm=4): return ("space", mm)


_TOK = re.compile(r"(\*\*.+?\*\*|`[^`]+`|(?<![\w*])\*[^*\n]+?\*(?![\w*]))")


def tokens(text):
    """-> [(kind, str)] with kind in plain|bold|code."""
    out = []
    for part in _TOK.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            out.append(("bold", part[2:-2]))
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            out.append(("code", part[1:-1]))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2 and not part.startswith("**"):
            out.append(("ital", part[1:-1]))
        else:
            out.append(("plain", part))
    return out


def render_formula(tex: str) -> Path:
    h = hashlib.sha1(tex.encode()).hexdigest()[:12]
    p = CACHE / f"f_{h}.png"
    if not p.exists():
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.text(0, 0, f"${tex}$", fontsize=15, color=NAVY)
        fig.savefig(p, dpi=240, bbox_inches="tight", pad_inches=0.08, transparent=True)
        plt.close(fig)
    return p


# ---------------------------------------------------------------- PDF (reportlab)
def build_pdf(blocks, out_path, title, subtitle, meta_lines, footer_text, toc_depth=2, glossary=None, cover_note=None):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (CondPageBreak, BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph, Preformatted,
                                    Spacer, Table, TableStyle)
    from reportlab.platypus.tableofcontents import TableOfContents
    from xml.sax.saxutils import escape

    fd = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    for name, f in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
                    ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVM", "DejaVuSansMono.ttf"), ("DVM-B", "DejaVuSansMono-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, os.path.join(fd, f)))
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")
    pdfmetrics.registerFontFamily("DVM", normal="DVM", bold="DVM-B", italic="DVM", boldItalic="DVM-B")

    navy, teal, grey, light = colors.HexColor(NAVY), colors.HexColor(TEAL), colors.HexColor(GREY), colors.HexColor(LIGHT)
    S = {
        "p": ParagraphStyle("p", fontName="DV", fontSize=8.7, leading=12.3, spaceAfter=4, alignment=TA_LEFT, textColor=colors.HexColor("#1B1F24")),
        "h1": ParagraphStyle("h1", keepWithNext=1, fontName="DV-B", fontSize=17, leading=21, textColor=navy, spaceBefore=6, spaceAfter=7),
        "h2": ParagraphStyle("h2", keepWithNext=1, fontName="DV-B", fontSize=12, leading=15, textColor=navy, spaceBefore=8, spaceAfter=4),
        "h3": ParagraphStyle("h3", keepWithNext=1, fontName="DV-B", fontSize=10, leading=13.5, textColor=teal, spaceBefore=8, spaceAfter=3),
        "bul": ParagraphStyle("bul", fontName="DV", fontSize=8.7, leading=12, leftIndent=13, bulletIndent=3, spaceAfter=1.8),
        "cap": ParagraphStyle("cap", fontName="DV-I", fontSize=8, leading=10.5, textColor=grey, spaceBefore=2, spaceAfter=8, alignment=1),
        "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.8, leading=10.2),
        "cellh": ParagraphStyle("cellh", fontName="DV-B", fontSize=7.8, leading=10.2, textColor=colors.white),
        "code": ParagraphStyle("code", fontName="DVM", fontSize=7.4, leading=9.6),
        "q": ParagraphStyle("q", fontName="DV-B", fontSize=8.7, leading=11.8, textColor=navy, spaceBefore=3.5, spaceAfter=1, keepWithNext=1),
        "a": ParagraphStyle("a", fontName="DV", fontSize=8.7, leading=12, leftIndent=10, spaceAfter=2),
        "co": ParagraphStyle("co", fontName="DV", fontSize=8.8, leading=12.4),
        "toc1": ParagraphStyle("toc1", fontName="DV-B", fontSize=9, leading=12, leftIndent=0, spaceBefore=2.5),
        "toc2": ParagraphStyle("toc2", fontName="DV", fontSize=7.8, leading=9.6, leftIndent=14),
        "cover_t": ParagraphStyle("ct", fontName="DV-B", fontSize=26, leading=31, textColor=navy),
        "cover_s": ParagraphStyle("cs", fontName="DV", fontSize=13, leading=18, textColor=teal, spaceAfter=14),
        "cover_m": ParagraphStyle("cm", fontName="DV", fontSize=9.5, leading=14, textColor=grey),
    }

    def rl(text, plain_code=False):
        out = []
        for kind, s in tokens(text):
            e = escape(s)
            if kind == "bold":
                out.append(f"<b>{e}</b>")
            elif kind == "ital":
                out.append(f"<i>{e}</i>")
            elif kind == "code" and plain_code:
                out.append(e)
            elif kind == "code":
                out.append(f'<font name="DVM" size="8" color="#7A2E5B">{e}</font>')
            else:
                out.append(e)
        return "".join(out)

    W, Hh = A4
    LM = RM = 17 * mm
    FW = W - LM - RM
    counters = {"fig": 0, "tab": 0}

    class Doc(BaseDocTemplate):
        def afterFlowable(self, fl):
            if isinstance(fl, Paragraph) and fl.style.name in ("h1", "h2"):
                lvl = 0 if fl.style.name == "h1" else 1
                text = fl.getPlainText()
                key = f"h{self.seq.nextf('toc')}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, level=lvl, closed=True)
                self.notify("TOCEntry", (lvl, text, self.page, key))

    def deco(canv, doc):
        canv.saveState()
        canv.setFont("DV", 7.5)
        canv.setFillColor(grey)
        canv.drawString(LM, 10 * mm, footer_text)
        canv.drawRightString(W - RM, 10 * mm, f"Page {doc.page}")
        canv.setStrokeColor(colors.HexColor("#C9D2DC"))
        canv.line(LM, 13.5 * mm, W - RM, 13.5 * mm)
        canv.restoreState()

    def cover(canv, doc):
        canv.saveState()
        canv.setFillColor(navy)
        canv.rect(0, Hh - 62 * mm, W, 62 * mm, stroke=0, fill=1)
        canv.setFillColor(colors.white)
        canv.setFont("DV-B", 11)
        canv.drawString(LM, Hh - 22 * mm, COVER_HEADER)
        canv.setFont("DV", 9)
        canv.drawString(LM, Hh - 30 * mm, "RESEARCH / DEMONSTRATION USE ONLY")
        canv.restoreState()

    doc = Doc(str(out_path), pagesize=A4, leftMargin=LM, rightMargin=RM, topMargin=15 * mm, bottomMargin=18 * mm,
              title=title, author="ANVESH project", subject=subtitle)
    frame = Frame(LM, 18 * mm, FW, Hh - 15 * mm - 18 * mm, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    cover_frame = Frame(LM, 30 * mm, FW, Hh - 62 * mm - 40 * mm, id="cf", leftPadding=0, rightPadding=0)
    doc.addPageTemplates([PageTemplate(id="cover", frames=[cover_frame], onPage=cover, autoNextPageTemplate="body"),
                          PageTemplate(id="body", frames=[frame], onPage=deco)])

    story = [Spacer(1, 12 * mm), Paragraph(rl(title), S["cover_t"]), Spacer(1, 4 * mm), Paragraph(rl(subtitle), S["cover_s"])]
    for ln in meta_lines:
        story.append(Paragraph(rl(ln), S["cover_m"]))
    if cover_note:
        story += [Spacer(1, 8 * mm), cover_note_box(cover_note, S, rl, FW, colors)]
    story.append(PageBreak())
    toc = TableOfContents()
    toc.levelStyles = [S["toc1"], S["toc2"]][:toc_depth]
    toc.dotsMinLevel = 0
    story += [Paragraph("Contents", ParagraphStyle("toch", parent=S["h1"])), toc, PageBreak()]

    def table_flow(header, rows, widths, caption):
        counters["tab"] += 1
        n = len(header)
        ws = widths or [1] * n
        tot = float(sum(ws))
        cw = [FW * w / tot for w in ws]
        data = [[Paragraph(rl(h, True), S["cellh"]) for h in header]]
        for r in rows:
            data.append([Paragraph(rl(str(c)), S["cell"]) for c in r])
        t = Table(data, colWidths=cw, repeatRows=1)
        st = [("BACKGROUND", (0, 0), (-1, 0), navy), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C4CDD6")),
              ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
              ("TOPPADDING", (0, 0), (-1, -1), 2.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6)]
        for i in range(1, len(data)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), light))
        t.setStyle(TableStyle(st))
        fl = []
        if caption:
            fl.append(Paragraph(f"<b>Table {counters['tab']}.</b> {rl(caption)}", ParagraphStyle("tc", parent=S["cap"], alignment=0, spaceAfter=3, keepWithNext=1)))
        fl.append(t)
        fl.append(Spacer(1, 6))
        return fl

    def callout_flow(kind, text, title):
        bg, bd, label = CALLOUT_STYLE[kind]
        head = f"<b>{escape(title or label)}</b><br/>"
        inner = [Paragraph(head + "<br/>".join(rl(x) for x in text.split("\n")), S["co"])]
        t = Table([[inner]], colWidths=[FW])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)), ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(bd)),
                               ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                               ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        return [t, Spacer(1, 6)]

    def image_flow(path, width, caption, formula=False):
        from PIL import Image as PI
        with PI.open(path) as im:
            w, h = im.size
        tw = FW * width
        th = tw * h / w
        maxh = 205 * mm
        if th > maxh:
            tw, th = tw * maxh / th, maxh
        img = Image(str(path), width=tw, height=th)
        img.hAlign = "CENTER"
        parts = [img]
        if caption:
            counters["fig"] += 1
            parts.append(Paragraph(f"<b>Figure {counters['fig']}.</b> {rl(caption)}", S["cap"]))
        return [KeepTogether(parts)]

    for b in blocks:
        k = b[0]
        if k in ("h1", "h2", "h3", "p"):
            if k == "h1" and len(story) > 8:
                story.append(CondPageBreak(110 * mm))
            story.append(Paragraph(rl(b[1]), S[k]))
        elif k == "bul":
            for it in b[1]:
                story.append(Paragraph(rl(it), S["bul"], bulletText="•"))
            story.append(Spacer(1, 3))
        elif k == "num":
            for i, it in enumerate(b[1], 1):
                story.append(Paragraph(rl(it), S["bul"], bulletText=f"{i}."))
            story.append(Spacer(1, 3))
        elif k == "table":
            story += table_flow(*b[1:])
        elif k == "fig":
            story += image_flow(b[1], b[2], b[3])
        elif k == "formula":
            p = render_formula(b[1])
            from PIL import Image as PI
            with PI.open(p) as im:
                w, h = im.size
            sc = 0.30 * 72 / 240 * 240 / 72  # points per pixel at 240 dpi => 72/240
            img = Image(str(p), width=w * 72 / 240, height=h * 72 / 240)
            img.hAlign = "CENTER"
            story.append(KeepTogether([Spacer(1, 2), img] + ([Paragraph(rl(b[2]), S["cap"])] if b[2] else [Spacer(1, 4)])))
        elif k == "callout":
            story += callout_flow(b[1], b[2], b[3])
        elif k == "code":
            t = Table([[Preformatted(b[1], S["code"])]], colWidths=[FW])
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F5F7")), ("BOX", (0, 0), (-1, -1), 0.3, colors.HexColor("#C4CDD6")),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
            story += [t, Spacer(1, 6)]
        elif k == "qa":
            for i, (q, a) in enumerate(b[1]):
                story.append(Paragraph(rl(q), S["q"]))
                story.append(Paragraph("<br/>".join(rl(x) for x in a.split("\n")), S["a"]))
        elif k == "pb":
            story.append(PageBreak())
        elif k == "space":
            story.append(Spacer(1, b[1] * mm))
    if glossary:
        story.append(PageBreak())
        story.append(Paragraph("Glossary", S["h1"]))
        story += table_flow(["Term", "Meaning"], glossary, [1, 3.2], None)
    doc.multiBuild(story)
    return doc


def cover_note_box(text, S, rl, FW, colors):
    from reportlab.platypus import Paragraph, Table, TableStyle
    t = Table([[Paragraph(rl(text), S["co"])]], colWidths=[FW])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF4E0")), ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor("#D98E04")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


# ---------------------------------------------------------------- DOCX (python-docx)
def build_docx(blocks, out_path, title, subtitle, meta_lines, footer_text, glossary=None, cover_note=None):
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    d = Document()
    sec = d.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(1.9)
    sec.top_margin, sec.bottom_margin = Cm(1.8), Cm(2.0)
    FWcm = 21.0 - 3.8
    st = d.styles
    st["Normal"].font.name = "Calibri"
    st["Normal"].font.size = Pt(10.5)
    st["Normal"].paragraph_format.space_after = Pt(5)
    for name, size, col in (("Heading 1", 20, NAVY), ("Heading 2", 14, NAVY), ("Heading 3", 11.5, TEAL)):
        s = st[name]
        s.font.name = "Calibri"
        s.font.size = Pt(size)
        s.font.bold = True
        s.font.color.rgb = RGBColor.from_string(col[1:])
        s.element.rPr.rFonts.set(qn("w:asciiTheme"), "minorHAnsi") if False else None

    def shade(cell, hexcol):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexcol[1:] if hexcol.startswith("#") else hexcol)
        tcPr.append(shd)

    def cell_border(cell, left=None):
        tcPr = cell._tc.get_or_add_tcPr()
        b = OxmlElement("w:tcBorders")
        if left:
            e = OxmlElement("w:left")
            e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "24"); e.set(qn("w:color"), left[1:])
            b.append(e)
        tcPr.append(b)

    def add_runs(par, text, size=None, color=None, bold_all=False):
        for kind, s in tokens(text):
            r = par.add_run(s)
            if size:
                r.font.size = Pt(size)
            if kind == "bold" or bold_all:
                r.bold = True
            if kind == "ital":
                r.italic = True
            if kind == "code" and not bold_all:
                r.font.name = "Consolas"
                r.font.size = Pt((size or 10.5) - 1)
                r.font.color.rgb = RGBColor(0x7A, 0x2E, 0x5B)
            elif color:
                r.font.color.rgb = RGBColor.from_string(color[1:])

    def field(par, instr, placeholder=""):
        r = par.add_run()
        for t, txt in (("begin", None), ("instr", instr), ("separate", None), ("text", placeholder), ("end", None)):
            if t == "instr":
                el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = txt
            elif t == "text":
                el = OxmlElement("w:t"); el.text = txt
            else:
                el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), t)
            r._r.append(el)

    # footer with page number
    fp = sec.footer.paragraphs[0]
    fr = fp.add_run(footer_text + "    |    Page ")
    fr.font.size = Pt(8); fr.font.color.rgb = RGBColor(0x5A, 0x65, 0x72)
    field(fp, "PAGE", "1")

    # cover
    p = d.add_paragraph()
    r = p.add_run("ANVESH  |  Intelligent Road Event Investigation & Traffic Intelligence"); r.bold = True; r.font.size = Pt(11); r.font.color.rgb = RGBColor(0x2A, 0x7F, 0x9E)
    p = d.add_paragraph(); r = p.add_run("RESEARCH / DEMONSTRATION USE ONLY"); r.font.size = Pt(9); r.font.color.rgb = RGBColor(0xB5, 0x41, 0x3B)
    for _ in range(3):
        d.add_paragraph()
    p = d.add_paragraph(); r = p.add_run(title); r.bold = True; r.font.size = Pt(28); r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    p = d.add_paragraph(); r = p.add_run(subtitle); r.font.size = Pt(14); r.font.color.rgb = RGBColor(0x2A, 0x7F, 0x9E)
    for ln in meta_lines:
        p = d.add_paragraph(); add_runs(p, ln, size=10, color=GREY)
    if cover_note:
        t = d.add_table(rows=1, cols=1); c = t.rows[0].cells[0]; shade(c, "#FFF4E0"); cell_border(c, "#D98E04")
        add_runs(c.paragraphs[0], cover_note, size=10)
    d.add_page_break()
    d.add_heading("Contents", level=1)
    heads = [(b[0], b[1]) for b in blocks if b[0] in ("h1", "h2")]
    if glossary:
        heads.append(("h1", "Glossary"))
    p = d.add_paragraph()
    r = p.add_run()
    for t, txt in (("begin", None), ("instr", 'TOC \\o "1-2" \\h \\z \\u'), ("separate", None)):
        if t == "instr":
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = txt
        else:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), t)
        r._r.append(el)
    for i, (lv, tx) in enumerate(heads):
        rr = p.add_run(("    " if lv == "h2" else "") + tx)
        rr.font.size = Pt(10 if lv == "h1" else 9)
        rr.bold = lv == "h1"
        rr.add_break()
    r2 = p.add_run()
    el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), "end"); r2._r.append(el)
    q = d.add_paragraph()
    add_runs(q, "In Word: right-click the list above and choose **Update Field** to show page numbers. (The PDF version has page numbers built in.)", size=8.5, color=GREY)
    d.add_page_break()

    counters = {"fig": 0, "tab": 0}

    def add_table(header, rows, widths, caption):
        counters["tab"] += 1
        if caption:
            cp = d.add_paragraph(); add_runs(cp, f"**Table {counters['tab']}.** {caption}", size=9, color=GREY)
            cp.paragraph_format.keep_with_next = True
        n = len(header)
        ws = widths or [1] * n
        tot = float(sum(ws))
        t = d.add_table(rows=1, cols=n)
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for i, h in enumerate(header):
            c = t.rows[0].cells[i]; shade(c, NAVY)
            add_runs(c.paragraphs[0], h, size=8.5, color="#FFFFFF", bold_all=True)
        trPr = t.rows[0]._tr.get_or_add_trPr(); th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trPr.append(th)
        for ri, row in enumerate(rows):
            cells = t.add_row().cells
            for i, val in enumerate(row):
                add_runs(cells[i].paragraphs[0], str(val), size=8.5)
                cells[i].paragraphs[0].paragraph_format.space_after = Pt(1)
                if ri % 2 == 1:
                    shade(cells[i], LIGHT)
        for row in t.rows:
            for i, c in enumerate(row.cells):
                c.width = Cm(FWcm * ws[i] / tot)
        d.add_paragraph().paragraph_format.space_after = Pt(2)

    def add_image(path, width, caption):
        from PIL import Image as PI
        with PI.open(path) as im:
            w, h = im.size
        wcm = FWcm * width
        hcm = wcm * h / w
        if hcm > 21:
            wcm, hcm = wcm * 21 / hcm, 21
        p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(path), width=Cm(wcm))
        p.paragraph_format.keep_with_next = bool(caption)
        if caption:
            counters["fig"] += 1
            cp = d.add_paragraph(); cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_runs(cp, f"**Figure {counters['fig']}.** {caption}", size=8.5, color=GREY)

    for b in blocks:
        k = b[0]
        if k == "h1":
            if d.paragraphs and d.paragraphs[-1].text != "":
                pass
            h = d.add_heading(b[1], level=1)
            h.paragraph_format.page_break_before = True
        elif k == "h2":
            d.add_heading(b[1], level=2)
        elif k == "h3":
            d.add_heading(b[1], level=3)
        elif k == "p":
            add_runs(d.add_paragraph(), b[1])
        elif k in ("bul", "num"):
            for it in b[1]:
                add_runs(d.add_paragraph(style="List Bullet" if k == "bul" else "List Number"), it)
        elif k == "table":
            add_table(*b[1:])
        elif k == "fig":
            add_image(b[1], b[2], b[3])
        elif k == "formula":
            p = render_formula(b[1])
            pp = d.add_paragraph(); pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            from PIL import Image as PI
            with PI.open(p) as im:
                w, h = im.size
            pp.add_run().add_picture(str(p), width=Cm(min(14, w / 240 * 2.54)))
            if b[2]:
                cp = d.add_paragraph(); cp.alignment = WD_ALIGN_PARAGRAPH.CENTER; add_runs(cp, b[2], size=8.5, color=GREY)
        elif k == "callout":
            bg, bd, label = CALLOUT_STYLE[b[1]]
            t = d.add_table(rows=1, cols=1); c = t.rows[0].cells[0]; shade(c, bg); cell_border(c, bd)
            p0 = c.paragraphs[0]; add_runs(p0, f"**{b[3] or label}**", size=9.5)
            for ln in b[2].split("\n"):
                add_runs(c.add_paragraph(), ln, size=9.5)
            d.add_paragraph().paragraph_format.space_after = Pt(2)
        elif k == "code":
            t = d.add_table(rows=1, cols=1); c = t.rows[0].cells[0]; shade(c, "#F4F5F7")
            for i, ln in enumerate(b[1].split("\n")):
                pp = c.paragraphs[0] if i == 0 else c.add_paragraph()
                r = pp.add_run(ln); r.font.name = "Consolas"; r.font.size = Pt(8.5); pp.paragraph_format.space_after = Pt(0)
            d.add_paragraph().paragraph_format.space_after = Pt(2)
        elif k == "qa":
            for q, a in b[1]:
                pq = d.add_paragraph(); add_runs(pq, q, bold_all=True, color=NAVY); pq.paragraph_format.keep_with_next = True; pq.paragraph_format.space_after = Pt(1)
                for ln in a.split("\n"):
                    pa = d.add_paragraph(); add_runs(pa, ln); pa.paragraph_format.left_indent = Cm(0.5)
        elif k == "pb":
            d.add_page_break()
        elif k == "space":
            d.add_paragraph()
    if glossary:
        h = d.add_heading("Glossary", level=1); h.paragraph_format.page_break_before = True
        add_table(["Term", "Meaning"], glossary, [1, 3.2], None)
    # ask Word to refresh the TOC on open
    s = d.settings.element
    pass
    d.core_properties.title = title
    d.core_properties.author = "ANVESH project"
    d.save(str(out_path))
