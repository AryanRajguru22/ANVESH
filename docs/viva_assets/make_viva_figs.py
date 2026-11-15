"""Draws the three diagrams of the presentation guide (synthetic drawings only - no video frames). Output: docs/viva_assets/viva_figs/*.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "viva_figs"
OUT.mkdir(exist_ok=True)
NAVY, TEAL, GREY, AMBER, RED, LIGHT = "#1F3A5F", "#2A7F9E", "#6B7280", "#D98E04", "#B5413B", "#EAF3FA"


def box(ax, x, y, w, h, text, fc=LIGHT, ec=TEAL, fs=7.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=NAVY, fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=NAVY, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=1.2, color=color, linestyle=ls))


# ---------------------------------------------------------------- 1 architecture
fig, ax = plt.subplots(figsize=(9.2, 4.6))
ax.set_xlim(0, 18.4); ax.set_ylim(0, 9.2); ax.axis("off")
row1 = [("Video source\nfile / synthetic /\nlive URL", 0.2), ("Frame queue\n(8 frames, blocks\nwhen full)", 3.0), ("YOLOv8n\ndetector\n(GPU)", 5.8), ("ByteTrack\ntracker\n(raw ids)", 8.6),
        ("TrackManager\nsession ids\n+ history", 11.4), ("Analytics\nlanes, speed, stops,\nwrong-way, jam cause", 14.2)]
for t, x in row1:
    box(ax, x, 6.2, 2.5, 2.2, t, fs=7)
for i in range(len(row1) - 1):
    arrow(ax, row1[i][1] + 2.5, 7.3, row1[i + 1][1], 7.3)
ax.text(0.2, 8.85, "Per-frame path (live, one session thread)", fontsize=8, color=GREY, fontweight="bold")
box(ax, 14.2, 3.4, 4.0, 2.0, "10 s windows\ntraffic state + congestion\n+ stop episodes", fs=7)
arrow(ax, 15.4, 6.2, 15.4, 5.4)
box(ax, 8.2, 3.4, 5.2, 2.0, "Incidents (candidates)\nSQLite + SHA-256 hashed evidence\n(frames, clip, thumbnail) + audit chain", fc="#FFF4E0", ec=AMBER, fs=7)
arrow(ax, 14.2, 4.4, 13.4, 4.4)
arrow(ax, 13.0, 6.2, 11.2, 5.4)
box(ax, 0.2, 3.4, 7.3, 2.0, "Evidence builder -> reliability -> Dempster-Shafer fusion\n-> ranked candidate causes H1-H6 (H7 = insufficient evidence)", fc="#F1EEF9", ec="#6B52AE", fs=7)
arrow(ax, 8.2, 4.4, 7.5, 4.4)
box(ax, 0.2, 0.4, 5.6, 2.0, "HTTP API (Python http.server)\nJSON polling 1-2 s +\nMJPEG video stream", fs=7)
box(ax, 6.6, 0.4, 5.0, 2.0, "Officer screen /officer\n(video + tiles + alerts)", fs=7, bold=True)
box(ax, 12.4, 0.4, 5.8, 2.0, "Full dashboard /advanced\nincidents, cases, evidence,\naudit, administration", fs=7)
arrow(ax, 5.8, 1.4, 6.6, 1.4); arrow(ax, 11.6, 1.4, 12.4, 1.4, ls="--")
arrow(ax, 10.8, 3.4, 4.0, 2.4); arrow(ax, 5.0, 3.4, 3.0, 2.4)
ax.text(18.3, 2.9, "browser", fontsize=7, color=GREY, ha="right")
fig.tight_layout(); fig.savefig(OUT / "fig_architecture.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- 2 tracker example
fig, axs = plt.subplots(1, 5, figsize=(9.2, 2.6))
titles = ["Frame 1", "Frame 2", "Frame 3", "Frame 4", "Frame 5"]
for i, a in enumerate(axs):
    a.set_xlim(0, 12); a.set_ylim(0, 6); a.set_xticks([]); a.set_yticks([]); a.set_title(titles[i], fontsize=8, color=NAVY)
    for s_ in a.spines.values():
        s_.set_color("#C9D3DF")
    a.add_patch(Rectangle((0, 2.2), 12, 1.6, fc="#EEF1F4", ec="none"))
def car(a, x, label, ec=TEAL, ls="-", ly=3.0):
    a.add_patch(Rectangle((x, 2.35), 2.4, 1.3, fc="none", ec=ec, lw=1.6, ls=ls)); a.text(x + 1.2, ly, label, ha="center", va="center", fontsize=6.8, color=ec)
car(axs[0], 1.0, "A\nID 1")
car(axs[1], 3.5, "A\nID 1")
car(axs[2], 6.5, "A\nID 1"); car(axs[2], 0.4, "B new\nID 2", ec=AMBER)
car(axs[3], 7.4, "", ec=RED, ls="--"); axs[3].text(8.6, 4.35, "A hidden", ha="center", fontsize=6.8, color=RED); car(axs[3], 5.6, "B\nID 2", ec=AMBER)
car(axs[4], 8.8, "A again\nID 1", ec=TEAL); car(axs[4], 3.4, "B\nID 2", ec=AMBER)
notes = ["detection (conf >= 0.25)\ncreates track 1", "Kalman predicts the box;\nIoU match -> same ID", "unmatched detection\n-> new track, new ID", "A has no detection:\ntrack kept (up to 30\nframes), ID reserved", "predicted box matches\nthe new detection ->\nID 1 again"]
for a, n in zip(axs, notes):
    a.text(6, 0.95, n, ha="center", va="center", fontsize=6.2, color=GREY)
fig.suptitle("ByteTrack example: an ID is kept by matching boxes (motion + overlap), not by recognising the car", fontsize=8, color=NAVY)
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(OUT / "fig_tracker.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- 3 homography
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.2))
a = axs[0]; a.set_xlim(0, 10); a.set_ylim(0, 7); a.axis("off"); a.set_title("Camera image (pixels u, v)", fontsize=8, color=NAVY)
a.add_patch(Polygon([(1, 0.5), (9, 0.5), (6.2, 6.3), (3.8, 6.3)], fc="#EEF1F4", ec=GREY))
for (x, y) in ((4.55, 5.0), (5.45, 5.0), (2.95, 2.0), (7.05, 2.0)):
    pass
for y, w in ((1.2, 1.7), (2.9, 1.1), (4.3, 0.75), (5.4, 0.5), (6.0, 0.35)):
    a.add_patch(Rectangle((5 - w / 2, y), w, 0.1, fc=TEAL, ec="none"))
a.annotate("same real distance,\nfewer pixels far away", (5.0, 5.0), (7.6, 5.6), fontsize=7, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
a.annotate("many pixels per metre\nnear the camera", (5.0, 1.3), (6.8, 0.15), fontsize=7, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
a = axs[1]; a.set_xlim(0, 10); a.set_ylim(0, 7); a.axis("off"); a.set_title("Road plane (metres X, Y) after homography H", fontsize=8, color=NAVY)
a.add_patch(Rectangle((3, 0.5), 4, 6, fc="#EEF1F4", ec=GREY))
for y in (1.3, 3.0, 4.7):
    a.add_patch(Rectangle((4.9, y), 0.2, 0.9, fc=TEAL, ec="none"))
a.annotate("", (8.2, 3.0), (8.2, 1.3), arrowprops=dict(arrowstyle="<->", color=NAVY)); a.text(8.35, 2.1, "9 m pitch\n(assumed:\n2 m dash\n+ 7 m gap)", fontsize=7, color=NAVY, va="center")
a.text(5, 0.05, "equal distances now look equal", ha="center", fontsize=7, color=GREY)
fig.text(0.5, 0.01, "(x, y, w)^T = H (u, v, 1)^T ;   X = x / w ,  Y = y / w", ha="center", fontsize=8, color=NAVY)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(OUT / "fig_homography.png", dpi=170); plt.close(fig)
print("figures written to", OUT)
