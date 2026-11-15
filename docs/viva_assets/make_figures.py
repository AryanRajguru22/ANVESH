"""Figures for the ANVESH viva handbook and panel guide (helper script, not part of the application).
Run with system Python:  python docs/viva_assets/make_figures.py
Numbers that come from the real homography / run logs are computed here and written to figures/numbers.json.
"""
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
NAVY, TEAL, GREY, AMBER, RED, GREEN, PURPLE = "#1F3A5F", "#2A7F9E", "#5A6572", "#D98E04", "#B5413B", "#3C8D4B", "#6B52AE"
PALE = {"navy": "#DCE6F2", "teal": "#D7ECF2", "amber": "#FDEBC8", "red": "#F8DEDC", "green": "#DDF0E0", "grey": "#ECEFF2", "purple": "#E8E2F5"}
NUM = {}


def new(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w * 10); ax.set_ylim(0, h * 10); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc="grey", ec=NAVY, fs=7.6, bold=False, tc="#1B1F24", ls="-", lw=1.0, r=1.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.25,rounding_size={r}", fc=PALE.get(fc, fc), ec=ec, lw=lw, ls=ls)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal", linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, p1, p2, color=GREY, style="-|>", lw=1.2, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=9, color=color, lw=lw, connectionstyle=f"arc3,rad={rad}", ls=ls))


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mid(b, side):
    x, y, w, h = b
    return {"l": (x - 0.2, y + h / 2), "r": (x + w + 0.2, y + h / 2), "t": (x + w / 2, y + h + 0.2), "b": (x + w / 2, y - 0.2)}[side]


# ------------------------------------------------------------------ 1 architecture
# ------------------------------------------------------------------ 2 per-frame flow
# ------------------------------------------------------------------ 3 YOLO concept
# ------------------------------------------------------------------ 4 IoU + NMS
# ------------------------------------------------------------------ 5 tracking timeline + ByteTrack stages
# ------------------------------------------------------------------ 6 ID layers
# ------------------------------------------------------------------ calibration (real numbers)
def load_h():
    cfg = json.load(open(REPO / "demo" / "camera_config.json"))
    pts = cfg["calibration"]["reference_points"]
    I = np.array([p["image_xy"] for p in pts]); W = np.array([p["world_xy"] for p in pts])
    H, _ = cv2.findHomography(I, W, method=0)
    return cfg, I, W, H


def ap(H, p):
    v = H @ np.array([p[0], p[1], 1.0]); return v[:2] / v[2]


def fig_homography():
    cfg, I, W, H = load_h()
    Hi = np.linalg.inv(H)
    res = [np.hypot(*(ap(H, i) - w)) for i, w in zip(I, W)]
    NUM["fit_rmse_m"] = float(np.sqrt(np.mean(np.square(res)))); NUM["H"] = H.tolist()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.7, 1]}, constrained_layout=True)
    ax = axs[0]
    ax.set_xlim(0, 640); ax.set_ylim(360, 0); ax.set_aspect("equal")
    ax.add_patch(Rectangle((0, 0), 640, 360, fc="#F3F5F7", ec=GREY))
    for lane in cfg["lanes"]:
        ax.add_patch(Polygon(lane["vertices"], fc="none", ec=GREEN, lw=0.8))
    ax.add_patch(Polygon(cfg["speed_zone"], fc=PURPLE, alpha=0.14, ec=PURPLE, lw=1.2))
    ax.scatter(I[:, 0], I[:, 1], c=RED, s=14, zorder=5, label="10 reference points")
    g = (480, 300)
    ax.scatter([g[0]], [g[1]], marker="*", s=80, c=AMBER, zorder=6, label="a ground point")
    ax.text(630, 218, "speed zone", color=PURPLE, fontsize=6.3, ha="right")
    ax.set_title("Image plane (pixels), 640 x 360", fontsize=7.5, color=NAVY, fontweight="bold"); ax.legend(fontsize=6, loc="lower left"); ax.tick_params(labelsize=6)
    ax = axs[1]
    ax.set_aspect("equal"); ax.set_xlim(-1.5, 9); ax.set_ylim(-1, 24)
    ax.add_patch(Rectangle((0, 0), 3.65, 24, fc="#F3F5F7", ec=GREY))
    ax.scatter(W[:, 0], W[:, 1], c=RED, s=14, zorder=5)
    w = ap(H, g)
    ax.scatter([w[0]], [w[1]], marker="*", s=80, c=AMBER, zorder=6)
    ax.annotate(f"H maps (480, 300) px\nto ({w[0]:.2f}, {w[1]:.2f}) m", w, xytext=(2.4, 11.5), fontsize=6.2, arrowprops=dict(arrowstyle="->", color=GREY))
    ax.set_title("Road plane (metres)", fontsize=7.5, color=NAVY, fontweight="bold"); ax.tick_params(labelsize=6)
    ax.set_xlabel("x across lane (m)", fontsize=6.3); ax.set_ylabel("y along road (m)", fontsize=6.3)
    save(fig, "fig_homography.png")
    # metres per pixel vs row
    rows = np.arange(225, 361, 5)
    mpp = [np.hypot(*(ap(H, (480, y)) - ap(H, (480, y - 1)))) for y in rows]
    NUM["mpp"] = {int(y): float(m) for y, m in zip(rows, mpp) if y in (225, 240, 280, 320, 360)}
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.plot(rows, mpp, color=TEAL, lw=2)
    ax.set_xlabel("image row (pixels from top)", fontsize=7.5); ax.set_ylabel("metres of road per pixel (along the road)", fontsize=7.5)
    ax.set_title("One pixel is not one fixed distance", fontsize=8.8, color=NAVY, fontweight="bold"); ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
    for y in (225, 360):
        ax.annotate(f"{NUM['mpp'][y]:.3f} m/px", (y, NUM["mpp"][y]), fontsize=7, xytext=(y - 45 if y > 300 else y + 8, NUM["mpp"][y] + 0.03))
    save(fig, "fig_mpp.png")


def fig_speed_example():
    cfg, I, W, H = load_h(); Hi = np.linalg.inv(H)
    rng = np.random.default_rng(7)
    fps, v_true = 25.0, 80 / 3.6
    t = np.arange(0, 1.0 + 1e-9, 1 / fps)
    y_world = 0.5 + v_true * t
    xw = 1.8
    px = []
    for yw in y_world:
        v = Hi @ np.array([xw, yw, 1.0]); p = v[:2] / v[2]
        px.append(p + rng.normal(0, 0.9, 2))     # ~1 px box jitter
    px = np.array(px)
    wp = np.array([ap(H, p) for p in px])
    tbar = t.mean(); stt = ((t - tbar) ** 2).sum()
    vx = ((t - tbar) * wp[:, 0]).sum() / stt; vy = ((t - tbar) * wp[:, 1]).sum() / stt
    xbar, ybar = wp[:, 0].mean(), wp[:, 1].mean()
    res = np.hypot(wp[:, 0] - (xbar + vx * (t - tbar)), wp[:, 1] - (ybar + vy * (t - tbar)))
    rms = math.sqrt((res ** 2).mean())
    v = math.hypot(vx, vy) * 3.6
    err = 2 * (rms / math.sqrt(stt)) * 3.6 + 1.0
    # naive two-point speed using first and last frame (for contrast) and consecutive frames
    naive = np.hypot(*(wp[1] - wp[0])) / (1 / fps) * 3.6
    NUM["speed_example"] = {"true_kmh": 80.0, "n": int(len(t)), "fit_kmh": v, "err_kmh": err, "rms_m": rms, "stt": float(stt), "naive_first_two_kmh": float(naive),
                            "p_first_px": px[0].tolist(), "p_last_px": px[-1].tolist(), "w_first": wp[0].tolist(), "w_last": wp[-1].tolist(), "dt_total": float(t[-1])}
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.scatter(t, wp[:, 1], s=14, c=NAVY, label="ground point -> road y (m), 1 px jitter")
    ax.plot(t, ybar + vy * (t - tbar), c=RED, lw=1.6, label=f"least-squares slope = {v:.1f} km/h (+/- {err:.1f})")
    ax.set_xlabel("time (s)", fontsize=7.5); ax.set_ylabel("position along road (m)", fontsize=7.5); ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
    ax.legend(fontsize=6.6); ax.set_title("Worked example: a car at a true 80 km/h, 26 frames (1.0 s)", fontsize=8.6, color=NAVY, fontweight="bold")
    save(fig, "fig_speed_example.png")


# ------------------------------------------------------------------ congestion FSM
# ------------------------------------------------------------------ case workflow
# ------------------------------------------------------------------ performance chart (real CSV)
def fig_perf():
    rows = list(csv.DictReader(open(REPO / "demo" / "runs" / "perf_freeze.csv")))
    rows = [r for r in rows if r["state"] in ("live", "completed")]
    t = np.array([float(r["stream_s"]) for r in rows]); fps = np.array([float(r["proc_fps_avg"]) for r in rows])
    p50 = np.array([float(r["p50_ms"]) for r in rows]); p95 = np.array([float(r["p95_ms"]) for r in rows]); rss = np.array([float(r["server_rss_mb"]) for r in rows])
    q = np.array([float(r["frames_q_depth"]) for r in rows])
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
    axs[0].plot(t, fps, c=TEAL, lw=1.6); axs[0].axhline(25, c=GREY, ls="--", lw=0.9); axs[0].set_ylim(20, 26.5)
    axs[0].set_title("Avg processing FPS (source 25)", fontsize=6.8, color=NAVY, fontweight="bold")
    axs[1].plot(t, p50, c=TEAL, lw=1.4, label="p50"); axs[1].plot(t, p95, c=AMBER, lw=1.4, label="p95"); axs[1].axhline(40, c=RED, ls="--", lw=0.9); axs[1].legend(fontsize=6.5, loc="center right")
    axs[1].set_ylim(0, 60); axs[1].text(30, 42, "40 ms = frame interval", fontsize=5.8, color=RED)
    axs[1].set_title("Frame latency, ms (running)", fontsize=6.8, color=NAVY, fontweight="bold")
    axs[2].plot(t, rss, c=PURPLE, lw=1.4); axs[2].set_ylim(1200, 1700)
    axs[2].set_title("Server memory, MB (RSS)", fontsize=6.8, color=NAVY, fontweight="bold")
    for a in axs:
        a.set_xlabel("video time (s)", fontsize=6.3); a.tick_params(labelsize=6); a.grid(alpha=0.25)
    NUM["perf"] = {"rows": len(rows), "fps_last": float(fps[-1]), "q_max_depth_sampled": float(q.max())}
    save(fig, "fig_perf.png")


# ------------------------------------------------------------------ bounded queue
# ------------------------------------------------------------------ audit chain + evidence
# ------------------------------------------------------------------ incident lifecycle
# ------------------------------------------------------------------ collision gates
# ------------------------------------------------------------------ stopped vehicle displacement
def fig_stop():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4))
    rng = np.random.default_rng(3)
    ax = axs[0]
    pts = np.cumsum(rng.normal(0, 0.12, (75, 2)), axis=0) + [5, 5]
    mean = pts[-25:].mean(axis=0)
    ax.scatter(pts[:, 0], pts[:, 1], s=7, c=NAVY)
    ax.add_patch(plt.Circle(mean, 0.75, fc="none", ec=RED, lw=1.5)); ax.scatter(*mean, c=RED, s=18)
    ax.set_aspect("equal"); ax.set_title("Stopped: ground points stay inside a 0.75 m circle\naround the mean of the last 3 s", fontsize=8, color=NAVY, fontweight="bold"); ax.tick_params(labelsize=6)
    ax.set_xlabel("x (m)", fontsize=7); ax.set_ylabel("y (m)", fontsize=7)
    ax = axs[1]
    pts = np.column_stack([np.linspace(0, 6, 40), np.linspace(0, 2, 40)]) + rng.normal(0, 0.05, (40, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=7, c=NAVY); ax.add_patch(plt.Circle(pts[-12:].mean(axis=0), 0.75, fc="none", ec=RED, lw=1.5))
    ax.set_aspect("equal"); ax.set_title("Moving (even slowly): points leave the circle -> not stopped", fontsize=8, color=NAVY, fontweight="bold"); ax.tick_params(labelsize=6)
    save(fig, "fig_stop.png")


# ------------------------------------------------------------------ RBAC + timeline for the guide

def fig_perspective():
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.add_patch(Polygon([(1, 0.5), (9, 0.5), (6, 4.5), (4, 4.5)], fc="#C9D0D8", ec=GREY))
    for k in range(5):
        y = 0.5 + k * 0.8 ** k * 0.9
    ys = [0.5, 1.7, 2.7, 3.5, 4.1, 4.5]
    for y in ys:
        f = (y - 0.5) / 4
        ax.plot([1 + 3 * f, 9 - 3 * f], [y, y], c="white", lw=1)
    ax.text(5, 0.1, "same 10 m of road in the near field: many pixels", fontsize=6.5, ha="center", color=GREY)
    ax.text(5, 4.75, "same 10 m far away: few pixels", fontsize=6.5, ha="center", color=GREY)
    ax.set_title("Perspective: equal road distances shrink towards the horizon", fontsize=8.5, color=NAVY, fontweight="bold")
    save(fig, "fig_perspective.png")


if __name__ == "__main__":
    import make_figures2 as m2
    for f in (fig_homography, fig_speed_example, fig_perf, fig_stop, fig_perspective):
        f(); print("ok", f.__name__)
    json.dump(NUM, open(OUT / "numbers.json", "w"), indent=1)
    for f in (m2.fig_architecture, m2.fig_frame_flow, m2.fig_yolo, m2.fig_iou_nms, m2.fig_tracking, m2.fig_id_layers, m2.fig_congestion, m2.fig_case, m2.fig_queue, m2.fig_chain, m2.fig_lifecycle, m2.fig_collision, m2.fig_schedule):
        f(); print("ok", f.__name__)
