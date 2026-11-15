"""Page-width diagrams (7.2 in canvas, 10 logical units per inch) - authored at final size so the text stays readable in the PDF."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from make_figures import (AMBER, GREEN, GREY, NAVY, PALE, PURPLE, RED, TEAL, arrow, box, mid, save)

W = 7.2


def canvas(h):
    fig, ax = plt.subplots(figsize=(W, h))
    ax.set_xlim(0, W * 10); ax.set_ylim(0, h * 10); ax.axis("off")
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.995)
    return fig, ax


def title(ax, y, t):
    ax.text(0.5, y, t, fontsize=8.6, fontweight="bold", color=NAVY, va="top")


def band(ax, y, h, label, c):
    ax.add_patch(Rectangle((0, y), 72, h, fc=PALE[c], ec="none", alpha=0.6))
    ax.text(71, y + h - 1.4, label, fontsize=6.3, fontweight="bold", color=GREY, ha="right", va="top")


def fig_architecture():
    fig, ax = canvas(9.3)
    title(ax, 92.4, "ANVESH architecture - what actually runs in the demonstration")
    F = 6.5
    band(ax, 66, 24.5, "A. PERCEPTION (reader + worker threads)", "teal")
    b1 = box(ax, 2, 75, 21, 11, "Video source\nfile | RTSP | webcam\n| synthetic", "white", fs=F)
    b2 = box(ax, 26, 75, 21, 11, "Decoder + pacing\nOpenCV, 25 fps\nrealtime_factor 1.0", "white", fs=F)
    b3 = box(ax, 50, 75, 20, 11, "Bounded frame queue\ncapacity 8\nBLOCK for files", "white", fs=F)
    b4 = box(ax, 2, 69, 21, 4.8, "YOLOv8n detector (GPU)", "white", fs=F, bold=True)
    b5 = box(ax, 26, 69, 21, 4.8, "ByteTrack (raw IDs)", "white", fs=F, bold=True)
    b6 = box(ax, 50, 69, 20, 4.8, "TrackManager (identity)", "white", fs=F)
    for a, b in ((b1, b2), (b2, b3), (b4, b5), (b5, b6)):
        arrow(ax, mid(a, "r"), mid(b, "l"))
    ax.annotate("", xy=(12.5, 74.1), xytext=(60, 74.9), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.0, connectionstyle="arc3,rad=0.0"))
    band(ax, 36, 28, "B. REASONING (per frame / per 10 s window)", "amber")
    c1 = box(ax, 2, 50, 21, 10, "Calibration + config\n10 points -> homography\nspeed zone, lanes", "white", fs=F)
    c2 = box(ax, 26, 50, 21, 10, "Streaming engine\ntrack store\n10 s windows", "white", fs=F, bold=True)
    c3 = box(ax, 50, 50, 20, 10, "Frame detectors\nspeed, braking, wrong-way,\nlane, rash, collision", "white", fs=F)
    c4 = box(ax, 2, 38.5, 68, 9.2, "Window reasoner: traffic state + congestion hysteresis | stop episodes | stopped / congestion events |\njam-cause ranking (Dempster-Shafer) | queue vs obstruction", "white", fs=F)
    arrow(ax, mid(c1, "r"), mid(c2, "l")); arrow(ax, mid(c2, "r"), mid(c3, "l")); arrow(ax, mid(c2, "b"), (36.5, 47.9))
    arrow(ax, (60, 68.8), (36.5, 60.3), color=NAVY)
    band(ax, 1, 33, "C. APPLICATION (storage, API, officer workflow)", "navy")
    d1 = box(ax, 2, 22, 21, 8.5, "IncidentService\nno duplicates", "white", fs=F, bold=True)
    d2 = box(ax, 26, 22, 21, 8.5, "Evidence capture\nframes, clip, SHA-256", "white", fs=F)
    d3 = box(ax, 50, 22, 20, 8.5, "SQLite (WAL) + audit\nhash chain", "white", fs=F, bold=True)
    d4 = box(ax, 2, 11.5, 21, 8.5, "HTTP API /api/v1\nThreadingHTTPServer", "white", fs=F, bold=True)
    d5 = box(ax, 26, 11.5, 21, 8.5, "Dashboard (vanilla JS)\npolling + MJPEG", "white", fs=F, bold=True)
    d6 = box(ax, 50, 11.5, 20, 8.5, "Case workflow\nfour-eyes rule", "white", fs=F)
    e1 = box(ax, 2, 2.2, 21, 7.4, "Plates: manual entry\n(no OCR installed)", "red", ec=RED, fs=F)
    e2 = box(ax, 26, 2.2, 21, 7.4, "Registry: SANDBOX only", "red", ec=RED, fs=F)
    e3 = box(ax, 50, 2.2, 20, 7.4, "e-Challan: SANDBOX only", "red", ec=RED, fs=F)
    for a, b in ((d1, d2), (d2, d3), (d4, d5), (d5, d6), (e1, e2), (e2, e3)):
        arrow(ax, mid(a, "r"), mid(b, "l"))
    arrow(ax, (12.5, 38.4), (12.5, 30.7), color=NAVY); arrow(ax, mid(d3, "b"), (60, 20.3)); arrow(ax, mid(d6, "b"), mid(e3, "t"))
    ax.annotate("", xy=(12.5, 20.3), xytext=(60, 21.8), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.0))
    save(fig, "fig_architecture.png")


def fig_frame_flow():
    steps = [
        ("1 Pace + decode", "OpenCV reads a frame; sleeps until its time (1.0x)", "teal", "reader"),
        ("2 Bounded queue", "capacity 8; files: BLOCK, live: DROP_OLDEST (counted)", "teal", "reader"),
        ("3 Close due windows", "windows whose end has passed are closed first", "amber", "worker"),
        ("4 YOLOv8n + ByteTrack", "model.track(conf=0.25, 5 vehicle classes)", "amber", "worker"),
        ("5 Identity + store", "raw ID -> session identity; ground point = box bottom-centre", "amber", "worker"),
        ("6 on_frame callback", "history, frame ring (JPEG), live preview, live speed", "amber", "worker"),
        ("7 Frame detectors", "over-speed, braking, wrong-way, lane, rash, collision", "green", "worker"),
        ("8 Window closes", "traffic state, congestion level, stop episodes, ranking", "green", "consumer"),
        ("9 Window detectors", "stopped vehicle, long stop, congestion episode", "green", "consumer"),
        ("10 IncidentService", "natural key -> create/update ONE incident", "navy", "either"),
        ("11 Evidence capture", "key frames, annotated copies, clip; SHA-256 each", "navy", "either"),
        ("12 Persist + audit", "SQLite rows; hash-chained audit entry", "navy", "either"),
        ("13 Dashboard", "browser polls 2.5-4 s; video = MJPEG of annotated frames", "purple", "browser"),
    ]
    h = 0.57 * len(steps) + 0.9
    fig, ax = canvas(h)
    title(ax, h * 10 - 0.5, "One frame, end to end (order from StreamingPerceptionEngine.process_frame)")
    y = h * 10 - 8
    prev = None
    for t, d, c, th in steps:
        box(ax, 1.5, y, 69, 4.6, "", c)
        ax.text(3, y + 2.3, t, fontsize=6.9, fontweight="bold", va="center", color=NAVY)
        ax.text(24, y + 2.3, d, fontsize=6.2, va="center")
        ax.text(69, y + 2.3, th, fontsize=5.8, va="center", ha="right", color=GREY, style="italic")
        if prev is not None:
            arrow(ax, (36, prev - 0.1), (36, y + 4.75), color=NAVY, lw=0.8)
        prev = y
        y -= 5.7
    save(fig, "fig_frame_flow.png")


def fig_yolo():
    fig, ax = canvas(3.5)
    title(ax, 34.8, "What the detector does to one frame (and what a detection looks like)")
    items = [("Input frame\n640 x 360\n(padded to\n640 x 384)", "grey"), ("Backbone + neck\nconvolution\nlayers\n(YOLOv8n)", "teal"), ("Detection head\nmany candidate\nboxes", "teal"),
             ("Filter + NMS\nscore >= 0.25\nIoU 0.7", "amber"), ("Output\nbox, class,\nscore", "green")]
    for i, (t, c) in enumerate(items):
        box(ax, 1.5 + i * 14.2, 20, 11.6, 11, t, c, fs=6.2)
        if i < 4:
            arrow(ax, (13.3 + i * 14.2, 25.5), (15.5 + i * 14.2, 25.5), color=NAVY)
    sx, sy = 0, 0
    ax.add_patch(Rectangle((1.5, 1), 68, 16, fc="#EEF1F4", ec=GREY))
    ax.add_patch(Polygon([(22, 1), (50, 1), (42, 17), (30, 17)], fc="#B9C1CA", ec="none"))
    for (x, y, w, h, lab, c) in ((25, 3, 12, 8, "car 0.91", TEAL), (41, 6, 9, 6, "truck 0.74", AMBER), (33, 12, 5, 3.5, "car 0.31 (dropped)", GREY)):
        ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=c, lw=1.5))
        ax.text(x, y + h + 0.6, lab, fontsize=5.8, color=c, fontweight="bold")
    ax.text(70, 14.5, "box = (x1, y1, x2, y2)\nscore = confidence", fontsize=5.8, color=GREY, ha="right")
    save(fig, "fig_yolo.png")


def fig_iou_nms():
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.5))
    ax = axs[0]; ax.set_xlim(0, 20); ax.set_ylim(15, 0); ax.axis("off")
    ax.add_patch(Rectangle((2, 2), 10, 8, fc=TEAL, alpha=0.28, ec=TEAL, lw=1.6)); ax.add_patch(Rectangle((7, 5), 10, 8, fc=AMBER, alpha=0.28, ec=AMBER, lw=1.6))
    ax.add_patch(Rectangle((7, 5), 5, 5, fc=NAVY, alpha=0.45, ec="none"))
    ax.text(3, 3.6, "A = 80", fontsize=6); ax.text(12.5, 12.2, "B = 80", fontsize=6); ax.text(7.4, 8, "25", fontsize=6.5, color="white")
    ax.set_title("IoU = overlap / union\n25 / (80 + 80 - 25) = 0.185", fontsize=6.8, color=NAVY, fontweight="bold")
    ax = axs[1]; ax.set_xlim(0, 20); ax.set_ylim(15, 0); ax.axis("off")
    for (x, y, w, h, s, c, ls) in ((3, 3, 10, 8, "0.91", GREEN, "-"), (4, 4, 10, 8, "0.78", RED, "--"), (5, 3.5, 10, 8, "0.55", RED, "--"), (14, 9, 5, 5, "0.40", GREEN, "-")):
        ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=c, lw=1.4, ls=ls)); ax.text(x + 0.2, y + 1.0, s, fontsize=6, color=c)
    ax.set_title("NMS keeps the best box and\ndrops boxes with IoU > 0.7 to it", fontsize=6.8, color=NAVY, fontweight="bold")
    ax = axs[2]; ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    rows = [("Precision", "of boxes drawn, how many\nare real vehicles"), ("Recall", "of real vehicles, how many\nwere found"), ("mAP@50", "average precision; a box is\ncorrect if IoU >= 0.50"), ("mAP@50-95", "same, averaged over IoU\n0.50 ... 0.95 (stricter)")]
    for i, (a, b) in enumerate(rows):
        ax.text(0, 8.6 - i * 2.4, a, fontsize=6.8, fontweight="bold", color=TEAL, va="center"); ax.text(0, 7.5 - i * 2.4, b, fontsize=5.8, va="center")
    ax.set_title("Detector quality words", fontsize=6.8, color=NAVY, fontweight="bold")
    fig.subplots_adjust(0.01, 0.02, 0.99, 0.84, wspace=0.05)
    save(fig, "fig_iou_nms.png")


def fig_tracking():
    fig, ax = canvas(6.4)
    title(ax, 63.5, "One vehicle through time: lost for 2 frames, re-found with the same raw ID")
    xs = [5 + i * 6.6 for i in range(10)]
    for i, x in enumerate(xs, 1):
        ax.text(x, 57.5, f"f{i}", ha="center", fontsize=6.4, color=GREY)
        miss = i in (4, 5)
        ax.add_patch(Rectangle((x - 2.4, 49.5), 4.8, 6, fc="none" if miss else PALE["teal"], ec=AMBER if miss else TEAL, lw=1.2, ls="--" if miss else "-"))
        ax.text(x, 52.5, "miss" if miss else "car", ha="center", va="center", fontsize=5.8, color=AMBER if miss else "#1B1F24")
    lab = {1: ("NEW\nID 7\ntentative", AMBER), 2: ("matched\n= confirmed", GREEN), 3: ("tracked\nID 7", GREEN), 4: ("LOST\npredicted", RED), 5: ("LOST\nkept alive", RED), 6: ("re-found\nID 7", GREEN), 7: ("tracked\nID 7", GREEN)}
    for i, (t, c) in lab.items():
        ax.text(xs[i - 1], 44, t, ha="center", va="top", fontsize=5.6, color=c, fontweight="bold")
    ax.text(xs[8], 44, "a different car:\nnew ID", ha="center", va="top", fontsize=5.6, color=NAVY)
    ax.text(36, 34.5, "ByteTrack kept the lost track for up to 30 frames (1.2 s at 25 fps). After that the same car would get a NEW raw ID, and a NEW ANVESH tracklet.", ha="center", fontsize=6, color=GREY)
    ax.text(0.5, 30, "ByteTrack association in Ultralytics 8.4.132 (cost in stage 1 = 1 - IoU x score):", fontsize=7.3, fontweight="bold", color=NAVY, va="top")
    bx = [("Detections\nscore >= 0.25", "teal"), ("Stage 1: tracks vs\nhigh-score boxes\ncost <= 0.8", "amber"), ("Stage 2: leftover\ntracks vs LOW boxes\n(0.1-0.25), IoU >= 0.5", "amber"), ("Unconfirmed vs\nleft-over boxes\ncost <= 0.7", "amber"), ("Left-over box\nscore >= 0.25\n= NEW track", "green")]
    bs = []
    for i, (t, c) in enumerate(bx):
        bs.append(box(ax, 0.8 + i * 14.3, 9, 12.3, 14, t, c, fs=5.7))
    for a, b in zip(bs, bs[1:]):
        arrow(ax, mid(a, "r"), mid(b, "l"), color=NAVY)
    ax.text(36, 5.8, "In ANVESH conf=0.25 is applied BEFORE tracking, so boxes under 0.25 never reach stage 2 (a finding of Session 7).", ha="center", fontsize=6, color=GREY)
    save(fig, "fig_tracking.png")


def fig_id_layers():
    fig, ax = canvas(3.4)
    title(ax, 33.4, "Two layers of vehicle IDs")
    a = box(ax, 2, 22, 66, 8.5, "LAYER 1 - ByteTrack raw ID (made by Ultralytics, not by YOLO)\ninteger counter: 1, 2, 3 ... (reset to 0 when a tracker is created)", "amber", fs=7.4)
    b = box(ax, 2, 11, 66, 8.5, "LAYER 2 - ANVESH session identity (TrackManager)\n(camera, session, serial): serial starts at 0, printed <run>-s000-t00051", "teal", fs=7.4)
    c = box(ax, 2, 0.8, 66, 8, "NOT established: a physical vehicle identity (no re-identification,\nno merging of tracklets, no link between cameras or sessions)", "red", ec=RED, fs=7.4)
    arrow(ax, mid(a, "b"), mid(b, "t"), color=NAVY); arrow(ax, mid(b, "b"), mid(c, "t"), color=NAVY)
    save(fig, "fig_id_layers.png")


def fig_congestion():
    fig, ax = canvas(4.7)
    title(ax, 46.6, "Congestion level: hysteresis state machine, one step per 10 s window")
    f = box(ax, 1.5, 24, 16, 8, "FREE_FLOW", "green", ec=GREEN, fs=7, bold=True)
    b = box(ax, 27, 33, 18, 8, "BUILDING", "amber", ec=AMBER, fs=7, bold=True)
    c = box(ax, 54, 24, 16, 8, "CONGESTED", "red", ec=RED, fs=7, bold=True)
    d = box(ax, 27, 14, 18, 8, "DISSIPATING", "teal", ec=TEAL, fs=7, bold=True)
    arrow(ax, (18, 31), (26, 35), rad=-0.1); ax.text(16, 36, "raw high", fontsize=5.6)
    arrow(ax, (26, 36), (18, 29.5), rad=-0.1); ax.text(18.5, 26, "raw low", fontsize=5.6)
    arrow(ax, (46, 37), (54, 31), rad=-0.1); ax.text(50, 36.5, "raw high", fontsize=5.6)
    arrow(ax, (54, 25), (46, 17), rad=0.1); ax.text(50, 18.5, "not high", fontsize=5.6)
    arrow(ax, (26, 17), (18, 24.5), rad=0.1); ax.text(15.5, 19.5, "raw low", fontsize=5.6)
    arrow(ax, (46, 15), (54, 23.5), rad=-0.1); ax.text(48, 12, "raw high", fontsize=5.6)
    box(ax, 1.5, 1, 69, 9.5, "Raw pressure of ONE window (calibrated camera):   low = mean speed >= 8 m/s (28.8 km/h) AND occupancy < 0.6\nhigh = mean speed <= 3 m/s (10.8 km/h) OR occupancy >= 0.6      medium = anything else\noccupancy = min(1, mean concurrency / 20): 0.6 means about 12 vehicles in view on average", "white", fs=6.4)
    save(fig, "fig_congestion.png")


def fig_case():
    fig, ax = canvas(6.4)
    title(ax, 63.4, "Violation case state machine (anvesh/app/cases.py) and who presses each button")
    st = [("detected", "investigator\nSubmit for evidence\nverification"), ("awaiting_evidence\n_verification", "reviewer\nVerify evidence\n(hashes re-checked)"), ("awaiting_plate\n_verification", "reviewer2 (a different\nperson)\nVerify plate"),
          ("ready_for_review", "officer\nApprove for submission\n(registry result needed)"), ("approved_for\n_submission", "officer\nSubmit to SANDBOX\nsimulator"), ("submitted ->\nchallan_reference\n_received", "TEST-CHALLAN-...\nthen closed")]
    pos = [(2, 46), (26, 46), (50, 46), (50, 24), (26, 24), (2, 24)]
    bs = []
    for (x, y), (t, a) in zip(pos, st):
        bs.append(box(ax, x, y, 20, 9, t, "teal", fs=6.0, bold=True))
        ax.text(x + 10, y - 1.2, a, ha="center", va="top", fontsize=6.1)
    arrow(ax, mid(bs[0], "r"), mid(bs[1], "l"), color=NAVY); arrow(ax, mid(bs[1], "r"), mid(bs[2], "l"), color=NAVY)
    arrow(ax, (62, 45.8), (62, 33.5), color=NAVY); arrow(ax, mid(bs[3], "l"), mid(bs[4], "r"), color=NAVY); arrow(ax, mid(bs[4], "l"), mid(bs[5], "r"), color=NAVY)
    box(ax, 2, 2.5, 69, 7.5, "Side exits: rejected, needs_further_review, submission_failed (retry). Every transition is permission-checked, written to case_transitions\n(append-only) and to the audit log. Demo: plate = DEMO PLACEHOLDER, registry = SANDBOX fixture.", "amber", ec=AMBER, fs=6.3)
    save(fig, "fig_case.png")


def fig_queue():
    fig, ax = canvas(3.9)
    title(ax, 38.6, "Bounded queue and back-pressure (anvesh/streaming/queues.py)")
    r = box(ax, 1.5, 24, 15, 9, "Reader thread\ndecode + pace", "teal", fs=6)
    for i in range(8):
        ax.add_patch(Rectangle((22 + i * 3.3, 25.5), 2.9, 6, fc=PALE["amber"] if i < 5 else "white", ec=AMBER))
    ax.text(35.5, 34.5, "frame_queue (capacity 8)", ha="center", fontsize=6)
    w = box(ax, 52, 24, 17, 9, "Worker thread\nYOLO + ByteTrack", "teal", fs=6)
    arrow(ax, mid(r, "r"), (21.5, 28.5)); arrow(ax, (48.7, 28.5), mid(w, "l"))
    box(ax, 1.5, 2, 33, 17, "FILE (recorded video): BLOCK\nqueue full -> the reader WAITS\nno frame lost, memory bounded;\nslow detector = video plays slower", "green", ec=GREEN, fs=6.4)
    box(ax, 37, 2, 33, 17, "LIVE (RTSP, webcam): DROP_OLDEST\nqueue full -> oldest frame discarded\nand counted (frames_dropped);\naffected windows marked degraded", "amber", ec=AMBER, fs=6.4)
    save(fig, "fig_queue.png")


def fig_chain():
    fig, ax = canvas(5.6)
    title(ax, 55.6, "Tamper evidence: the audit hash chain and the evidence hashes")
    ys = [42, 31, 20]
    for i, y in enumerate(ys):
        box(ax, 2, y, 66, 8, f"audit row {i + 1}:  hash = SHA-256( prev_hash + canonical(row) )     prev_hash = " + ("000...0 (genesis)" if i == 0 else f"hash of row {i}"), "red" if i == 1 else "teal", ec=RED if i == 1 else TEAL, fs=6)
        if i < 2:
            arrow(ax, (35, y - 0.1), (35, y - 2.9), color=NAVY)
    ax.text(35, 17.5, "Edit or delete row 2 -> its hash changes -> row 3 no longer matches. 'Verify hash chain' reports the first bad row.", ha="center", fontsize=5.9, color=RED)
    box(ax, 2, 1.5, 32, 12.5, "Evidence files\nwritten once, SHA-256 stored in an\nimmutable table; 'Verify' re-hashes\nthe file and compares", "navy", fs=6.4)
    box(ax, 37, 1.5, 32, 12.5, "Limit: tamper-EVIDENT, not -PROOF\nwhoever can rewrite the DB file and\nthe code could rewrite the chain\n(head hash not published elsewhere)", "amber", ec=AMBER, fs=6.4)
    save(fig, "fig_chain.png")


def fig_lifecycle():
    fig, ax = canvas(4.2)
    title(ax, 41.6, "From event to investigation")
    items = [("Detector event", "a candidate with indicators,\nseverity and uncertainty", "teal"), ("Incident (detected)", "one row per natural key;\nre-sent events update it", "amber"), ("Evidence", "key frames, annotated frames,\nclip; SHA-256 each", "amber"),
             ("Investigator review", "detected > under_investigation >\nconfirmed/dismissed/escalated > closed", "navy"), ("Case", "evidence + plate verified by\ndifferent people, then approval", "navy"), ("Sandbox hand-off", "no real e-challan;\nTEST-CHALLAN-... reference", "red")]
    pos = [(2, 24), (26, 24), (50, 24), (2, 6), (26, 6), (50, 6)]
    bs = []
    for (x, y), (t, d, c) in zip(pos, items):
        bs.append(box(ax, x, y, 20, 12, "", c, ec=RED if c == "red" else NAVY))
        ax.text(x + 10, y + 9.3, t, ha="center", fontsize=6.2, fontweight="bold", color=NAVY, va="center")
        ax.text(x + 10, y + 4.2, d, ha="center", fontsize=6.0, va="center")
    arrow(ax, mid(bs[0], "r"), mid(bs[1], "l"), color=NAVY); arrow(ax, mid(bs[1], "r"), mid(bs[2], "l"), color=NAVY)
    arrow(ax, (60, 23.8), (36, 18.4), color=NAVY, rad=0.0); arrow(ax, mid(bs[3], "r"), mid(bs[4], "l"), color=NAVY); arrow(ax, mid(bs[4], "r"), mid(bs[5], "l"), color=NAVY)
    ax.annotate("", xy=(12, 18.3), xytext=(60, 23.8), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.0))
    save(fig, "fig_lifecycle.png")


def fig_collision():
    fig, ax = canvas(6.6)
    title(ax, 65.6, "Possible-collision candidate: every gate must pass (CollisionCandidateDetector)")
    gates = [("1 Contact", "IoU >= 0.10 for >= 1.0 s", False), ("2 Moving before", "> 0.8 box-widths/s, measured 1.5 s earlier", False), ("3 Closing", "centre distance shrank >= 0.15 widths in the last second", False),
             ("4 Joint stop", "both <= 0.15 widths/s for >= 2.0 s", False), ("5 Sudden", "stationary within 2.5 s of first contact", False),
             ("6 Not a queue", ">= 3 neighbours already stopped -> suppress", True), ("7 Not a standstill", ">= 60 % of >= 4 other vehicles stopped -> suppress", True)]
    y = 55
    for i, (t, d, sup) in enumerate(gates):
        box(ax, 1.5, y, 44, 5.4, "", "red" if sup else "teal", ec=RED if sup else TEAL)
        ax.text(3, y + 2.7, t, fontsize=6.2, fontweight="bold", va="center", color=NAVY); ax.text(15.5, y + 2.7, d, fontsize=6.0, va="center")
        if i < 6:
            arrow(ax, (23, y - 0.1), (23, y - 1.9), color=NAVY, lw=0.9)
        y -= 7.4
    box(ax, 49, 36, 22, 22, "ALL gates pass\n= 'possible_collision'\nseverity high,\nconfidence LOW,\nCANDIDATE,\nexperimental", "green", ec=GREEN, fs=6)
    box(ax, 49, 4, 22, 28, "Any gate fails = NO event\n\nsim-queue (8 stopped\ncars) fails gates 2 and 6.\nPerspective overlap\nfails 3 or 5.", "amber", ec=AMBER, fs=6.4)
    save(fig, "fig_collision.png")


def fig_schedule():
    fig, axs = plt.subplots(3, 1, figsize=(W, 4.6))
    plans = [("5-minute compressed (video started first, runs in the background)", 5, [(0, 0.6, "Intro", NAVY), (0.6, 1.6, "Live: track, FPS, speed", TEAL), (1.6, 2.3, "Congestion", AMBER), (2.3, 3.1, "SIM collision+queue", PURPLE), (3.1, 4.4, "Case > sandbox", GREEN), (4.4, 5, "Audit", NAVY)]),
             ("10-minute standard", 10, [(0, 1, "Intro", NAVY), (1, 3, "Live: tracks, FPS, speed", TEAL), (3, 4.2, "Congestion", AMBER), (4.2, 5.5, "Incident+evidence", AMBER), (5.5, 6.8, "SIM x3", PURPLE), (6.8, 9, "Case, plate, registry, challan", GREEN), (9, 10, "Audit", NAVY)]),
             ("15-minute detailed", 15, [(0, 1.5, "Problem", NAVY), (1.5, 4, "Live view in depth", TEAL), (4, 6, "Congestion", AMBER), (6, 8, "Incident+evidence", AMBER), (8, 10, "SIM scenarios", PURPLE), (10, 13, "Case workflow (4 users)", GREEN), (13, 14.3, "Audit", NAVY), (14.3, 15, "End", NAVY)])]
    for ax, (t, total, segs) in zip(axs, plans):
        ax.set_xlim(0, total); ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_title(t, fontsize=7, color=NAVY, fontweight="bold", loc="left", pad=2)
        for a, b, lab, c in segs:
            ax.barh(0.5, b - a, left=a, height=0.85, color=c, alpha=0.88, edgecolor="white")
            ax.text((a + b) / 2, 0.5, lab, ha="center", va="center", fontsize=6.2, color="white")
        ax.tick_params(labelsize=6)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    axs[-1].set_xlabel("minutes", fontsize=6.5)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_schedule.png")


if __name__ == "__main__":
    for f in (fig_architecture, fig_frame_flow, fig_yolo, fig_iou_nms, fig_tracking, fig_id_layers, fig_congestion, fig_case, fig_queue, fig_chain, fig_lifecycle, fig_collision, fig_schedule):
        f(); print("ok", f.__name__)
