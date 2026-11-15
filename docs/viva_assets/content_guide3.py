from pathlib import Path
from docbuilder import *

SHOTS = Path(__file__).resolve().parent / "screens"
blocks = []
A = blocks.append

A(H1("6. The officer screen demonstration (added in Sessions 8-9)"))
A(P("The earlier 21-step demonstration uses the full dashboard (`/advanced`). The **officer screen** is a shorter, video-first alternative. It starts a real processing session with one click, so it is the better opening when time is short (5 minutes)."))
A(H2("6.1 Start it"))
A(CODE('Start ANVESH Demo.bat\n   or:  .venv\\Scripts\\python.exe scripts\\run_demo.py --reset --one-click --open\nthe browser opens http://127.0.0.1:8760/  (signed in automatically; this shortcut exists only on this computer)'))
A(H2("6.2 Click path and what to say"))
A(TABLE(["Step", "Do", "Say"], [
    ["1", "Look at the chooser: five cards (real motorway video, three SIMULATED test scenes, number-plate photographs)", "\"Every card says whether it is real video or a scripted test. Nothing here is a pre-computed result: each click starts the detector on the video.\""],
    ["2", "Click **Real motorway CCTV**. Wait a few seconds the first time (the model loads)", "\"The picture is the stream of frames the detector and tracker are processing now, with their boxes and track numbers. The line above it says LIVE, which camera, and the session id.\""],
    ["3", "Point at the six tiles: Lanes, Speed, Wrong-way, Accident, Number plate, Jam cause", "\"Speed is marked approximate: the scale comes from lane-marking dashes, not a survey. The number plate tile says 'Too small' because cars are about 28 pixels wide in this 640 by 360 video.\""],
    ["4", "Click **Pause**, then **Resume**", "\"Pause freezes the source; the screen says so. Live camera sources cannot be paused.\""],
    ["5", "Click **Test: jam from a breakdown**", "\"A scripted scene, labelled SIMULATED, where one vehicle breaks down and a queue builds. The jam-cause tile names a *candidate* with the facts it is based on: did it stop first, how many queued behind it. It cannot know why it stopped.\""],
    ["6", "Click **Test: wrong-way driver**, then **Test: accident**", "\"Real footage contains no wrong-way or collision event, so these detectors are shown on scripted scenes. I do not claim they are validated on real events.\""],
    ["7", "Click an alert thumbnail to enlarge it", "\"Each alert keeps the picture from the moment it was raised.\""],
    ["8", "Click **Stop**", "\"The screen says Stopped, not Finished; the session really ends on the server and the video stream closes.\""],
    ["9", "Click **Full dashboard** (top right) and sign in with the demo credentials file", "\"The incidents, evidence hashes, the case workflow and the audit log are in the full dashboard.\""],
], [0.5, 2.9, 3.4], "Officer-screen demonstration (about 5 minutes)"))
A(H2("6.3 Screenshots from the running application"))
for name, cap in (("officer_chooser.png", "Officer screen: choose what to watch (labels say REAL VIDEO / SIMULATED TEST)"),
                  ("officer_live.png", "Officer screen: real motorway footage being processed (LIVE state line, tiles and alerts)"),
                  ("officer_jam_sim.png", "Officer screen: scripted SIMULATED breakdown jam with the jam-cause candidate")):
    if (SHOTS / name).is_file():
        A(FIG(SHOTS / name, 0.95, cap))
A(CALLOUT("note", "These screenshots were captured from the running application with `docs/viva_assets/capture_screens.py`. They contain frames of the demonstration video, whose licence is unverified, so the images and the PDFs built from them are not committed to GitHub; regenerate them locally.", "Provenance"))
A(H2("6.4 Extra difficult questions for the officer screen"))
A(TABLE(["Question", "Answer"], [
    ["Is the video really processed live, or is it a recording of results?", "The server decodes the file at video speed, runs YOLO + ByteTrack on each frame, draws the boxes on that frame and streams the JPEG. Open the server log or the processing-fps value in the dashboard: the frame counter rises with the clock. The only things not live are the sandbox registry and the finished-session history."],
    ["Why polling instead of WebSockets?", "Standard library only, no extra dependency, and a 1 s refresh is enough for tiles and alerts. The video itself is a continuous MJPEG stream."],
    ["What happens if I close the browser tab?", "The stream handler sees the broken connection and frees its thread. The processing session keeps running until it ends or someone presses Stop."],
    ["Does it read number plates?", "There is a live plate reader (YOLOv8n plate detector + RapidOCR), but it only tries vehicles at least 110 px wide and it was exercised on photographs. On the motorway clip it never fires. Every read is a candidate for an officer to check."],
    ["Is the jam cause the real cause?", "No. It is the vehicle at the front of a queue, with whether it stopped first. It does not know whether it was a breakdown, a crash or a red light."],
    ["How fast is it?", "Real-time run of the first 20 minutes of the demo clip: see the 'Performance' chapter of the handbook (numbers are measured on one laptop GPU, one video)."],
], [2.5, 4.3], "Questions"))
