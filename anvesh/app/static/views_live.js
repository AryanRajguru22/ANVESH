/* Live monitoring: a video-first workspace.
   The processed (annotated) video is the dominant element; telemetry sits beside it and details live in collapsible sections below.
   Backend contract is unchanged: GET /cameras, /sessions, /sessions/{id}, /incidents, POST /cameras/{id}/sessions, /sessions/{id}/stop|pause|resume|signal,
   and the existing MJPEG stream /api/v1/sessions/{id}/stream.mjpg. Every value shown comes from those responses. Nothing is derived beyond a
   frames-per-second figure computed from two successive real readings of frames_processed / wall_elapsed_s, which is labelled as such. */
"use strict";
(function (A) {
  const { h } = A;
  const ACTIVE = ["starting", "connecting", "live", "stalled", "reconnecting", "paused"];
  const LIVE_SOURCES = ["webcam", "rtsp", "rtmp", "http_stream"];
  const BLANK_GIF = "data:image/gif;base64,R0lGODlhAQABAAAAACw=";
  const POLL_MS = 2000;

  const SPEED_STATUS = { outside_speed_zone: "outside speed zone", insufficient_history: "measuring...", no_metric_calibration: "no calibration", insufficient_precision: "too imprecise", rejected_unstable: "unstable track", rejected_implausible: "implausible", clipped_at_frame_edge: "at frame edge" };
  const PLATE_STATUS = { unreadable_too_small: ["unreadable (too small)", "bad"], marginal: ["marginal - needs review", "warn"], candidate_size: ["size may allow a read", "info"] };

  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage may be unavailable */ } },
  };
  const shortTime = (iso) => (iso ? A.fmtUtc(iso).slice(5, 19) : "-");
  const isActive = (state) => ACTIVE.includes(state);

  function trackTable(rt) {
    const rows = rt.live_tracks || [];
    if (!rows.length) return h("div", { class: "muted small" }, "No tracked vehicles at the moment.");
    return h("div", {}, h("div", { class: "muted small" }, `${rows.length} longest-tracked of ${rt.concurrent_vehicles} vehicles in view`),
      h("div", { class: "table-wrap" }, h("table", { class: "small" }, h("thead", {}, h("tr", {}, ["ID", "Class", "Speed", "Plate (est. px)", "Plate status"].map((c) => h("th", {}, c)))),
        h("tbody", {}, rows.slice(0, 12).map((r) => {
          const ps = PLATE_STATUS[r.plate_readability] || ["-", ""];
          return h("tr", {}, h("td", {}, r.track_id), h("td", {}, r.vehicle_class + (r.confidence !== undefined ? ` (${Math.round(r.confidence * 100)}%)` : "")),
            h("td", {}, r.speed_kmh !== null && r.speed_kmh !== undefined ? `${r.speed_kmh.toFixed(0)} km/h ± ${r.speed_error_kmh}${rt.geometry && rt.geometry.approximate ? " ~approx" : ""}` : h("span", { class: "muted" }, SPEED_STATUS[r.speed_status] || r.speed_status)),
            h("td", {}, r.est_plate_px), h("td", {}, A.chip(ps[0], ps[1])));
        })))),
      h("div", { class: "muted small" }, "Plate size is a HEURISTIC (about 27% of the vehicle box width). The live plate reader (shown on the officer screen) tries a read only on real-footage vehicles at least ~110 px wide, and every read is a candidate an officer must verify. OCR on saved incident evidence is a separate step and is not connected here; plates on incidents are entered by an officer."));
  }

  // ---------------------------------------------------------------- route
  A.route("live", (main, params) => {
    if (main.parentElement) main.parentElement.classList.add("lv");
    if (params[0] === "cam" && params[1]) return workspace(main, null, params[1]);
    if (params[0]) return workspace(main, params[0], null);
    return autoSelect(main);
  });

  // `#/live` on its own opens the most useful session: an active one (the demo camera first), else the newest session, else the first camera.
  async function autoSelect(main) {
    const holder = h("div", { class: "empty" }, "Opening live monitoring...");
    main.appendChild(holder);
    let cams, ses;
    try { [cams, ses] = await Promise.all([A.get("/cameras"), A.get("/sessions?limit=50")]); }
    catch (e) { A.clear(main).appendChild(A.banner("bad", "Live monitoring unavailable: " + e.message)); return; }
    if (location.hash.replace(/\/+$/, "") !== "#/live") return; // the user already navigated elsewhere
    const active = cams.items.filter((c) => c.active_session && isActive(c.active_session.state));
    const pick = active.find((c) => c.camera_id === "demo-motorway") || active[0];
    let target;
    if (pick) target = "#/live/" + pick.active_session.session_id;
    else if (ses.items.length) target = "#/live/" + ses.items[0].session_id;
    else if (cams.items.length) target = "#/live/cam/" + encodeURIComponent(cams.items[0].camera_id);
    else { A.clear(main).appendChild(A.empty("No cameras are configured. Add one from Cameras.")); return; }
    location.replace(target);
  }

  // ---------------------------------------------------------------- workspace
  function workspace(main, sid, camIdFromRoute) {
    const S = { sid, camId: camIdFromRoute, cams: [], sessions: [], sess: null, rt: null, tab: null, stageMode: null, fit: store.get("anvesh.lv.fit") || "fit",
      prev: null, fpsNow: null, sigCam: "", sigSes: "", incidents: [], player: null, stream: null };

    // -- skeleton ---------------------------------------------------------------
    const camSel = h("select", { "aria-label": "Camera", class: "lv-sel" });
    const sesSel = h("select", { "aria-label": "Session", class: "lv-sel lv-sel-ses" });
    const stateBox = h("span", { class: "row", style: "gap:6px" });
    const ctrl = h("div", { class: "row lv-ctrl" });
    const head = h("div", { class: "lv-head" },
      h("label", { class: "lv-lbl" }, "Camera", camSel), h("label", { class: "lv-lbl" }, "Session", sesSel), stateBox, h("div", { class: "lv-spacer" }), ctrl);

    const indicator = h("span", { class: "lv-ind" });
    const stageInfo = h("span", { class: "lv-info" });
    const tabBtns = h("div", { class: "lv-seg" });
    const fitBtn = h("button", { class: "lv-btn", title: "Switch between fitting the whole frame in the window and filling the column width", onclick: () => { S.fit = S.fit === "fit" ? "wide" : "fit"; store.set("anvesh.lv.fit", S.fit); applyFit(); } });
    const fsBtn = h("button", { class: "lv-btn", title: "Fullscreen video", onclick: toggleFullscreen }, "Fullscreen");
    const bar = h("div", { class: "lv-bar" }, indicator, stageInfo, h("div", { class: "lv-spacer" }), tabBtns, fitBtn, fsBtn);
    const frame = h("div", { class: "lv-frame" });
    const stage = h("div", { class: "lv-stage" }, bar, frame);

    const side = h("aside", { class: "lv-side", "aria-label": "Live intelligence" });
    const work = h("div", { class: "lv-work" }, stage, side);

    const secEvents = section("Recent events", true), secSession = section("Session details"), secStats = section("Processing statistics"), secDet = section("Detection information"), secHist = section("Session history");
    const lower = h("div", { class: "lv-lower" }, secEvents.el, secSession.el, secStats.el, secDet.el, secHist.el);
    main.appendChild(h("div", { class: "lv-page" }, head, work, lower));

    function section(title) {
      const body = h("div", { class: "lv-secbody" }), sum = h("summary", {}, h("span", { class: "lv-sectitle" }, title), h("span", { class: "lv-secmeta muted small" }));
      return { el: h("details", { class: "lv-sec" }, sum, body), body, meta: sum.lastChild };
    }

    // -- stream handling (one MJPEG connection at a time; always closed explicitly) ----------------
    function closeStream() {
      const img = S.stream; S.stream = null;
      if (img) { img.onload = null; img.onerror = null; img.src = BLANK_GIF; img.removeAttribute("src"); }
    }
    function applyFit() {
      stage.classList.toggle("wide", S.fit === "wide");
      fitBtn.textContent = S.fit === "fit" ? "Fit: window" : "Fit: width";
    }
    function toggleFullscreen() {
      if (document.fullscreenElement) { document.exitFullscreen().catch(() => {}); return; }
      if (!stage.requestFullscreen) { A.toast("Fullscreen is not available in this browser.", "bad"); return; }
      stage.requestFullscreen().catch((e) => A.toast("Fullscreen was refused: " + e.message, "bad"));
    }
    const onFs = () => { fsBtn.textContent = document.fullscreenElement === stage ? "Exit fullscreen" : "Fullscreen"; };
    document.addEventListener("fullscreenchange", onFs);
    A.onLeave(() => {
      closeStream();
      document.removeEventListener("fullscreenchange", onFs);
      if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
      if (main.parentElement) main.parentElement.classList.remove("lv");
    });

    function sourceType() { return (S.sess && S.sess.source_snapshot && S.sess.source_snapshot.source_type) || (S.rt && S.rt.source_type) || ""; }
    const isFile = () => sourceType() === "file";

    function drawStage() {
      closeStream();
      A.clear(frame);
      S.player = null;
      const active = S.rt && isActive(S.rt.state);
      if (S.tab === "replay" && isFile()) {
        S.stageMode = "replay";
        S.player = h("video", { class: "lv-media", controls: true, preload: "metadata", src: `/api/v1/sessions/${S.sid}/video` });
        frame.style.aspectRatio = "";
        frame.appendChild(S.player);
        frame.appendChild(h("div", { class: "lv-note" }, "Plays the ORIGINAL source file. Boxes are not drawn here; open an incident for annotated evidence."));
      } else if (S.sid && active) {
        S.stageMode = "stream";
        const img = h("img", { class: "lv-media", alt: "annotated live view with detections and track IDs" });
        img.onload = () => { if (img.naturalWidth && img.naturalHeight) frame.style.aspectRatio = `${img.naturalWidth} / ${img.naturalHeight}`; };
        img.src = `/api/v1/sessions/${S.sid}/stream.mjpg`;
        S.stream = img;
        frame.appendChild(img);
      } else {
        S.stageMode = "idle";
        frame.style.aspectRatio = "";
        let msg = "Select a camera and press Start processing to see the annotated video here.";
        if (S.sid) msg = isFile() ? "This session is not running. Switch to Source replay for the recorded file, or open an incident for its saved evidence." : "This session is not running. Open an incident for its saved evidence, or start a new session.";
        frame.appendChild(h("div", { class: "lv-idle" }, h("div", { class: "lv-idle-t" }, S.sid ? "Processing stopped" : "No session yet"), h("div", { class: "muted" }, msg)));
      }
      drawTabs();
    }
    function drawTabs() {
      A.clear(tabBtns);
      if (!S.sid) return;
      [["live", "Annotated"], ["replay", "Source replay"]].forEach(([k, l]) => tabBtns.appendChild(h("button", { class: S.tab === k ? "on" : "", disabled: k === "replay" && !isFile(), onclick: () => { S.tab = k; drawStage(); } }, l)));
    }

    // -- indicators ---------------------------------------------------------------
    function indicatorFor() {
      const st = S.rt ? S.rt.state : (S.sess ? S.sess.state : null);
      if (!S.sid) return ["IDLE", ""];
      if (st === "live") return LIVE_SOURCES.includes(sourceType()) ? ["LIVE", "ok"] : ["PROCESSING", "ok"];
      if (st === "paused") return ["PAUSED", "warn"];
      if (st === "starting" || st === "connecting") return ["CONNECTING", "warn"];
      if (st === "reconnecting") return ["RECONNECTING", "warn"];
      if (st === "stalled") return ["STALLED", "bad"];
      if (st === "failed" || st === "interrupted") return ["STOPPED", "bad"];
      return ["STOPPED", ""];
    }
    function camObj() { return S.cams.find((c) => c.camera_id === S.camId) || null; }

    function drawHead() {
      // camera + session selectors are rebuilt only when their options change, so an open dropdown is never disturbed
      const camSig = JSON.stringify(S.cams.map((c) => [c.camera_id, c.name, c.data_class])) + S.camId;
      if (camSig !== S.sigCam) {
        S.sigCam = camSig; A.clear(camSel);
        S.cams.forEach((c) => camSel.appendChild(h("option", { value: c.camera_id, selected: c.camera_id === S.camId }, c.name + (c.data_class === "simulated" ? "  [SIMULATED]" : (c.data_class === "demo" ? "  [demo data]" : "")))));
        if (!S.cams.length) camSel.appendChild(h("option", {}, "no cameras"));
      }
      const rows = S.sessions.filter((r) => r.camera_id === S.camId);
      const sesSig = JSON.stringify(rows.map((r) => [r.session_id, r.runtime ? r.runtime.state : r.state])) + S.sid;
      if (sesSig !== S.sigSes) {
        S.sigSes = sesSig; A.clear(sesSel);
        if (!rows.length) sesSel.appendChild(h("option", { value: "" }, "no sessions yet"));
        rows.forEach((r) => sesSel.appendChild(h("option", { value: r.session_id, selected: r.session_id === S.sid }, `${shortTime(r.started_at)} - ${r.runtime ? r.runtime.state : r.state} - ${r.session_id.slice(-5)}`)));
        sesSel.disabled = !rows.length;
      }
      const cam = camObj();
      const st = S.rt ? S.rt.state : (S.sess ? S.sess.state : null);
      A.clear(stateBox);
      if (st) stateBox.appendChild(A.stateChip(st));
      if (S.sess) stateBox.appendChild(A.classChip(S.sess.data_class)); else if (cam) stateBox.appendChild(A.classChip(cam.data_class));
      if (S.rt && S.rt.simulated) stateBox.appendChild(A.chip("SIMULATED SOURCE", "sim"));
      // controls
      A.clear(ctrl);
      const can = A.can("session.control");
      const running = S.rt && isActive(S.rt.state);
      const camActive = cam && cam.active_session && isActive(cam.active_session.state);
      if (can && !running && !camActive) ctrl.appendChild(h("button", { class: "primary", disabled: !cam || !cam.enabled, title: cam && !cam.enabled ? "Camera is disabled" : "", onclick: startProcessing }, S.sid ? "Restart processing" : "Start processing"));
      if (can && running) {
        ctrl.appendChild(h("button", { class: "danger", onclick: () => A.act(() => A.post(`/sessions/${S.sid}/stop`), "Stopping") }, "Stop processing"));
        if (S.rt.supports_pause) ctrl.appendChild(h("button", { onclick: () => A.act(() => A.post(`/sessions/${S.sid}/${S.rt.state === "paused" ? "resume" : "pause"}`)) }, S.rt.state === "paused" ? "Resume" : "Pause"));
      }
    }
    async function startProcessing() {
      const r = await A.act(() => A.post(`/cameras/${S.camId}/sessions`), "Session starting");
      if (r && r.session_id) A.go("#/live/" + r.session_id);
    }
    camSel.addEventListener("change", () => {
      const c = S.cams.find((x) => x.camera_id === camSel.value); if (!c) return;
      if (c.active_session && isActive(c.active_session.state)) A.go("#/live/" + c.active_session.session_id);
      else if (c.last_session) A.go("#/live/" + c.last_session.session_id);
      else A.go("#/live/cam/" + encodeURIComponent(c.camera_id));
    });
    sesSel.addEventListener("change", () => { if (sesSel.value && sesSel.value !== S.sid) A.go("#/live/" + sesSel.value); });

    function drawStageBar() {
      const [txt, cls] = indicatorFor();
      indicator.className = "lv-ind " + cls; indicator.textContent = "";
      indicator.appendChild(h("i", { class: "dot" })); indicator.appendChild(document.createTextNode(txt));
      const cam = camObj(); const snap = (S.sess && S.sess.source_snapshot) || {};
      const src = sourceType();
      A.clear(stageInfo).appendChild(h("span", {}, h("b", {}, snap.name || (cam && cam.name) || S.camId || "No camera"), " ",
        S.sid ? h("span", { class: "mono muted" }, S.sid) : null, " ",
        src ? h("span", { class: "chip" }, src === "file" ? "recorded file" : (src === "synthetic" ? "synthetic" : src.replace("_", " "))) : null, " ",
        S.rt && S.rt.progress !== null && S.rt.progress !== undefined ? h("span", { class: "muted small" }, `video ${A.fmtOff(S.rt.stream_time_s)} / ${A.fmtOff(S.rt.duration_s)}`) : null));
    }

    // -- side panel ---------------------------------------------------------------
    function tile(n, label, sub) { return h("div", { class: "lv-tile" }, h("div", { class: "n" }, n), h("div", { class: "l" }, label), sub ? h("div", { class: "s muted" }, sub) : null); }
    function drawSide() {
      const rt = S.rt, s = S.sess;
      const kids = [];
      if (rt) {
        if (rt.error) kids.push(A.banner("bad", rt.error));
        if (rt.errors.length) kids.push(A.banner("bad", rt.errors[rt.errors.length - 1]));
        const cls = Object.entries(rt.concurrent_by_class);
        const geo = rt.geometry || {};
        kids.push(h("div", { class: "lv-tiles" },
          tile(S.fpsNow === null ? "-" : S.fpsNow.toFixed(1), "FPS (current)", `${rt.processing_fps.toFixed(1)} avg since start`),
          tile(rt.frames_processed, "Frames processed", `${rt.frames_dropped} dropped`),
          tile(rt.concurrent_vehicles, "Vehicles in view", "tracked now"),
          tile(rt.unique_tracks_total, "Tracks this session", "not unique vehicles")));
        kids.push(h("div", { class: "lv-blk" }, h("div", { class: "lv-h" }, "Vehicle categories in view"),
          cls.length ? h("div", { class: "row", style: "gap:4px" }, cls.map(([k, v]) => A.chip(`${v} ${k}`))) : h("div", { class: "muted small" }, "None in view")));
        if (rt.progress !== null && rt.progress !== undefined) kids.push(h("div", { class: "bar", title: `${(rt.progress * 100).toFixed(1)}% of the source video` }, h("i", { style: `width:${Math.min(100, rt.progress * 100).toFixed(1)}%` })));
        kids.push(h("div", { class: "lv-blk" }, h("div", { class: "lv-h" }, "Status"),
          h("div", { class: "row", style: "gap:4px" }, A.chip("engine " + rt.engine_health, rt.engine_health === "live" ? "ok" : "warn"),
            A.chip(geo.metric_speed ? "speed: km/h" + (geo.approximate ? " (~approx)" : "") : "speed: not shown (image-space)", geo.metric_speed ? (geo.approximate ? "warn" : "ok") : ""),
            rt.signal_mode ? A.chip(`signal ${rt.signal_state || "unknown"} (${rt.signal_mode})`, "") : null),
          h("div", { class: "muted small" }, `${rt.events_emitted} event candidates emitted this session`)));
        const trend = rt.congestion_trend.slice(-6);
        if (trend.length) kids.push(h("div", { class: "lv-blk" }, h("div", { class: "lv-h" }, "Congestion (recent windows, occupancy proxy)"),
          h("div", { class: "row", style: "gap:4px" }, trend.map((w) => A.chip(w.level || "unavailable", w.level === "congested" ? "bad" : (w.level === "building" ? "warn" : ""))))));
      } else if (s) {
        const st = s.stats || {};
        if (s.error) kids.push(A.banner("bad", s.error));
        kids.push(h("div", { class: "lv-tiles" }, tile(st.frames_processed === undefined ? "-" : st.frames_processed, "Frames processed", `${st.frames_dropped || 0} dropped`),
          tile(Number(st.processing_fps || 0).toFixed(1), "FPS (session avg)"), tile(st.unique_tracks_total === undefined ? "-" : st.unique_tracks_total, "Tracks this session", "not unique vehicles"),
          tile(st.events_emitted === undefined ? "-" : st.events_emitted, "Event candidates")));
        kids.push(h("div", { class: "muted small" }, "Session ended. Figures are its final totals; no live values are available."));
      } else {
        kids.push(h("div", { class: "muted small" }, "No session selected. Live figures appear here once processing starts."));
      }
      kids.push(h("div", { class: "lv-blk" }, h("div", { class: "lv-h" }, "Recent event candidates"),
        S.incidents.length ? S.incidents.slice(0, 5).map((i) => h("div", { class: "lv-ev" }, A.sevChip(i.severity), h("a", { href: "#/incidents/" + i.incident_id }, A.title(i.event_type)), h("span", { class: "muted small" }, A.fmtOff(i.detected_offset_s))))
          : h("div", { class: "muted small" }, S.sid ? "None detected yet." : "-"),
        S.incidents.length > 5 ? h("a", { class: "small", href: "#/live/" + S.sid, onclick: (e) => { e.preventDefault(); secEvents.el.open = true; secEvents.el.scrollIntoView({ behavior: "smooth", block: "nearest" }); } }, `All ${S.incidents.length} events below`) : null,
        h("div", { class: "muted small" }, "Candidates for human review, not findings. Confidence tiers are heuristic labels, not probabilities.")));
      A.clear(side);
      kids.forEach((k) => side.appendChild(k));
    }

    // -- lower sections -------------------------------------------------------------
    function drawLower() {
      const rt = S.rt, s = S.sess, isF = isFile();
      secEvents.meta.textContent = S.sid ? `${S.incidents.length} for this session` : "";
      A.clear(secEvents.body).appendChild(S.incidents.length ? h("div", { class: "col" }, S.incidents.map((i) => h("div", { class: "row sp card tight" },
        h("div", { class: "row" }, A.thumb(i.thumbnail_asset_id), h("div", {}, h("a", { href: "#/incidents/" + i.incident_id }, A.title(i.event_type)), h("div", { class: "muted small" }, `video ${A.fmtOff(i.detected_offset_s)} - ${A.fmtUtc(i.detected_utc)}`))),
        h("div", { class: "row" }, A.sevChip(i.severity), isF && S.tab === "replay" ? h("button", { onclick: () => { if (S.player) { S.player.currentTime = Math.max(0, i.start_offset_s - 2); S.player.play(); } } }, "Jump") : null))))
        : A.empty(S.sid ? "No events detected yet." : "No session selected."));

      const snap = (s && s.source_snapshot) || {};
      secSession.meta.textContent = s ? s.session_id : "";
      A.clear(secSession.body).appendChild(s ? A.kv({ camera: snap.name || s.camera_id, camera_id: s.camera_id, session_id: s.session_id, data_class: s.data_class, source_type: A.title(snap.source_type), location: snap.location_label,
        started: A.fmtUtc(s.started_at), ended: A.fmtUtc(s.ended_at), time_basis: rt ? A.title(rt.time_basis) : A.title(s.time_basis), requested_by: s.requested_by }) : A.empty("No session selected."));

      secStats.meta.textContent = rt ? `${rt.frames_processed} frames` : "";
      A.clear(secStats.body).appendChild(rt ? A.kv({ tracker: rt.tracker, stream_time: A.fmtOff(rt.stream_time_s), source_fps: rt.fps_nominal ? rt.fps_nominal.toFixed(1) : "-", processing_fps: `${rt.processing_fps.toFixed(1)} (average since start)`,
        frames: `${rt.frames_processed} processed, ${rt.frames_dropped} dropped`, queue_depth_now: rt.queues ? `${rt.queues.frames.depth} / ${rt.queues.frames.capacity}` : "-",
        last_frame: rt.last_frame_age_s === null ? "-" : rt.last_frame_age_s.toFixed(1) + "s ago", latency_p50_p95: rt.latency && rt.latency.frame_processing ? `${rt.latency.frame_processing.p50_ms.toFixed(0)} / ${rt.latency.frame_processing.p95_ms.toFixed(0)} ms` : "-",
        queue_high_water: rt.queues ? `frames ${rt.queues.frames.high_water}/${rt.queues.frames.capacity}, dropped ${rt.queues.frames.dropped}` : "-",
        source_health: rt.source_health ? `${rt.source_health.state}, ${rt.source_health.reconnects} reconnects` : "-", engine_health: rt.engine_health, windows_emitted: rt.windows_emitted })
        : (s ? A.kv({ frames: (s.stats || {}).frames_processed, dropped: (s.stats || {}).frames_dropped, events: (s.stats || {}).events_emitted, fps: Number((s.stats || {}).processing_fps || 0).toFixed(1) }) : A.empty("No session selected.")));

      const det = A.clear(secDet.body);
      secDet.meta.textContent = rt ? `${rt.concurrent_vehicles} in view` : "";
      if (!rt) { det.appendChild(A.empty(s ? "Detection information is available while a session is running." : "No session selected.")); }
      else {
        det.appendChild(h("div", { class: "col" },
          h("div", { class: "row small" }, h("span", { class: "muted" }, "Detectors:"), (rt.detectors || []).map((d) => A.chip(d))),
          h("div", { class: "row small" }, h("span", { class: "muted" }, "Vehicles in view:"), h("b", {}, rt.concurrent_vehicles), Object.entries(rt.concurrent_by_class).map(([k, v]) => A.chip(`${v} ${k}`))),
          h("div", { class: "muted small" }, `${rt.unique_tracks_total} distinct tracks this session (session-scoped identities, not unique vehicles)`),
          rt.geometry && rt.geometry.metric_speed
            ? A.banner(rt.geometry.synthetic ? "test" : "notice", "Speed (km/h, shown with ~ when approximate): " + rt.geometry.label + (rt.speed_zone ? ". Shown only inside the magenta speed zone (near field)." : "."))
            : A.banner("notice", "Speed: not shown. Without a validated calibration, motion is image-space (pixels); no km/h is derived. Direction arrows are image-space headings."),
          trackTable(rt),
          rt.signal_mode && rt.signal_mode !== "schedule" && A.can("session.control") ? h("div", { class: "row" }, h("span", {}, "Signal (" + rt.signal_mode + "): " + (rt.signal_state || "unknown")), ["red", "amber", "green"].map((c) => h("button", { onclick: () => A.act(() => A.post(`/sessions/${S.sid}/signal`, { state: c }), "Signal set to " + c) }, c)))
            : (rt.signal_mode ? h("div", { class: "small muted" }, `Signal: ${rt.signal_state || "unknown"} (${rt.signal_mode}; not read from video)`) : null)));
      }

      secHist.meta.textContent = `${S.sessions.length} recent`;
      A.clear(secHist.body).appendChild(S.sessions.length ? h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, ["Camera", "State", "Started", "Frames", "Data class", ""].map((c) => h("th", {}, c)))),
        h("tbody", {}, S.sessions.map((r) => h("tr", { class: "click", onclick: () => A.go("#/live/" + r.session_id) }, h("td", {}, r.camera_id), h("td", {}, A.stateChip(r.runtime ? r.runtime.state : r.state)),
          h("td", {}, A.fmtUtc(r.started_at)), h("td", {}, r.runtime ? r.runtime.frames_processed : ((r.stats || {}).frames_processed || "-")), h("td", {}, A.classChip(r.data_class)), h("td", {}, h("a", { href: "#/live/" + r.session_id }, "open"))))))) : A.empty("No sessions yet. Start a camera first."));
    }

    // -- polling --------------------------------------------------------------------
    function updateFps(rt) {
      if (!rt || !isActive(rt.state)) { S.prev = null; S.fpsNow = null; return; }
      if (S.prev && rt.wall_elapsed_s > S.prev.wall && rt.frames_processed >= S.prev.frames) S.fpsNow = (rt.frames_processed - S.prev.frames) / (rt.wall_elapsed_s - S.prev.wall);
      S.prev = { frames: rt.frames_processed, wall: rt.wall_elapsed_s };
    }
    let first = true;
    A.poll(async () => {
      let cams, ses, sess = null, inc = { items: [] };
      try {
        [cams, ses] = await Promise.all([A.get("/cameras"), A.get("/sessions?limit=50")]);
        if (S.sid) [sess, inc] = await Promise.all([A.get("/sessions/" + S.sid), A.get("/incidents" + A.qs({ session_id: S.sid, limit: 50, order: "newest" }))]);
      } catch (e) { if (first) { first = false; A.clear(main).appendChild(A.banner("bad", "Live monitoring unavailable: " + e.message)); } throw e; }
      S.cams = cams.items; S.sessions = ses.items; S.sess = sess; S.rt = sess ? sess.runtime : null; S.incidents = inc.items;
      if (sess) S.camId = sess.camera_id;
      else if (!S.camId || !S.cams.some((c) => c.camera_id === S.camId)) S.camId = (S.cams[0] || {}).camera_id || null;
      // The camera was restarted (here or from the Cameras page): follow its new session instead of staring at the finished one.
      const cam = camObj();
      const finished = !S.sid || S.rt === null || !isActive(S.rt.state);
      if (finished && cam && cam.active_session && isActive(cam.active_session.state) && cam.active_session.session_id !== S.sid) { A.go("#/live/" + cam.active_session.session_id); return; }
      updateFps(S.rt);
      if (S.tab === null) S.tab = (S.rt && isActive(S.rt.state)) ? "live" : (isFile() ? "replay" : "live");
      const wanted = S.tab === "replay" && isFile() ? "replay" : ((S.sid && S.rt && isActive(S.rt.state)) ? "stream" : "idle");
      if (first || wanted !== S.stageMode) { applyFit(); drawStage(); }
      first = false;
      drawHead(); drawStageBar(); drawSide(); drawLower();
    }, POLL_MS);
  }
})(window.ANV);
