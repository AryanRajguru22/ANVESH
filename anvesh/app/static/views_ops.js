/* Overview and camera management (live monitoring lives in views_live.js). */
"use strict";
(function (A) {
  const { h } = A;

  // ---------------------------------------------------------------- Overview
  A.route("overview", (main) => {
    const body = h("div", { class: "col" });
    main.appendChild(h("div", { class: "row sp" }, h("h1", {}, "Command overview"), h("span", { class: "muted small", id: "ov-time" })));
    main.appendChild(body);
    A.poll(async () => {
      let o;
      try { o = await A.get("/overview"); } catch (e) { A.clear(body).appendChild(A.banner("bad", "Overview unavailable: " + e.message)); return; }
      const sev = o.incidents.open_by_severity || {};
      const stat = (n, l, cls) => h("div", { class: "card stat" }, h("div", { class: "n " + (cls || "") }, n), h("div", { class: "l" }, l));
      const camStates = Object.entries(o.cameras.by_state).map(([k, v]) => `${v} ${k}`).join(", ") || "none";
      A.clear(body).appendChild(h("div", { class: "col" },
        h("div", { class: "grid g4" },
          stat(`${o.sessions.length}/${o.cameras.total}`, "cameras processing (" + camStates + ")"),
          stat(o.incidents.open_total, "open incidents (" + (["critical", "high", "medium", "low", "info"].filter((k) => sev[k]).map((k) => `${sev[k]} ${k}`).join(", ") || "none") + ")"),
          stat(o.cases.awaiting_review, "cases in review pipeline (" + o.cases.ready_for_officer + " ready for officer)"),
          stat(o.alerts.pending + o.alerts.failed, `alerts: ${o.alerts.pending} pending, ${o.alerts.failed} failed`)),
        o.errors.length ? A.banner("bad", "Attention: " + o.errors.map((e) => `${e.camera_id} ${e.state}${e.errors.length ? " - " + e.errors[e.errors.length - 1] : ""}`).join(" | ")) : null,
        h("div", { class: "grid g2" },
          h("div", { class: "card" }, h("h2", {}, "Active processing"),
            o.sessions.length ? h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, ["Camera", "State", "Vehicles", "FPS", "Dropped", "Last frame", "Events"].map((c) => h("th", {}, c)))),
              h("tbody", {}, o.sessions.map((s) => h("tr", { class: "click", onclick: () => A.go("#/live/" + s.session_id) }, h("td", {}, s.camera_id, " ", s.simulated ? A.chip("simulated", "sim") : null), h("td", {}, A.stateChip(s.state)),
                h("td", {}, s.concurrent_vehicles), h("td", {}, s.processing_fps.toFixed(1)), h("td", {}, s.frames_dropped), h("td", {}, s.last_frame_age_s === null ? "-" : s.last_frame_age_s.toFixed(1) + "s ago"), h("td", {}, s.events_emitted)))))) : A.empty("No camera is processing. Start one from Cameras.")),
          h("div", { class: "card" }, h("h2", {}, "Congestion and obstruction"),
            o.congestion.length ? o.congestion.map((c) => h("div", { class: "row sp", style: "padding:4px 0" }, h("span", {}, h("a", { href: "#/incidents/" + c.incident_id }, c.camera_id), " ", A.chip(c.current_level || "-", c.current_level === "congested" ? "bad" : "warn")),
              h("span", { class: "muted small" }, `peak ${A.title(c.peak_level)}, ${Math.round(c.duration_s || 0)}s`))) : h("div", { class: "muted" }, "No open congestion episode."))),
        h("div", { class: "card" }, h("div", { class: "row sp" }, h("h2", {}, "Incident queue (highest priority first)"), h("a", { href: "#/incidents" }, "All incidents")),
          o.incidents.top.length ? h("div", { class: "table-wrap" }, h("table", {}, h("tbody", {}, o.incidents.top.map((i) => h("tr", { class: "click", onclick: () => A.go("#/incidents/" + i.incident_id) },
            h("td", {}, A.thumb(i.thumbnail_asset_id)), h("td", {}, h("b", {}, A.title(i.event_type)), h("div", { class: "muted small" }, i.incident_id)), h("td", {}, A.sevChip(i.severity)),
            h("td", {}, i.camera_id, h("div", { class: "muted small" }, A.fmtUtc(i.detected_utc))), h("td", {}, A.stateChip(i.status)), h("td", {}, A.classChip(i.data_class))))))) : A.empty("No open incidents.")),
        h("div", { class: "grid g2" },
          h("div", { class: "card" }, h("h2", {}, "Violation cases by state"),
            Object.keys(o.cases.by_state).length ? Object.entries(o.cases.by_state).map(([k, v]) => h("div", { class: "row sp" }, A.stateChip(k), h("a", { href: "#/cases?state=" + k }, v))) : h("div", { class: "muted" }, "No cases yet.")),
          h("div", { class: "card" }, h("h2", {}, "Recent activity"),
            o.recent_activity.map((a) => h("div", { class: "small", style: "padding:2px 0" }, h("span", { class: "muted" }, a.ts.slice(11, 19) + " "), h("b", {}, a.actor), " " + a.action, a.entity_id ? h("span", { class: "muted" }, " " + a.entity_id) : null))))));
      const t = document.getElementById("ov-time"); if (t) t.textContent = "updated " + new Date().toLocaleTimeString();
    }, 3000);
  });

  // ---------------------------------------------------------------- Cameras
  const SESSION_ACTIVE = ["starting", "connecting", "live", "stalled", "reconnecting", "paused"];

  A.route("cameras", (main) => {
    const list = h("div", { class: "col" });
    main.appendChild(h("div", { class: "row sp" }, h("h1", {}, "Cameras and sources"), A.can("camera.manage") ? h("button", { class: "primary", onclick: () => addCamera(load) }, "Add camera / source") : null));
    main.appendChild(A.banner("notice", "Only recorded files have been exercised on real media in this build. Live adapters (webcam, RTSP, RTMP, HTTP) are implemented and tested against simulated captures and a local fake stream; no real camera or RTSP/RTMP server has been validated."));
    main.appendChild(list);
    async function load() {
      let r;
      try { r = await A.get("/cameras"); } catch (e) { A.clear(list).appendChild(A.banner("bad", e.message)); return; }
      A.clear(list).appendChild(r.items.length ? h("div", { class: "grid g2" }, r.items.map(cameraCard)) : A.empty("No cameras configured yet."));
    }
    function cameraCard(c) {
      const s = c.active_session, active = s && SESSION_ACTIVE.includes(s.state);
      const act = (fn, msg) => async () => { await A.act(fn, msg); load(); };
      return h("div", { class: "card col" },
        h("div", { class: "row sp" }, h("h2", {}, c.name), h("div", { class: "row" }, A.classChip(c.data_class), active ? A.stateChip(s.state) : A.stateChip(c.last_session ? c.last_session.state : "idle"))),
        A.kv({ camera_id: c.camera_id, source: `${A.title(c.source_type)}  ${c.source_uri_display}`, location: c.location_label || "-", geometry: c.geometry.label, timezone: `UTC${c.utc_offset_minutes >= 0 ? "+" : ""}${c.utc_offset_minutes / 60}`,
          stop_criterion: (c.config.stop_criterion || "step_speed") + (c.config.stop_criterion === "displacement" ? " (opt-in, false-positive rate not yet evaluated)" : "") }),
        active ? h("div", { class: "small" }, h("div", {}, `${s.frames_processed} frames, ${s.processing_fps.toFixed(1)} fps, ${s.frames_dropped} dropped, ${s.concurrent_vehicles} vehicles in view`),
          s.source_health ? h("div", { class: "muted" }, `source: ${s.source_health.state}, ${s.source_health.reconnects} reconnects, ${s.source_health.source_gap_frames} gap frames${s.source_health.last_error ? ", last error: " + s.source_health.last_error : ""}`) : null,
          s.errors.length ? A.banner("bad", s.errors[s.errors.length - 1]) : null) : (c.last_session && c.last_session.error ? A.banner("bad", "Last session: " + c.last_session.error) : null),
        h("div", { class: "row" },
          A.can("session.control") && !active ? h("button", { class: "primary", disabled: !c.enabled, onclick: act(() => A.post(`/cameras/${c.camera_id}/sessions`), "Session starting") }, "Start processing") : null,
          active ? h("button", { onclick: () => A.go("#/live/" + s.session_id) }, "Open live view") : null,
          active && A.can("session.control") ? h("button", { class: "danger", onclick: act(() => A.post(`/sessions/${s.session_id}/stop`), "Stopping") }, "Stop") : null,
          active && s.supports_pause && A.can("session.control") ? h("button", { onclick: act(() => A.post(`/sessions/${s.session_id}/${s.state === "paused" ? "resume" : "pause"}`)) }, s.state === "paused" ? "Resume" : "Pause") : null,
          h("button", { onclick: () => capabilities(c) }, "Event capabilities"),
          A.can("camera.manage") ? h("button", { onclick: () => editCamera(c, load) }, "Edit") : null));
    }
    load(); A.poll(load, 4000);
  });

  async function capabilities(c) {
    const r = await A.act(() => A.get("/capabilities?camera_id=" + encodeURIComponent(c.camera_id)));
    if (!r) return;
    A.modal("Event capabilities - " + c.name, h("div", { class: "col" },
      A.banner("notice", "A category listed here is operational only if its detector exists, the camera supplies the inputs it needs, and the status below says so. Nothing is claimed beyond that."),
      h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, ["Event", "Status on this camera", "Missing / requires", "Real-footage validation"].map((x) => h("th", {}, x)))),
        h("tbody", {}, r.events.map((e) => h("tr", {}, h("td", {}, h("b", {}, e.label), h("div", { class: "muted small" }, e.event_type)),
          h("td", {}, A.chip(e.available ? "can run" : "cannot run", e.available ? "ok" : "bad"), " ", A.chip(e.status, e.status === "operational" ? "ok" : (e.status === "experimental" ? "warn" : ""))),
          h("td", { class: "small" }, e.missing.length ? e.missing.join("; ") : e.requires.join("; ")),
          h("td", { class: "small" }, e.validated_on_real_footage ? A.chip("demonstrated", "ok") : A.chip("not validated", "warn"), h("div", { class: "muted" }, e.real_footage_note)))))))));
  }

  function cameraForm(existing, onSaved) {
    const c = existing || {};
    const f = {};
    const field = (label, el, cls) => h("div", { class: "field " + (cls || "") }, h("label", {}, label), el);
    f.id = h("input", { value: c.camera_id || "", disabled: !!existing, placeholder: "auto-generated if empty" });
    f.name = h("input", { value: c.name || "" });
    f.type = h("select", { disabled: !!existing }, ["file", "webcam", "rtsp", "rtmp", "http_stream", "synthetic"].map((t) => h("option", { value: t, selected: (c.source_type || "file") === t }, A.title(t))));
    f.uri = h("input", { placeholder: "path / device index / stream URL", value: existing ? "" : "" });
    f.fps = h("input", { type: "number", min: "1", max: "240", value: c.nominal_fps || "", placeholder: "required for live sources" });
    f.cls = h("select", {}, ["demo", "operational", "simulated"].map((t) => h("option", { value: t, selected: (c.data_class || "demo") === t }, t)));
    f.loc = h("input", { value: c.location_label || "", placeholder: "e.g. NH-48 / Sector 12 junction" });
    f.lat = h("input", { type: "number", step: "any", value: c.latitude === null || c.latitude === undefined ? "" : c.latitude });
    f.lon = h("input", { type: "number", step: "any", value: c.longitude === null || c.longitude === undefined ? "" : c.longitude });
    f.off = h("input", { type: "number", value: c.utc_offset_minutes === undefined ? 330 : c.utc_offset_minutes });
    f.start = h("input", { placeholder: "2026-10-01T10:00:00.000Z (recorded files; blank = processing clock)", value: (c.config && c.config.recording_start_utc) || "" });
    f.crit = h("select", {}, [["step_speed", "step_speed (default)"], ["displacement", "displacement (opt-in, experimental)"]].map(([v, l]) => h("option", { value: v, selected: ((c.config || {}).stop_criterion || "step_speed") === v }, l)));
    f.cfg = h("textarea", { placeholder: '{"direction_zones": [...], "lanes": [...], "signal": {...}, "speed_limit_kmh": 50, "calibration": {...}, "thresholds": {...}}' }, c.config ? JSON.stringify(stripKnown(c.config), null, 1) : "{}");
    const upload = h("input", { type: "file", accept: "video/*" });
    const upStatus = h("span", { class: "small muted" });
    upload.addEventListener("change", async () => {
      const file = upload.files[0]; if (!file) return;
      upStatus.textContent = `uploading ${(file.size / 1e6).toFixed(1)} MB...`;
      try { const r = await A.api("POST", "/uploads?filename=" + encodeURIComponent(file.name), file, { raw: true }); f.uri.value = r.path; f.type.value = "file"; upStatus.textContent = `uploaded: ${r.frames} frames @ ${r.fps.toFixed(1)} fps (${r.duration_s}s)`; }
      catch (e) { upStatus.textContent = ""; A.fail(e); }
    });
    const hint = h("div", { class: "muted small span2" }, "Live sources must declare nominal_fps so a camera that is down at start can still reconnect. Synthetic sources are always 'simulated'. Stream credentials are never shown again after saving.");
    const form = h("div", { class: "form" }, field("Name", f.name), field("Camera ID", f.id), field("Source type", f.type), field("Source (path, device index or URL)", f.uri),
      existing ? null : field("...or upload a recorded video", h("div", { class: "row" }, upload, upStatus)), field("Nominal FPS (live)", f.fps), field("Data class", f.cls),
      field("Location label", f.loc), field("Latitude", f.lat), field("Longitude", f.lon), field("UTC offset (minutes)", f.off), field("Recording start (UTC)", f.start), field("Stop criterion", f.crit),
      hint, field("Detection configuration (JSON)", f.cfg, "span2"));
    const save = async () => {
      let cfg;
      try { cfg = JSON.parse(f.cfg.value || "{}"); } catch (e) { A.toast("Detection configuration is not valid JSON.", "bad"); return false; }
      if (f.start.value.trim()) cfg.recording_start_utc = f.start.value.trim(); else delete cfg.recording_start_utc;
      cfg.stop_criterion = f.crit.value;
      const num = (el) => el.value === "" ? null : Number(el.value);
      const body = { name: f.name.value, location_label: f.loc.value || null, latitude: num(f.lat), longitude: num(f.lon), utc_offset_minutes: Number(f.off.value), config: cfg, data_class: f.cls.value };
      if (f.fps.value) body.nominal_fps = Number(f.fps.value);
      if (!existing) { Object.assign(body, { camera_id: f.id.value.trim() || undefined, source_type: f.type.value, source_uri: f.uri.value.trim() }); }
      else if (f.uri.value.trim()) body.source_uri = f.uri.value.trim();
      const r = await A.act(() => existing ? A.patch("/cameras/" + existing.camera_id, body) : A.post("/cameras", body), existing ? "Camera updated" : "Camera added");
      if (r) { onSaved(); return true; } return false;
    };
    return { el: form, save };
  }
  function stripKnown(cfg) { const c = Object.assign({}, cfg); delete c.recording_start_utc; delete c.stop_criterion; return c; }
  function addCamera(onSaved) { const fm = cameraForm(null, onSaved); const close = A.modal("Add camera / source", h("div", { class: "col" }, fm.el, h("div", { class: "row" }, h("button", { class: "primary", onclick: async () => { if (await fm.save()) close(); } }, "Save")))); }
  function editCamera(c, onSaved) { const fm = cameraForm(c, onSaved); const close = A.modal("Edit " + c.name, h("div", { class: "col" }, fm.el, h("div", { class: "row" }, h("button", { class: "primary", onclick: async () => { if (await fm.save()) close(); } }, "Save")))); }
})(window.ANV);
