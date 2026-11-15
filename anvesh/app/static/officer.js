/* ANVESH officer screen: one page, one click. Talks only to the existing API:
     GET /auth/me, GET|POST /auth/demo (loopback one-click), POST /auth/login, GET /cameras, GET /sessions,
     POST /cameras/{id}/sessions, POST /sessions/{id}/stop|pause|resume, GET /sessions/{id}/officer,
     MJPEG /sessions/{id}/stream.mjpg, GET /sessions/{id}/alerts/{n}/thumb.jpg.
   No inline script and no inline handlers (the server's CSP forbids them). */
"use strict";
(function () {
  const API = "/api/v1";
  const $ = (id) => document.getElementById(id);
  const ACTIVE = ["starting", "connecting", "live", "stalled", "reconnecting", "paused"];
  const TERMINAL = ["stopped", "completed", "failed", "interrupted"];
  const BLANK = "data:image/gif;base64,R0lGODlhAQABAAAAACw=";

  // What to offer, in this order. Anything not listed still appears under "More".
  const SOURCES = [
    { id: "demo-motorway", title: "Real motorway CCTV", kind: "real", desc: "Recorded camera footage processed live: lanes, approximate speed, stopped vehicles and the cause of the jam." },
    { id: "sim-jam", title: "Test: jam from a breakdown", kind: "sim", desc: "A scripted scene where one vehicle breaks down and a queue builds behind it. Shows the jam-cause marker." },
    { id: "sim-wrong-way", title: "Test: wrong-way driver", kind: "sim", desc: "A scripted scene where one vehicle drives against the traffic. Shows the wrong-way alert." },
    { id: "sim-accident", title: "Test: accident", kind: "sim", desc: "A scripted scene where two vehicles collide and stop. Shows the possible-accident alert." },
    { id: "demo-plates", title: "Test: number plates", kind: "photo", desc: "Real photographs of vehicles with plates, played as a video, to show the plate reader at work." },
  ];

  const ICON = {
    lanes: '<svg viewBox="0 0 24 24"><path d="M6 3 3 21M18 3l3 18M12 3v3M12 10v4M12 18v3"/></svg>',
    speed: '<svg viewBox="0 0 24 24"><path d="M4 18a8 8 0 1 1 16 0"/><path d="m12 18 4-6"/></svg>',
    wrongway: '<svg viewBox="0 0 24 24"><path d="M12 21V6M6 12l6-6 6 6"/><path d="M4 4l16 16"/></svg>',
    accident: '<svg viewBox="0 0 24 24"><path d="M12 3 2 20h20Z"/><path d="M12 10v4M12 17h.01"/></svg>',
    plate: '<svg viewBox="0 0 24 24"><rect x="2" y="7" width="20" height="10" rx="2"/><path d="M6 12h.01M10 12h8"/></svg>',
    jam: '<svg viewBox="0 0 24 24"><rect x="2" y="9" width="6" height="6" rx="1"/><rect x="9" y="9" width="6" height="6" rx="1"/><rect x="16" y="9" width="6" height="6" rx="1"/><path d="M3 19h18"/></svg>',
  };

  const S = { cams: [], cam: null, sid: null, timer: null, last: null, paused: false, ending: false, user: null, bad: 0 };

  // ------------------------------------------------------------------ http
  function csrfHeaders(extra) { return Object.assign({ "X-Requested-With": "anvesh", "Content-Type": "application/json" }, extra || {}); }
  async function api(method, path, body) {
    const r = await fetch(API + path, { method, headers: csrfHeaders(), credentials: "same-origin", body: body ? JSON.stringify(body) : undefined });
    let data = null;
    try { data = await r.json(); } catch (e) { /* empty body */ }
    if (!r.ok) {
      const err = new Error((data && data.error && data.error.message) || (data && data.message) || ("HTTP " + r.status));
      err.status = r.status;
      throw err;
    }
    return data;
  }
  function toast(msg) {
    const t = $("toast"); t.textContent = msg; t.classList.add("show");
    clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove("show"), 3500);
  }
  const el = (tag, attrs, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else if (k === "html") n.innerHTML = v; else n.setAttribute(k, v);
    }
    for (const k of kids.flat()) if (k != null) n.append(k.nodeType ? k : document.createTextNode(String(k)));
    return n;
  };
  const mmss = (t) => { t = Math.max(0, Math.floor(t || 0)); return String(Math.floor(t / 60)).padStart(2, "0") + ":" + String(t % 60).padStart(2, "0"); };
  const show = (id) => { for (const s of ["login", "chooser", "live"]) $(s).hidden = s !== id; };

  // ------------------------------------------------------------------ sign-in
  async function boot() {
    try {
      const d = await api("GET", "/auth/demo");   // public: tells whether the one-click demo sign-in exists (loopback only)
      if (d.one_click) { const r = await api("POST", "/auth/demo"); S.user = r.user; }
    } catch (e) { /* not available: use the normal sign-in */ }
    if (!S.user) {
      try { S.user = (await api("GET", "/auth/me")).user; } catch (e) { /* not signed in */ }
    }
    if (!S.user) { show("login"); $("u").focus(); return; }
    $("who").textContent = S.user.display_name || S.user.username;
    await loadCameras();
    const running = pickRunning();
    if (running) attach(running.camera, running.session_id); else showChooser();
  }
  $("loginForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    $("loginErr").textContent = "";
    try {
      const r = await api("POST", "/auth/login", { username: $("u").value.trim(), password: $("p").value });
      S.user = r.user; $("p").value = "";
      $("who").textContent = S.user.display_name || S.user.username;
      await loadCameras();
      showChooser();
    } catch (e) { $("loginErr").textContent = e.message; }
  });

  // ------------------------------------------------------------------ sources
  async function loadCameras() {
    const r = await api("GET", "/cameras");
    S.cams = r.items || [];
  }
  function camById(id) { return S.cams.find((c) => c.camera_id === id); }
  function offered() {
    const out = [];
    for (const s of SOURCES) { const c = camById(s.id); if (c && c.enabled !== false) out.push(Object.assign({}, s, { cam: c })); }
    for (const c of S.cams) if (!SOURCES.some((s) => s.id === c.camera_id) && !/^sim-(collision|queue)$/.test(c.camera_id)) out.push({ id: c.camera_id, title: c.name, kind: c.data_class === "simulated" ? "sim" : "real", desc: c.location_label || "", cam: c });
    return out;
  }
  function pickRunning() {
    const act = S.cams.filter((c) => c.active_session && ACTIVE.includes(c.active_session.state));
    if (!act.length) return null;
    const c = act.find((x) => x.camera_id === "demo-motorway") || act[0];
    return { camera: c.camera_id, session_id: c.active_session.session_id };
  }
  function showChooser() {
    stopPolling(); S.sid = null; S.cam = null;
    const box = $("cards"); box.textContent = "";
    for (const s of offered()) {
      const b = el("button", { class: "card", type: "button", "data-cam": s.id },
        el("span", { class: "tag " + s.kind, text: s.kind === "real" ? "REAL VIDEO" : (s.kind === "photo" ? "REAL PHOTOS" : "SIMULATED TEST") }),
        el("span", { class: "ct", text: s.title }), el("span", { class: "cd", text: s.desc }), el("span", { class: "go", text: "Click to start →" }));
      b.addEventListener("click", () => begin(s.id));
      box.append(b);
    }
    if (!box.children.length) box.append(el("p", { class: "sub", text: "No cameras are set up yet. Open the full dashboard to add one." }));
    show("chooser");
  }

  // Starting a source stops every other running one and then starts this one. Clicks that arrive while that is in progress are coalesced
  // (the LAST choice wins) and run one after another, so interleaved start/stop calls can never leave two sessions running.
  const startQueue = { busy: false, want: null };
  async function begin(camId) {
    startQueue.want = camId;
    if (startQueue.busy) return;
    startQueue.busy = true;
    try {
      while (startQueue.want !== null) { const id = startQueue.want; startQueue.want = null; await beginOne(id); }
    } finally { startQueue.busy = false; }
  }
  async function beginOne(camId) {
    show("live"); buildSourceBar(camId);
    setMsg("Starting… loading the detector (a few seconds the first time)");
    $("endCard").hidden = true;
    try {
      // one video at a time: stop anything else that is running so the machine is not shared
      await loadCameras();
      for (const c of S.cams) {
        if (c.camera_id !== camId && c.active_session && ACTIVE.includes(c.active_session.state)) {
          try { await api("POST", "/sessions/" + c.active_session.session_id + "/stop"); } catch (e) { /* already gone */ }
        }
      }
      const cam = camById(camId);
      let sid = cam && cam.active_session && ACTIVE.includes(cam.active_session.state) ? cam.active_session.session_id : null;
      if (!sid) { const r = await api("POST", "/cameras/" + encodeURIComponent(camId) + "/sessions"); sid = r.session_id; }
      attach(camId, sid);
    } catch (e) {
      setMsg("Could not start: " + e.message);
      toast("Could not start: " + e.message);
    }
  }

  function buildSourceBar(active) {
    const bar = $("srcbar"); bar.textContent = "";
    for (const s of offered()) {
      const b = el("button", { class: "src", type: "button", text: s.title, "aria-pressed": String(s.id === active) });
      b.addEventListener("click", () => { if (s.id !== S.cam || S.ending) begin(s.id); });
      bar.append(b);
    }
    const back = el("button", { class: "src", type: "button", text: "All choices" });
    back.addEventListener("click", async () => { await stopCurrent(); await loadCameras(); showChooser(); });
    bar.append(back);
  }

  // ------------------------------------------------------------------ live view
  const STATE_PILL = {
    live: ["LIVE", "ok"], starting: ["STARTING", "warn"], connecting: ["CONNECTING", "warn"], reconnecting: ["RECONNECTING", "warn"], stalled: ["STALLED", "bad"],
    paused: ["PAUSED", "warn"], completed: ["FINISHED", "off"], stopped: ["STOPPED", "off"], failed: ["FAILED", "bad"], interrupted: ["INTERRUPTED", "bad"],
  };
  // One line above the video: what is running (state, camera, session id) and whether the picture is recorded, simulated or a live feed.
  function setStatus(kind, label, st) {
    const box = $("statusline"); box.textContent = "";
    const pill = STATE_PILL[kind] || [label || kind.toUpperCase(), "off"];
    box.append(el("span", { class: "pill-state " + pill[1] }, el("i", { class: "dot" }), label || pill[0]));
    if (st) {
      box.append(el("b", { class: "sl-cam", text: st.camera_name || st.camera_id }));
      const src = st.simulated ? "SIMULATED scene" : (st.data_class === "demo" ? "recorded footage, processed live" : "");
      if (src) box.append(el("span", { class: "sl-src" + (st.simulated ? " sim" : ""), text: src }));
      box.append(el("span", { class: "sl-id", title: "Session identifier", text: st.session_id }));
    }
  }
  function setMsg(text) { const m = $("stageMsg"); m.hidden = !text; m.textContent = text || ""; }

  function attach(camId, sid) {
    S.cam = camId; S.sid = sid; S.ending = false; S.last = null; S.bad = 0;
    show("live"); buildSourceBar(camId);
    $("endCard").hidden = true;
    S.retry = 0;
    $("video").src = API + "/sessions/" + sid + "/stream.mjpg?ts=" + Date.now();
    setMsg("Starting… loading the detector (a few seconds the first time)");
    $("tiles").textContent = ""; $("alerts").textContent = "";
    $("alerts").append(el("li", { class: "empty", text: "Nothing to report yet. Alerts appear here the moment they are detected." }));
    $("alertCount").textContent = "";
    stopPolling();
    poll();
    S.timer = setInterval(poll, 1000);
  }
  $("video").addEventListener("error", () => {
    // fires when the stream connection drops; if the session is still running, reconnect (a few times) instead of freezing on the last frame
    if (!S.sid || S.ending || !$("video").getAttribute("src") || (S.retry || 0) >= 3) return;
    S.retry = (S.retry || 0) + 1;
    const sid = S.sid;
    setTimeout(() => { if (S.sid === sid && !S.ending) $("video").src = API + "/sessions/" + sid + "/stream.mjpg?ts=" + Date.now(); }, 1500);
  });
  function stopPolling() { if (S.timer) { clearInterval(S.timer); S.timer = null; } }

  async function poll() {
    if (!S.sid) return;
    const sid = S.sid;   // the session this request is for; a late answer for an old session is dropped, never shown
    let st;
    try { st = await api("GET", "/sessions/" + sid + "/officer"); } catch (e) {
      if (sid !== S.sid) return;
      if (e.status === 401) { stopPolling(); toast("Signed out - reload the page"); return; }
      // the session object is gone (server restarted): fall back to the stored row once
      if (++S.bad >= 3) { stopPolling(); setMsg("Lost the connection to the session."); setStatus("lost", "CONNECTION LOST"); }
      return;
    }
    if (sid !== S.sid) return;
    S.bad = 0;
    S.last = st;
    render(st);
  }

  function render(st) {
    const live = ACTIVE.includes(st.state);
    if (st.state === "starting" || st.state === "connecting" || (live && !st.fps && !st.t)) setMsg("Starting… loading the detector (a few seconds the first time)");
    else if (st.state === "reconnecting" || st.state === "stalled") setMsg("Waiting for the video…");
    else if (st.state === "paused") setMsg("Paused - press Resume to continue");
    else setMsg("");
    setStatus(st.state, null, st);
    $("clock").textContent = mmss(st.t) + (st.duration_s ? " / " + mmss(st.duration_s) : "");
    $("bar").style.width = Math.round((st.progress || 0) * 100) + "%";
    S.paused = st.state === "paused";
    $("pauseBtn").textContent = S.paused ? "Resume" : "Pause";
    $("pauseBtn").disabled = !live || !st.supports_pause;
    $("stopBtn").disabled = !live;
    renderTiles(st.tiles || []);
    renderAlerts(st);
    if (TERMINAL.includes(st.state) && !S.ending) {
      S.ending = true; stopPolling();
      const n = Object.values(st.alert_counts || {}).reduce((a, b) => a + b, 0);
      const TITLE = { failed: "The video could not be processed", stopped: "Stopped", interrupted: "Processing was interrupted" };
      $("endTitle").textContent = TITLE[st.state] || "Finished";
      $("endText").textContent = st.state === "failed" ? (st.errors && st.errors[0]) || "Something went wrong." :
        `${st.vehicles_total} vehicles were followed and ${n} alert${n === 1 ? "" : "s"} raised` + (st.state === "completed" ? "" : " before it ended") + ". The alerts stay on the side panel. The picture above is the last frame processed.";
      $("endCard").hidden = false;
    }
  }

  function renderTiles(tiles) {
    const box = $("tiles");
    if (box.children.length !== tiles.length) {
      box.textContent = "";
      for (const t of tiles) {
        box.append(el("div", { class: "tile off", "data-key": t.key },
          el("div", { class: "tt", html: (ICON[t.key] || "") }, el("span", { text: t.title })), el("div", { class: "tv" }), el("div", { class: "td" })));
      }
    }
    tiles.forEach((t, i) => {
      const n = box.children[i];
      const cls = "tile " + t.state;
      if (n.className !== cls) n.className = cls;
      const v = n.querySelector(".tv"), d = n.querySelector(".td");
      if (v.textContent !== t.value) v.textContent = t.value;
      if (d.textContent !== t.detail) d.textContent = t.detail;
    });
    // the jam-cause tile gets the whole row: it carries the main sentence
    const jam = box.querySelector('[data-key="jam"]'); if (jam) jam.classList.add("wide");
  }

  const seen = { sid: null, ids: new Set() };
  function renderAlerts(st) {
    const list = $("alerts"), items = st.alerts || [];
    $("alertCount").textContent = items.length ? items.length + " shown" : "";
    if (seen.sid !== st.session_id) { seen.sid = st.session_id; seen.ids = new Set(); list.textContent = ""; }
    if (!items.length) { if (!list.children.length) list.append(el("li", { class: "empty", text: "Nothing to report yet. Alerts appear here the moment they are detected." })); return; }
    const empty = list.querySelector(".empty"); if (empty) empty.remove();
    for (const a of items.slice().reverse()) {   // oldest first so the newest ends up on top
      if (seen.ids.has(a.id)) continue;
      seen.ids.add(a.id);
      const url = API + "/sessions/" + st.session_id + "/alerts/" + a.id + "/thumb.jpg";
      const img = a.has_thumb ? el("img", { class: "ph", src: url, alt: "Picture at the time of the alert", loading: "lazy" }) : el("div", { class: "nop" });
      const b = el("button", { class: "alert " + (a.severity === "bad" ? "bad" : (a.severity === "info" ? "info" : "")), type: "button" }, img,
        el("div", {}, el("div", { class: "at" }, a.title, a.simulated ? el("span", { class: "badge-sim", text: "SIMULATED" }) : null),
          el("div", { class: "ad", text: a.detail }), el("div", { class: "am", text: "at video time " + mmss(a.t) + (a.extra && a.extra.rating ? " · cause rated " + a.extra.rating : "") })));
      if (a.has_thumb) b.addEventListener("click", () => openLightbox(url));
      list.prepend(b);
    }
  }

  // ------------------------------------------------------------------ controls
  async function stopCurrent() {
    if (!S.sid) return;
    stopPolling();
    try { await api("POST", "/sessions/" + S.sid + "/stop"); } catch (e) { /* ended already */ }
    $("video").removeAttribute("src");
    S.sid = null;
    $("statusline").textContent = "";
  }
  $("pauseBtn").addEventListener("click", async () => {
    if (!S.sid) return;
    try { await api("POST", "/sessions/" + S.sid + (S.paused ? "/resume" : "/pause")); poll(); } catch (e) { toast(e.message); }
  });
  $("stopBtn").addEventListener("click", async () => {
    if (!S.sid) return;
    try { await api("POST", "/sessions/" + S.sid + "/stop"); } catch (e) { toast(e.message); }
    poll();
  });
  $("againBtn").addEventListener("click", () => { if (S.cam) begin(S.cam); else showChooser(); });
  $("fsBtn").addEventListener("click", () => {
    const st = $("stage");
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    else if (st.requestFullscreen) st.requestFullscreen().catch(() => toast("Full screen was refused by the browser"));
  });
  function openLightbox(url) { $("lbImg").src = url; $("lightbox").hidden = false; }
  $("lbClose").addEventListener("click", () => { $("lightbox").hidden = true; $("lbImg").removeAttribute("src"); });
  $("lightbox").addEventListener("click", (e) => { if (e.target === $("lightbox")) $("lbClose").click(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("lightbox").hidden) $("lbClose").click(); });

  boot();
})();
