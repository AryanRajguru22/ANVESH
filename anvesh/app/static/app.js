/* ANVESH dashboard core: DOM builder, API client, router, shared components.
   Every string from the server is rendered with textContent (via h()), never innerHTML, so plate text,
   notes and camera names cannot inject markup. */
"use strict";
window.ANV = (function () {
  const A = { user: null, views: {}, cleanups: [], offset: 330 };

  // ---------------------------------------------------------------- DOM builder
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") el.className = v;
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (k === "value") el.value = v;
      else if (k === "checked" || k === "disabled" || k === "selected") el[k] = !!v;
      else el.setAttribute(k, v === true ? "" : v);
    }
    (function add(list) {
      for (const c of list) {
        if (c === null || c === undefined || c === false) continue;
        if (Array.isArray(c)) add(c);
        else el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
      }
    })(kids);
    return el;
  }
  A.h = h;
  const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); return el; };
  A.clear = clear;

  // ---------------------------------------------------------------- API
  class ApiErr extends Error { constructor(status, body) { super((body && body.error && body.error.message) || ("HTTP " + status)); this.status = status; this.body = body; } }
  A.ApiErr = ApiErr;
  async function api(method, path, body, opts) {
    const init = { method, credentials: "same-origin", headers: { "X-Requested-With": "anvesh" } };
    if (body !== undefined && !(opts && opts.raw)) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(body); }
    if (opts && opts.raw) { init.body = body; init.headers["Content-Type"] = "application/octet-stream"; }
    const r = await fetch("/api/v1" + path, init);
    const ct = r.headers.get("Content-Type") || "";
    const data = ct.includes("json") ? await r.json() : await r.text();
    if (!r.ok) {
      if (r.status === 401 && A.user) { A.user = null; A.showLogin("Your session ended. Please sign in again."); }
      throw new ApiErr(r.status, data);
    }
    return data;
  }
  A.api = api;
  A.get = (p) => api("GET", p);
  A.post = (p, b) => api("POST", p, b === undefined ? {} : b);
  A.patch = (p, b) => api("PATCH", p, b);
  A.put = (p, b) => api("PUT", p, b);
  const qs = (o) => { const p = new URLSearchParams(); for (const [k, v] of Object.entries(o || {})) if (v !== "" && v !== null && v !== undefined) p.set(k, v); const s = p.toString(); return s ? "?" + s : ""; };
  A.qs = qs;

  // ---------------------------------------------------------------- feedback
  A.toast = function (msg, kind) {
    const t = h("div", { class: kind || "" }, msg);
    document.getElementById("toast").appendChild(t);
    setTimeout(() => t.remove(), kind === "bad" ? 9000 : 4500);
  };
  A.fail = (e) => {
    let msg = e.message || String(e);
    if (e.body && e.body.error && e.body.error.details) { const d = e.body.error.details; msg += ": " + (Array.isArray(d) ? d.join("; ") : JSON.stringify(d)); }
    A.toast(msg, "bad");
  };
  A.act = async function (fn, okMsg) { try { const r = await fn(); if (okMsg) A.toast(okMsg, "ok"); return r; } catch (e) { A.fail(e); return null; } };
  A.modal = function (title, body, onClose) {
    const bg = h("div", { class: "modal-bg", onclick: (e) => { if (e.target === bg) close(); } });
    const close = () => { bg.remove(); if (onClose) onClose(); };
    bg.appendChild(h("div", { class: "modal", role: "dialog", "aria-label": title }, h("div", { class: "row sp" }, h("h2", {}, title), h("button", { onclick: close }, "Close")), body));
    document.body.appendChild(bg);
    return close;
  };
  A.confirmNote = function (title, label, required, onOk, extra) {
    const note = h("textarea", { placeholder: label });
    const fields = extra ? extra() : { el: null, get: () => ({}) };
    const close = A.modal(title, h("div", { class: "col" }, fields.el, h("div", { class: "field" }, h("label", {}, label + (required ? " (required)" : "")), note),
      h("div", { class: "row" }, h("button", { class: "primary", onclick: async () => {
        if (required && !note.value.trim()) { A.toast("A note is required.", "bad"); return; }
        close(); await onOk(note.value.trim(), fields.get());
      } }, "Confirm"))));
  };

  // ---------------------------------------------------------------- formatting / components
  A.fmtUtc = (iso, offset) => {
    if (!iso) return "-";
    const off = offset === undefined ? A.offset : offset;
    const d = new Date(new Date(iso).getTime() + off * 60000);
    const p = (n) => String(n).padStart(2, "0");
    const sign = off >= 0 ? "+" : "-", ao = Math.abs(off);
    return `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())} ${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())} UTC${sign}${p(Math.floor(ao / 60))}:${p(ao % 60)}`;
  };
  A.fmtOff = (s) => (s === null || s === undefined) ? "-" : `${Number(s).toFixed(1)}s`;
  // Detections are CANDIDATES for officer review; the labels say so where the evidence is indirect.
  const EVENT_LABELS = { rash_driving: "aggressive-driving candidate", possible_collision: "possible collision (candidate)", sudden_braking: "sudden braking (candidate)",
    overspeed: "overspeed (candidate)", wrong_way: "wrong-way (candidate)", wrong_lane: "wrong-lane (candidate)" };
  A.title = (s) => EVENT_LABELS[s] || String(s || "").replace(/_/g, " ");
  A.chip = (text, cls) => h("span", { class: "chip " + (cls || "") }, A.title(text));
  A.sevChip = (s) => A.chip(s, s);
  const STATE_CLS = { live: "ok", completed: "info", idle: "", connecting: "warn", reconnecting: "warn", stalled: "bad", failed: "bad", stopped: "", paused: "warn", interrupted: "bad", starting: "warn",
    ready_for_review: "warn", approved_for_submission: "ok", rejected: "bad", needs_further_review: "warn", submitted: "info", submission_failed: "bad", challan_reference_received: "ok", closed: "",
    detected: "info", under_investigation: "warn", confirmed: "bad", dismissed: "", escalated: "crit" };
  A.stateChip = (s) => A.chip(s, STATE_CLS[s] || "");
  A.classChip = (c) => c === "operational" ? A.chip("operational", "ok") : (c === "simulated" ? A.chip("SIMULATED", "sim") : A.chip("demo data", "warn"));
  A.banner = (kind, text) => h("div", { class: "banner " + kind, role: kind === "bad" ? "alert" : null }, text);
  A.empty = (text) => h("div", { class: "empty" }, text);
  A.kv = (obj) => h("table", {}, h("tbody", {}, Object.entries(obj).map(([k, v]) => h("tr", {}, h("th", { style: "width:34%" }, A.title(k)), h("td", {}, v === null || v === undefined || v === "" ? "-" : (v instanceof Node ? v : String(v)))))));
  A.json = (o) => h("pre", { class: "json" }, JSON.stringify(o, null, 1));
  A.pager = (total, limit, offset, go) => h("div", { class: "pager" }, h("span", { class: "muted small" }, total ? `${offset + 1}-${Math.min(total, offset + limit)} of ${total}` : "0 results"),
    h("button", { disabled: offset <= 0, onclick: () => go(Math.max(0, offset - limit)) }, "Prev"), h("button", { disabled: offset + limit >= total, onclick: () => go(offset + limit) }, "Next"));
  A.thumb = (assetId) => assetId ? h("img", { class: "thumb", src: `/api/v1/evidence/${assetId}/file`, alt: "evidence thumbnail", loading: "lazy" }) : h("div", { class: "thumb muted small", style: "display:flex;align-items:center;justify-content:center" }, "no image");
  A.spark = function (values, w, hgt) {
    w = w || 300; hgt = hgt || 70;
    const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg");
    svg.setAttribute("class", "spark"); svg.setAttribute("viewBox", `0 0 ${w} ${hgt}`); svg.setAttribute("preserveAspectRatio", "none");
    const vals = values.filter((v) => v !== null && v !== undefined);
    if (vals.length < 2) return svg;
    const max = Math.max(...vals, 1), step = w / (values.length - 1);
    const pts = values.map((v, i) => [i * step, hgt - 4 - ((v || 0) / max) * (hgt - 10)]);
    const line = document.createElementNS(ns, "polyline"); line.setAttribute("class", "l"); line.setAttribute("points", pts.map((p) => p.join(",")).join(" "));
    const area = document.createElementNS(ns, "polygon"); area.setAttribute("class", "a"); area.setAttribute("points", `0,${hgt} ` + pts.map((p) => p.join(",")).join(" ") + ` ${w},${hgt}`);
    svg.appendChild(area); svg.appendChild(line);
    return svg;
  };
  A.lightbox = (src, caption) => A.modal(caption || "Evidence", h("img", { src, alt: caption || "evidence frame" }));
  A.can = (perm) => !!(A.user && A.user.permissions.includes(perm));

  // ---------------------------------------------------------------- polling / cleanup
  A.onLeave = (fn) => A.cleanups.push(fn);
  A.poll = function (fn, ms) {
    let stop = false, timer = null;
    const tick = async () => { if (stop) return; try { await fn(); } catch (e) { /* surfaced by the view */ } if (!stop) timer = setTimeout(tick, ms); };
    tick();
    A.onLeave(() => { stop = true; clearTimeout(timer); });
  };

  // ---------------------------------------------------------------- router + layout
  const NAV = [["overview", "Overview", "view.overview"], ["cameras", "Cameras", "view.cameras"], ["live", "Live monitoring", "view.cameras"], ["incidents", "Incidents", "view.incidents"],
    ["cases", "Violations / e-challan", "view.cases"], ["congestion", "Congestion", "view.congestion"], ["history", "History & audit", "view.incidents"], ["admin", "Administration", "admin.settings"]];
  A.route = function (name, fn) { A.views[name] = fn; };
  A.go = (hash) => { location.hash = hash; };

  function render() {
    A.cleanups.splice(0).forEach((f) => { try { f(); } catch (e) { /* ignore */ } });
    const main = document.getElementById("main");
    if (!main) return;
    const parts = (location.hash || "#/overview").replace(/^#\/?/, "").split("/");
    const view = A.views[parts[0]] || A.views.overview;
    document.querySelectorAll("nav.side a").forEach((a) => a.classList.toggle("active", a.dataset.route === parts[0]));
    clear(main);
    if (main.parentElement) main.parentElement.classList.remove("lv");
    try { view(main, parts.slice(1).map(decodeURIComponent)); } catch (e) { main.appendChild(A.banner("bad", "This view failed to render: " + e.message)); }
  }

  A.showLogin = function (message) {
    const root = clear(document.getElementById("root"));
    A.cleanups.splice(0).forEach((f) => f());
    const u = h("input", { id: "u", autocomplete: "username", "aria-label": "username", placeholder: "username" });
    const p = h("input", { id: "p", type: "password", autocomplete: "current-password", "aria-label": "password", placeholder: "password" });
    const go = async () => {
      try { const r = await api("POST", "/auth/login", { username: u.value, password: p.value }); A.user = r.user; if (r.user.must_change_password) A.showPasswordChange(); else A.layout(); }
      catch (e) { A.toast(e.status === 429 ? "Too many attempts. Try again later." : "Sign-in failed: " + e.message, "bad"); }
    };
    p.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
    root.appendChild(h("div", { class: "login" }, h("div", { class: "card" }, h("h1", {}, "ANVESH"), h("div", { class: "muted small" }, "Road event investigation and traffic intelligence"),
      message ? A.banner("notice", message) : null, h("div", { class: "field" }, h("label", {}, "Username"), u), h("div", { class: "field" }, h("label", {}, "Password"), p),
      h("button", { class: "primary", onclick: go }, "Sign in"),
      h("div", { class: "muted small" }, "Authorised personnel only. Activity is audited."))));
    u.focus();
  };

  // A temporary password (set by an administrator) may only be used to choose a real one: the API refuses everything else until then.
  A.showPasswordChange = function () {
    const root = clear(document.getElementById("root"));
    const cur = h("input", { type: "password", autocomplete: "current-password", "aria-label": "temporary password" });
    const n1 = h("input", { type: "password", autocomplete: "new-password", "aria-label": "new password" });
    const n2 = h("input", { type: "password", autocomplete: "new-password", "aria-label": "repeat new password" });
    const go = async () => {
      if (n1.value !== n2.value) { A.toast("The two new passwords differ.", "bad"); return; }
      try { await api("POST", "/auth/password", { current_password: cur.value, new_password: n1.value }); A.user = null; A.showLogin("Password changed. Sign in with the new password."); }
      catch (e) { A.fail(e); }
    };
    root.appendChild(h("div", { class: "login" }, h("div", { class: "card" }, h("h1", {}, "Choose a new password"),
      A.banner("notice", "Your password was set by an administrator and is temporary. Choose your own (at least 10 characters) to continue."),
      h("div", { class: "field" }, h("label", {}, "Temporary password"), cur), h("div", { class: "field" }, h("label", {}, "New password"), n1), h("div", { class: "field" }, h("label", {}, "Repeat new password"), n2),
      h("button", { class: "primary", onclick: go }, "Change password"))));
    cur.focus();
  };

  A.layout = function () {
    const root = clear(document.getElementById("root"));
    const nav = h("nav", { class: "side", "aria-label": "Main" }, h("div", { class: "brand" }, "ANVESH"), h("div", { class: "tag" }, "Road event investigation"),
      NAV.filter((n) => A.can(n[2])).map((n) => h("a", { href: "#/" + n[0], "data-route": n[0] }, n[1])));
    const top = h("div", { class: "topbar" }, h("div", { id: "envbar" }), h("div", { class: "who" }, h("span", {}, A.user.display_name), A.chip(A.user.role, "info"),
      h("button", { onclick: async () => { await A.act(() => A.post("/auth/logout")); A.user = null; A.showLogin(); } }, "Sign out")));
    root.appendChild(h("div", { class: "app" }, nav, h("main", {}, top,
      // One compact bar; the full wording of both standing notices is one click away and is never removed.
      h("details", { class: "warnbar" },
        h("summary", {}, h("b", {}, "RESEARCH / DEMO USE ONLY"), h("span", { class: "wb-short" }, " - not cleared for enforcement" + (A.user.is_demo ? " (DEMO account)" : "") + ". Detections are CANDIDATES for human review; the reviewing officer is the final authority."), h("span", { class: "wb-more" }, "Read full notice")),
        h("div", { class: "wb-body" },
          h("p", {}, "RESEARCH / DEMO USE ONLY - not cleared for operational enforcement: no authorised registry or e-challan is connected, detector accuracy on your cameras is unmeasured, licences are not cleared, and MFA is not implemented." + (A.user.is_demo ? " You are signed in with a DEMO account." : "")),
          h("p", {}, "Detections are automated CANDIDATES for human review, not findings. Confidence tiers are heuristic labels, not probabilities. The reviewing officer is the final authority."))),
      h("div", { id: "main" }))));
    addEventListener("hashchange", render);
    if (!location.hash) location.hash = "#/overview"; else render();
    A.refreshEnv();
  };

  A.refreshEnv = async function () {
    try {
      const s = await A.get("/system/status");
      const env = document.getElementById("envbar");
      if (!env) return;
      const ints = s.integrations;
      const challan = ints.challan.find((x) => x.key === "official"), reg = ints.registry.find((x) => x.key === "production");
      clear(env).appendChild(h("div", { class: "row small" },
        A.chip("OCR: " + (ints.ocr.available ? "available" : "unavailable"), ints.ocr.available ? "ok" : "warn"),
        A.chip("Registry: " + (reg.state === "connected" ? "connected" : "not connected"), reg.state === "connected" ? "ok" : "warn"),
        A.chip("e-Challan: " + (challan.state === "connected" ? "connected" : "not connected"), challan.state === "connected" ? "ok" : "warn"),
        A.chip("Alerts: " + (ints.alerts.some((a) => a.connected && !a.is_test) ? "live channel" : "test adapter only"), ints.alerts.some((a) => a.connected && !a.is_test) ? "ok" : "warn"),
        A.chip(s.gpu.cuda ? "GPU" : "CPU", s.gpu.cuda ? "ok" : "")));
    } catch (e) { /* non-critical */ }
  };

  A.start = async function () {
    try { const r = await api("GET", "/auth/me"); A.user = r.user; if (r.user.must_change_password) A.showPasswordChange(); else A.layout(); } catch (e) { A.showLogin(); }
  };
  return A;
})();
