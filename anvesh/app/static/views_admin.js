/* Congestion intelligence, history/reporting/audit, administration. */
"use strict";
(function (A) {
  const { h } = A;
  const LEVEL_CLS = { congested: "bad", building: "warn", dissipating: "info", free_flow: "ok" };

  // ---------------------------------------------------------------- Congestion
  A.route("congestion", (main) => {
    main.appendChild(h("h1", {}, "Congestion intelligence"));
    main.appendChild(A.banner("notice", "Congestion level is a hysteresis-damped count/occupancy PROXY in image space unless the camera has validated metric geometry. Candidate causes are ranked hypotheses (belief/plausibility), never established causes; affected lanes appear only where lane polygons are configured."));
    const box = h("div", { class: "col" }); main.appendChild(box);
    A.poll(async () => {
      const r = await A.get("/congestion");
      A.clear(box).appendChild(r.cameras.length ? h("div", { class: "grid g2" }, r.cameras.map((c) => {
        const trend = c.live_trend, last = trend[trend.length - 1];
        return h("div", { class: "card col" }, h("div", { class: "row sp" }, h("h2", {}, c.name), h("div", { class: "row" }, A.classChip(c.data_class), A.stateChip(c.state))),
          c.location ? h("div", { class: "muted small" }, c.location) : null,
          trend.length ? h("div", { class: "col" }, h("div", { class: "row sp" }, h("span", {}, "Now: ", last && last.level ? A.chip(last.level, LEVEL_CLS[last.level]) : A.chip("unavailable", "bad")), h("span", { class: "muted small" }, last ? `${last.vehicle_count} vehicles in window, flow ${last.flow_rate}/s` : "")),
            A.spark(trend.map((w) => w.vehicle_count)), h("div", { class: "row" }, trend.slice(-10).map((w) => A.chip(w.level || "n/a", LEVEL_CLS[w.level] || "")))) : h("div", { class: "muted" }, "No live trend (camera idle)."),
          Object.keys(c.unique_tracks_by_class).length ? h("div", { class: "row small" }, h("span", { class: "muted" }, "distinct tracks:"), Object.entries(c.unique_tracks_by_class).map(([k, v]) => A.chip(`${v} ${k}`))) : null,
          h("h3", {}, "Recent congestion episodes"), c.recent_episodes.length ? c.recent_episodes.map((e) => h("div", { class: "card tight col" }, h("div", { class: "row sp" }, h("a", { href: "#/incidents/" + e.incident_id }, e.incident_id), h("span", {}, A.sevChip(e.severity), " ", A.stateChip(e.status))),
            h("div", { class: "small" }, `peak ${A.title(e.indicators.peak_level)}, ${Math.round(e.indicators.duration_s || 0)}s span` + (e.indicators.non_free_s !== undefined ? ` (${Math.round(e.indicators.non_free_s)}s not free-flowing, ${e.indicators.interruptions || 0} short free-flow gap(s) merged)` : "") + `, ${e.indicators.windows} windows; started ${A.fmtUtc(e.start_utc)}`),
            e.candidate_causes.length ? h("div", { class: "small" }, h("b", {}, "Candidate causes: "), e.candidate_causes.map((x) => x.hypothesis_id ? `${x.hypothesis_id} (belief ${x.belief}, plausibility ${x.plausibility}, ${x.confidence_tier})` : A.title(x.outcome)).join("; ")) : h("div", { class: "small muted" }, "No cause ranked."),
            h("ul", { class: "small" }, e.uncertainty.slice(0, 3).map((u) => h("li", {}, u))))) : h("div", { class: "muted" }, "None recorded."));
      })) : A.empty("No cameras."));
    }, 4000);
  });

  // ---------------------------------------------------------------- History & audit
  A.route("history", (main) => {
    main.appendChild(h("h1", {}, "History, reporting and audit"));
    let tab = "stats"; const tabs = h("div", { class: "tabs" }), body = h("div", {});
    const defs = [["stats", "Operational statistics"], ["incidents", "Incident history"], ["cases", "Case history"]].concat(A.can("audit.view") ? [["audit", "Audit log"]] : []);
    const draw = () => { A.clear(tabs); defs.forEach(([k, l]) => tabs.appendChild(h("button", { class: tab === k ? "on" : "", onclick: () => { tab = k; draw(); show(); } }, l))); };
    main.appendChild(tabs); main.appendChild(body);
    draw(); show();
    async function show() {
      A.clear(body);
      if (tab === "stats") return stats(body);
      if (tab === "incidents") return history(body, "/incidents", ["incident_id", "event_type", "severity", "status", "camera_id", "detected_utc", "data_class"], (r) => "#/incidents/" + r.incident_id, "from_utc");
      if (tab === "cases") return history(body, "/cases", ["case_id", "offence_code", "state", "incident_id", "created_by", "updated_at", "data_class"], (r) => "#/cases/" + r.case_id);
      if (tab === "audit") return auditView(body);
    }
  });
  async function stats(box) {
    const s = await A.act(() => A.get("/stats")); if (!s) return;
    const total = Object.values(s.by_type).reduce((a, b) => a + b, 0) || 1;
    const days = {}; s.incidents_by_day.forEach((r) => { days[r.day] = (days[r.day] || 0) + r.n; });
    box.appendChild(h("div", { class: "grid g2" },
      h("div", { class: "card" }, h("h2", {}, "Incidents by type"), Object.keys(s.by_type).length ? Object.entries(s.by_type).sort((a, b) => b[1] - a[1]).map(([k, v]) => h("div", { class: "col", style: "margin:4px 0" }, h("div", { class: "row sp small" }, h("span", {}, A.title(k)), h("span", {}, v)), h("div", { class: "bar" }, h("i", { style: `width:${(v / total) * 100}%` })))) : A.empty("No incidents.")),
      h("div", { class: "card" }, h("h2", {}, "Incidents by day"), Object.keys(days).length ? Object.entries(days).map(([d, n]) => h("div", { class: "row sp small" }, h("span", {}, d), h("span", {}, n))) : A.empty("No data.")),
      h("div", { class: "card" }, h("h2", {}, "Case outcomes"), s.case_states.length ? s.case_states.map((r) => h("div", { class: "row sp" }, A.stateChip(r.state), r.n)) : A.empty("No cases.")),
      h("div", { class: "card" }, h("h2", {}, "Alert deliveries"), s.alert_deliveries.length ? s.alert_deliveries.map((r) => h("div", { class: "row sp small" }, h("span", {}, `${r.status} ${r.is_test ? "(test adapter)" : ""} ${r.delivery_confirmed ? "confirmed" : "not confirmed"}`), r.n)) : A.empty("No alerts."),
        h("h3", {}, "Processing sessions"), s.sessions.map((r) => h("div", { class: "row sp small" }, A.stateChip(r.state), r.n)))));
  }
  async function history(box, path, cols, link, dateKey) {
    const f = { q: "", from_utc: "", to_utc: "" }; let offset = 0; const list = h("div", {});
    box.appendChild(h("div", { class: "card row" }, h("input", { placeholder: "search", onkeydown: (e) => { if (e.key === "Enter") { f.q = e.target.value; offset = 0; load(); } } }),
      dateKey ? h("span", { class: "row small" }, "from", h("input", { type: "date", onchange: (e) => { f.from_utc = e.target.value ? e.target.value + "T00:00:00.000Z" : ""; load(); } }), "to", h("input", { type: "date", onchange: (e) => { f.to_utc = e.target.value ? e.target.value + "T23:59:59.999Z" : ""; load(); } })) : null));
    box.appendChild(list);
    async function load() {
      const r = await A.act(() => A.get(path + A.qs(Object.assign({ limit: 30, offset }, f)))); if (!r) return;
      A.clear(list).appendChild(h("div", { class: "card" }, r.items.length ? h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, cols.map((c) => h("th", {}, A.title(c))))), h("tbody", {}, r.items.map((row) => h("tr", { class: "click", onclick: () => A.go(link(row)) }, cols.map((c) => h("td", { class: "small" }, c.endsWith("_utc") || c.endsWith("_at") ? A.fmtUtc(row[c]) : String(row[c])))))))) : A.empty("Nothing found."), A.pager(r.total, r.limit, r.offset, (o) => { offset = o; load(); })));
    }
    load();
  }
  async function auditView(box) {
    const f = { actor: "", action: "", entity_type: "", entity_id: "" }; let offset = 0; const list = h("div", {});
    const inputs = Object.keys(f).map((k) => h("input", { placeholder: k, onkeydown: (e) => { if (e.key === "Enter") { f[k] = e.target.value; offset = 0; load(); } } }));
    box.appendChild(h("div", { class: "card row" }, inputs, A.can("audit.verify") ? h("button", { onclick: async () => { const r = await A.act(() => A.get("/audit/verify")); if (r) A.toast(r.ok ? `Audit chain intact: ${r.rows} entries, head ${r.head.hash.slice(0, 12)}...` : `CHAIN BROKEN at entry ${r.first_bad_seq}`, r.ok ? "ok" : "bad"); } }, "Verify hash chain") : null));
    box.appendChild(list);
    async function load() {
      const r = await A.act(() => A.get("/audit" + A.qs(Object.assign({ limit: 50, offset }, f)))); if (!r) return;
      A.clear(list).appendChild(h("div", { class: "card" }, h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, ["#", "Time", "Actor", "Action", "Entity", "Detail"].map((c) => h("th", {}, c)))),
        h("tbody", {}, r.items.map((a) => h("tr", {}, h("td", { class: "small" }, a.seq), h("td", { class: "small nowrap" }, A.fmtUtc(a.ts)), h("td", {}, a.actor, a.actor_role ? h("span", { class: "muted small" }, " (" + a.actor_role + ")") : null), h("td", {}, a.action), h("td", { class: "small mono" }, (a.entity_type || "") + " " + (a.entity_id || "")), h("td", { class: "small mono" }, JSON.stringify(a.details).slice(0, 140))))))),
        A.pager(r.total, r.limit, r.offset, (o) => { offset = o; load(); })));
    }
    load();
  }

  // ---------------------------------------------------------------- Administration
  A.route("admin", (main) => {
    main.appendChild(h("h1", {}, "Administration"));
    let tab = "integrations"; const tabs = h("div", { class: "tabs" }), body = h("div", {});
    const defs = [["integrations", "Integrations"], ["users", "Users"], ["detection", "Detection and thresholds"], ["offences", "Offence catalogue"], ["alerts", "Alert destinations"], ["retention", "Retention"], ["capabilities", "Event capabilities"]];
    const draw = () => { A.clear(tabs); defs.forEach(([k, l]) => tabs.appendChild(h("button", { class: tab === k ? "on" : "", onclick: () => { tab = k; draw(); show(); } }, l))); };
    main.appendChild(tabs); main.appendChild(body); draw(); show();
    async function show() { A.clear(body); try { await ({ integrations, users, detection, offences, alertsTab, retention, capsTab })[tab](body); } catch (e) { body.appendChild(A.banner("bad", e.message)); } }
  });

  async function integrations(box) {
    const s = await A.get("/system/status");
    const row = (label, state, detail, setup) => h("tr", {}, h("td", {}, h("b", {}, label)), h("td", {}, A.chip(state, state === "connected" || state === "available" ? "ok" : (state === "sandbox" ? "sim" : "warn"))), h("td", { class: "small" }, detail, setup ? h("div", { class: "muted" }, "Setup: " + setup) : null));
    const rows = [
      row("Plate OCR", s.integrations.ocr.available ? "connected" : "not_connected", s.integrations.ocr.available ? s.integrations.ocr.engine : s.integrations.ocr.reason, s.integrations.ocr.setup),
      row("Plate detector", s.integrations.plate_detector.available ? "connected" : "not_connected", s.integrations.plate_detector.available ? s.integrations.plate_detector.model : s.integrations.plate_detector.reason, s.integrations.plate_detector.setup)];
    s.integrations.registry.forEach((r) => rows.push(row("Registry: " + r.label, r.state, r.detail, r.setup)));
    s.integrations.challan.forEach((r) => rows.push(row("e-Challan: " + r.label, r.state, r.detail)));
    s.integrations.alerts.forEach((a) => rows.push(row("Alerts: " + a.label, a.connected ? (a.is_test ? "sandbox" : "connected") : "not_connected",
      a.is_test ? "TEST adapter: writes a local file, notifies no one" : (a.connected ? "configured" : "not configured on this server (set the ANVESH_ALERT_WEBHOOK_URL / SMTP variables)"))));
    const head = h("tr", {}, ["Integration", "State", "Detail / setup"].map((c) => h("th", {}, c)));
    const sys = A.kv({ version: s.version, schema: s.schema_version, gpu: s.gpu.cuda ? s.gpu.device : "none (CPU)", detector_weights: s.detector_weights_present ? "present" : "MISSING",
      four_eyes: s.policy.require_distinct_approver ? "enforced" : "disabled", audit_head: "#" + s.audit_head.seq + " " + s.audit_head.hash.slice(0, 16) + "..." });
    box.appendChild(h("div", { class: "col" },
      A.banner("notice", "Each external system is shown as it truly is. Nothing is reported connected unless a real, configured, tested channel exists. ANVESH holds no government-database access."),
      h("div", { class: "card table-wrap" }, h("table", {}, h("thead", {}, head), h("tbody", {}, rows))),
      h("div", { class: "card" }, h("h2", {}, "System"), sys)));
  }

  async function users(box) {
    const [r, reqs] = await Promise.all([A.get("/users"), A.get("/users/requests?status=pending")]);
    const roleSel = (cur) => h("select", {}, r.roles.map((x) => h("option", { value: x, selected: x === cur }, x)));
    const u = h("input", { placeholder: "username" }), d = h("input", { placeholder: "display name" }), p = h("input", { type: "password", placeholder: "temporary password (min 10)" }), rs = roleSel("investigator");
    // A privileged change (administrator / officer accounts) comes back as a REQUEST that a different administrator must approve.
    const after = (res, okMsg) => { if (!res) return; A.toast(res.status === "pending" ? "Request created: " + res.note : okMsg, "ok"); show(); };
    const reqRows = reqs.items.map((q) => {
      const mine = q.requested_by === A.user.username;
      return h("tr", {}, h("td", {}, A.title(q.kind)), h("td", {}, h("b", {}, q.target_username), q.role ? h("div", { class: "muted small" }, "role: " + q.role) : null),
        h("td", { class: "small" }, q.requested_by), h("td", { class: "small" }, A.fmtUtc(q.requested_at)), h("td", { class: "small" }, A.fmtUtc(q.expires_at)),
        h("td", { class: "row" },
          mine ? h("span", { class: "muted small" }, "yours - a different administrator must approve") : h("button", { class: "primary", onclick: () => A.confirmNote("Approve account request", "Why is this correct? (identity checked how?)", true, async (note) => { after(await A.act(() => A.post("/users/requests/" + q.request_id + "/approve", { note }), "Approved"), "Approved"); }) }, "Approve"),
          h("button", { onclick: () => A.confirmNote(mine ? "Withdraw request" : "Reject request", "Reason", false, async (note) => { after(await A.act(() => A.post("/users/requests/" + q.request_id + "/reject", { note }), "Done"), "Done"); }) }, mine ? "Withdraw" : "Reject")));
    });
    box.appendChild(h("div", { class: "col" },
      A.banner("notice", "Creating or changing an administrator or officer account, resetting their password, or re-enabling them needs approval by a DIFFERENT administrator. Passwords you set for others are temporary: they must change them at first sign-in. MFA is not implemented. Recovery when no second administrator exists: scripts/manage_users.py on the server."),
      reqs.items.length ? h("div", { class: "card" }, h("h2", {}, "Pending account requests (" + reqs.items.length + ")"), h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, ["Request", "Account", "Requested by", "When", "Expires", ""].map((c) => h("th", {}, c)))), h("tbody", {}, reqRows)))) : null,
      h("div", { class: "card" }, h("h2", {}, "Users and roles"), h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, ["User", "Role", "Active", "Last login", ""].map((c) => h("th", {}, c)))),
        h("tbody", {}, r.items.map((x) => { const sel = roleSel(x.role); return h("tr", {}, h("td", {}, h("b", {}, x.username), x.is_demo ? A.chip("demo account", "warn") : null, x.must_change_password ? A.chip("temporary password", "info") : null, h("div", { class: "muted small" }, x.display_name)), h("td", {}, sel), h("td", {}, x.active ? A.chip("active", "ok") : A.chip("disabled", "bad")), h("td", { class: "small" }, A.fmtUtc(x.last_login_at)),
          h("td", { class: "row" }, h("button", { onclick: async () => { after(await A.act(() => A.patch("/users/" + x.user_id, { role: sel.value })), "Role updated"); } }, "Set role"),
            h("button", { class: x.active ? "danger" : "", onclick: async () => { after(await A.act(() => A.patch("/users/" + x.user_id, { active: !x.active })), x.active ? "Disabled" : "Enabled"); } }, x.active ? "Disable" : "Enable"),
            h("button", { onclick: () => A.confirmNote("Reset password for " + x.username, "Reason", false, async (note, f) => { after(await A.act(() => A.patch("/users/" + x.user_id, { password: f.pw })), "Password reset"); },
              () => { const i = h("input", { type: "password", autocomplete: "new-password", placeholder: "temporary password (min 10)" }); return { el: h("div", { class: "field" }, h("label", {}, "Temporary password (they must change it at first sign-in)"), i), get: () => ({ pw: i.value }) }; }) }, "Reset password"),
            h("button", { onclick: async () => { const res = await A.act(() => A.post("/users/" + x.user_id + "/unlock", {})); if (res) A.toast("Cleared " + res.cleared + " counter(s) on this server process", "ok"); } }, "Unlock"))); })))),
        h("div", { class: "muted small" }, "Roles: administrator configures the system but cannot approve cases or see owner details; investigator opens cases; reviewer verifies evidence and plates; officer approves and submits. Demo accounts can never act on operational cases.")),
      h("div", { class: "card" }, h("h2", {}, "Add user"), h("div", { class: "row" }, u, d, rs, p, h("button", { class: "primary", onclick: async () => { after(await A.act(() => A.post("/users", { username: u.value, display_name: d.value, role: rs.value, password: p.value })), "User created"); } }, "Create")))));
    function show() { const b = box.parentNode; A.clear(b); users(b); }
  }

  async function detection(box) {
    const s = await A.get("/settings"), d = s.detection_defaults;
    const inputs = {}; for (const k of Object.keys(d)) inputs[k] = h("input", { type: "number", step: "any", value: d[k] });
    box.appendChild(h("div", { class: "card col" }, h("h2", {}, "Default detection thresholds"), h("div", { class: "muted small" }, "Defaults pre-filled for new cameras; each camera can override them in its configuration. These are untuned policy values, not accuracy claims."),
      h("div", { class: "form" }, Object.entries(inputs).map(([k, el]) => h("div", { class: "field" }, h("label", {}, A.title(k)), el))),
      h("button", { class: "primary", onclick: async () => { const body = {}; for (const [k, el] of Object.entries(inputs)) body[k] = Number(el.value); await A.act(() => A.put("/settings", { detection_defaults: body }), "Saved"); } }, "Save"),
      h("div", { class: "muted small" }, "Per-camera geometry, lanes, zones, signals, speed limits and calibration are configured on the camera. A displacement stop criterion is opt-in per camera; its false-positive rate is not yet evaluated.")));
  }

  async function offences(box) {
    const s = await A.get("/settings"); const rows = s.offences.map((o) => ({ code: h("input", { value: o.code }), desc: h("input", { value: o.description }), types: h("input", { value: o.event_types.join(", ") }), rule: h("input", { value: o.rule_reference || "", placeholder: "department to configure" }) }));
    box.appendChild(h("div", { class: "card col" }, h("h2", {}, "Offence catalogue"), h("div", { class: "banner notice" }, "ANVESH ships NO legal section references. The rule reference for each offence must be entered by the department; operational cases cannot be approved without it."),
      h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, ["Code", "Description", "Event types (comma separated)", "Rule reference"].map((c) => h("th", {}, c)))), h("tbody", {}, rows.map((r) => h("tr", {}, h("td", {}, r.code), h("td", {}, r.desc), h("td", {}, r.types), h("td", {}, r.rule)))))),
      h("button", { class: "primary", onclick: async () => { const items = rows.map((r) => ({ code: r.code.value, description: r.desc.value, event_types: r.types.value.split(",").map((x) => x.trim()).filter(Boolean), rule_reference: r.rule.value })); await A.act(() => A.put("/settings", { offences: items }), "Offence catalogue saved"); } }, "Save catalogue")));
  }

  async function alertsTab(box) {
    const s = await A.get("/settings"); const sev = ["info", "low", "medium", "high", "critical"];
    const adapters = s.alert_adapters, list = h("tbody", {}), rules = [];
    function addRow(r) {
      const row = { id: r.rule_id, name: h("input", { value: r.name }), min: h("select", {}, sev.map((x) => h("option", { value: x, selected: x === r.min_severity }, x))),
        types: h("input", { value: (r.event_types || []).join(", "), placeholder: "all types" }),
        adapter: h("select", {}, adapters.map((a) => h("option", { value: a.adapter, selected: a.adapter === r.adapter, disabled: !a.connected }, a.adapter + (a.connected ? "" : " (not configured)")))),
        rcpt: h("input", { value: r.recipient }), en: h("input", { type: "checkbox", checked: r.enabled }) };
      rules.push(row);
      list.appendChild(h("tr", {}, h("td", {}, row.en), h("td", {}, row.name), h("td", {}, row.min), h("td", {}, row.types), h("td", {}, row.adapter), h("td", {}, row.rcpt)));
    }
    s.alert_rules.forEach(addRow);
    const head = h("tr", {}, ["On", "Name", "Min severity", "Event types", "Channel", "Recipient"].map((c) => h("th", {}, c)));
    box.appendChild(h("div", { class: "card col" }, h("h2", {}, "Alert destinations"),
      A.banner("notice", "Rules route incidents to authorised recipients. Demo/simulated incidents may only use the TEST adapter (nothing is sent externally). A webhook/email channel appears as usable only when configured on the server; delivery is 'confirmed' only if the receiving endpoint accepts it."),
      h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, head), list)),
      h("div", { class: "row" },
        h("button", { onclick: () => addRow({ name: "New rule", min_severity: "high", event_types: [], adapter: "test", recipient: "", enabled: true }) }, "Add rule"),
        h("button", { class: "primary", onclick: async () => {
          const items = rules.map((r) => ({ rule_id: r.id, name: r.name.value, enabled: r.en.checked, min_severity: r.min.value, event_types: r.types.value.split(",").map((x) => x.trim()).filter(Boolean), adapter: r.adapter.value, recipient: r.rcpt.value }));
          await A.act(() => A.put("/settings", { alert_rules: items }), "Alert rules saved"); } }, "Save rules"))));
  }

  async function retention(box) {
    const [s, rep] = await Promise.all([A.get("/settings"), A.get("/retention/report")]);
    const ev = h("input", { type: "number", value: s.retention.evidence_days }), au = h("input", { type: "number", value: s.retention.audit_days });
    const planBox = h("div", { class: "col" });
    const fmtBytes = (n) => n >= 1048576 ? (n / 1048576).toFixed(1) + " MiB" : (n / 1024).toFixed(1) + " KiB";
    async function showPlan() {
      A.clear(planBox);
      let plan;
      try { plan = await A.get("/retention/plan"); } catch (e) { A.fail(e); return; }
      const nothing = plan.assets_to_purge === 0 && plan.files_pending_removal === 0;
      planBox.appendChild(A.kv({ plan_id: plan.plan_id, evaluated: A.fmtUtc(plan.evaluated_at), cutoff: A.fmtUtc(plan.cutoff_utc), retention_days: plan.evidence_retention_days,
        files_to_purge: plan.assets_to_purge, space_to_free: fmtBytes(plan.bytes_to_free), files_pending_removal: plan.files_pending_removal,
        already_purged_before: plan.already_purged_assets, purge_allowed: plan.purge_allowed ? "yes" : "NO - " + plan.purge_blocked_reason }));
      if (plan.eligible_incidents.length) planBox.appendChild(h("div", { class: "card" }, h("h3", {}, "Incidents whose evidence files WOULD be deleted"), h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, ["Incident", "Detected", "Files", "Size"].map((c) => h("th", {}, c)))),
        h("tbody", {}, plan.eligible_incidents.map((i) => h("tr", {}, h("td", {}, i.incident_id), h("td", { class: "small" }, A.fmtUtc(i.detected_utc)), h("td", {}, i.assets), h("td", {}, fmtBytes(i.bytes)))))))));
      if (plan.files_to_purge.length) planBox.appendChild(h("details", {}, h("summary", {}, "File list (" + plan.files_to_purge.length + (plan.files_truncated ? "+, truncated" : "") + ")"),
        h("pre", { class: "json" }, plan.files_to_purge.map((f) => f.kind + "  " + f.relpath + "  " + fmtBytes(f.size_bytes) + "  sha256 " + f.sha256.slice(0, 12)).join("\n"))));
      planBox.appendChild(h("div", { class: "card" }, h("h3", {}, "PROTECTED - will not be touched (" + plan.protected_incidents.length + ")"),
        plan.protected_incidents.length ? h("table", {}, h("tbody", {}, plan.protected_incidents.map((i) => h("tr", {}, h("td", {}, i.incident_id), h("td", { class: "small" }, i.reasons.join("; ")))))) : h("div", { class: "muted small" }, "None past the retention period. Open incidents and any case that is not closed or rejected are always protected.")));
      planBox.appendChild(h("div", { class: "muted small" }, "Evidence rows, SHA-256 hashes and every audit record are kept; the audit log is never deleted. A purge cannot be undone except from a backup of the evidence directory."));
      if (!A.can("retention.manage") || nothing || !plan.purge_allowed) { if (nothing) planBox.appendChild(A.banner("notice", "Nothing is due for purging.")); return; }
      const reason = h("textarea", { placeholder: "Why is this purge being run? (at least 10 characters)" });
      const typed = h("input", { placeholder: "type the plan id to confirm", autocomplete: "off" });
      const ack = h("input", { type: "checkbox" });
      const go = h("button", { class: "danger", disabled: true }, "Delete these evidence files");
      const sync = () => { go.disabled = !(reason.value.trim().length >= 10 && typed.value.trim() === plan.plan_id && ack.checked); };
      [reason, typed].forEach((e) => e.addEventListener("input", sync)); ack.addEventListener("change", sync);
      go.addEventListener("click", async () => {
        if (go.disabled) return;
        go.disabled = true;
        try {
          const res = await A.post("/retention/purge", { plan_id: plan.plan_id, reason: reason.value.trim() });
          A.toast("Removed " + res.files_removed + " file(s), freed " + fmtBytes(res.bytes_freed) + (res.failed.length ? "; " + res.failed.length + " failed" : ""), res.failed.length ? "bad" : "ok");
          retentionRefresh();
        } catch (e) { A.fail(e); if (e.status === 409) showPlan(); }
      });
      planBox.appendChild(h("div", { class: "card col" }, h("h3", {}, "Execute purge"), h("div", { class: "field" }, h("label", {}, "Reason"), reason),
        h("div", { class: "field" }, h("label", {}, "Plan id (confirms you reviewed exactly this plan)"), typed),
        h("label", { class: "row small" }, ack, "I understand this permanently deletes the listed evidence files; hashes and audit records remain."), go));
    }
    function retentionRefresh() { const b = box.parentNode; A.clear(b); retention(b); }
    box.appendChild(h("div", { class: "col" }, h("div", { class: "grid g2" }, h("div", { class: "card col" }, h("h2", {}, "Retention policy"), h("div", { class: "field" }, h("label", {}, "Evidence retention (days)"), ev), h("div", { class: "field" }, h("label", {}, "Audit retention (days; recorded only - audit records are never deleted)"), au),
      h("button", { class: "primary", onclick: async () => { await A.act(() => A.put("/settings", { retention: { evidence_days: Number(ev.value), audit_days: Number(au.value) } }), "Saved"); } }, "Save"), h("div", { class: "muted small" }, s.retention_note)),
      h("div", { class: "card" }, h("h2", {}, "Retention report"), A.kv({ cutoff: A.fmtUtc(rep.cutoff_utc), incidents_past_retention: rep.past_retention, protected_by_open_cases: rep.blocked_by_open_cases, files_eligible: rep.assets_to_purge, purge_scope: rep.purge_scope, audit_purge: "not implemented (by design)" }))),
      h("div", { class: "card col" }, h("h2", {}, "Purge plan"), h("div", { class: "muted small" }, "Nothing is deleted automatically. Preview exactly what a purge would remove, then confirm."),
        h("button", { onclick: showPlan }, "Preview purge plan"), planBox)));
  }

  async function capsTab(box) {
    const r = await A.get("/capabilities");
    box.appendChild(h("div", { class: "card table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, ["Event", "Status", "Requires", "Real-footage validation", "Enforcement candidate"].map((c) => h("th", {}, c)))),
      h("tbody", {}, r.events.map((e) => h("tr", {}, h("td", {}, h("b", {}, e.label), h("div", { class: "muted small" }, e.description)), h("td", {}, A.chip(e.status, e.status === "operational" ? "ok" : (e.status === "experimental" ? "warn" : ""))), h("td", { class: "small" }, e.requires.join("; ")),
        h("td", { class: "small" }, e.validated_on_real_footage ? A.chip("demonstrated", "ok") : A.chip("not validated", "warn"), h("div", { class: "muted" }, e.real_footage_note)), h("td", {}, e.enforcement_candidate ? "yes" : "no")))))));
  }
})(window.ANV);
