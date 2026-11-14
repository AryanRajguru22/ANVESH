/* Incident centre and violation / e-challan centre. */
"use strict";
(function (A) {
  const { h } = A;
  const EVENT_TYPES = ["stopped_vehicle", "long_duration_stop", "congestion", "wrong_way", "wrong_lane", "overspeed", "sudden_braking", "rash_driving", "possible_collision", "red_light_violation", "illegal_turn"];
  const SEVERITIES = ["critical", "high", "medium", "low", "info"];
  const INC_STATUS = ["detected", "under_investigation", "confirmed", "dismissed", "escalated", "closed"];
  const CASE_STATES = ["detected", "awaiting_evidence_verification", "awaiting_plate_verification", "ready_for_review", "approved_for_submission", "rejected", "needs_further_review", "submitted", "submission_failed", "challan_reference_received", "closed"];
  const evUrl = (id) => `/api/v1/evidence/${id}/file`;
  const select = (opts, cur, onchange, all) => h("select", { onchange: (e) => onchange(e.target.value) }, all ? h("option", { value: "" }, all) : null, opts.map((o) => h("option", { value: o, selected: o === cur }, A.title(o))));

  // ---------------------------------------------------------------- Incident list
  A.route("incidents", (main, params) => {
    if (params[0]) return incidentDetail(main, params[0]);
    const f = { event_type: "", severity: "", status: "", data_class: "", q: "", order: "priority" };
    const q = new URLSearchParams((location.hash.split("?")[1]) || ""); for (const [k, v] of q) if (k in f) f[k] = v;
    let offset = 0;
    const box = h("div", {});
    main.appendChild(h("h1", {}, "Incident centre"));
    const search = h("input", { placeholder: "search id, type, vehicle, notes", value: f.q, onkeydown: (e) => { if (e.key === "Enter") { f.q = e.target.value; offset = 0; load(); } } });
    main.appendChild(h("div", { class: "card row" }, search, select(EVENT_TYPES, f.event_type, (v) => { f.event_type = v; offset = 0; load(); }, "all event types"), select(SEVERITIES, f.severity, (v) => { f.severity = v; offset = 0; load(); }, "all severities"),
      select(INC_STATUS, f.status, (v) => { f.status = v; offset = 0; load(); }, "all statuses"), select(["operational", "demo", "simulated"], f.data_class, (v) => { f.data_class = v; offset = 0; load(); }, "all data classes"),
      select(["priority", "newest"], f.order, (v) => { f.order = v; load(); })));
    main.appendChild(box);
    async function load() {
      const r = await A.act(() => A.get("/incidents" + A.qs(Object.assign({ limit: 25, offset }, f))));
      if (!r) return;
      A.clear(box).appendChild(h("div", { class: "card" }, r.items.length ? h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, ["", "Event", "Severity", "Camera / time", "Confidence", "Status", "Evidence", "Data"].map((c) => h("th", {}, c)))),
        h("tbody", {}, r.items.map((i) => h("tr", { class: "click", onclick: () => A.go("#/incidents/" + i.incident_id) }, h("td", {}, A.thumb(i.thumbnail_asset_id)), h("td", {}, h("b", {}, A.title(i.event_type)), " ", i.data_class === "simulated" ? A.chip("SIMULATED", "sim") : null, (i.indicators || {}).queue_member ? A.chip("queue member", "") : null, h("div", { class: "muted small" }, i.incident_id)),
          h("td", {}, A.sevChip(i.severity)), h("td", {}, i.camera_id, h("div", { class: "muted small" }, A.fmtUtc(i.detected_utc), i.time_basis === "processing_clock" ? " (processing clock)" : "")),
          h("td", {}, A.chip(i.confidence_tier, "")), h("td", {}, A.stateChip(i.status)), h("td", {}, A.chip(i.evidence_completeness, i.evidence_completeness === "full" ? "ok" : "warn")), h("td", {}, A.classChip(i.data_class))))))) : A.empty("No incidents match."),
        A.pager(r.total, r.limit, r.offset, (o) => { offset = o; load(); })));
    }
    load();
  });

  // ---------------------------------------------------------------- Incident detail
  async function incidentDetail(main, iid) {
    const root = h("div", { class: "col" });
    main.appendChild(root);
    async function load() {
      let inc, offences;
      try { [inc, offences] = await Promise.all([A.get("/incidents/" + iid), A.get("/offences")]); } catch (e) { A.clear(root).appendChild(A.banner("bad", e.message)); return; }
      const assets = inc.evidence;
      const clip = assets.find((a) => a.kind === "clip");
      const annotated = assets.filter((a) => a.kind === "frame_annotated"), raw = assets.filter((a) => a.kind === "frame");
      let eTab = clip ? "clip" : "frames";
      const evBox = h("div", {}), evTabs = h("div", { class: "tabs" });
      const player = { v: null };
      const drawEv = () => {
        A.clear(evTabs); [["clip", "Clip"], ["frames", "Annotated frames"], ["raw", "Original frames"]].forEach(([k, l]) => evTabs.appendChild(h("button", { class: eTab === k ? "on" : "", onclick: () => { eTab = k; drawEv(); } }, l)));
        A.clear(evBox);
        if (eTab === "clip") evBox.appendChild(clip ? h("div", {}, (player.v = h("video", { class: "clip", controls: true, src: evUrl(clip.asset_id) })), h("div", { class: "muted small" }, `Annotated review clip (${clip.meta.codec}, ${clip.meta.frames} frames @ ${clip.meta.fps} fps), sha256 ${clip.sha256.slice(0, 16)}...${clip.meta.browser_playable === false ? " - codec may not play in this browser" : ""}`)) : A.empty("No clip was captured for this incident."));
        else {
          const list = eTab === "frames" ? annotated : raw;
          evBox.appendChild(list.length ? h("div", { class: "gallery" }, list.map((a) => h("div", { class: "fig" }, h("img", { src: evUrl(a.asset_id), alt: a.meta.label || "frame", onclick: () => A.lightbox(evUrl(a.asset_id), `${a.meta.label} - video ${A.fmtOff(a.offset_s)}`) }),
            h("div", { class: "small muted" }, `${a.meta.label || ""} - video ${A.fmtOff(a.offset_s)}${a.meta.gap_s > 0.3 ? ` (nearest retained frame, ${a.meta.gap_s}s from requested)` : ""}`)))) : A.empty("No frames."));
        }
      };
      const caps = inc.capability;
      A.clear(root).appendChild(h("div", { class: "col" },
        h("div", { class: "row sp" }, h("div", {}, h("a", { href: "#/incidents" }, "< Incidents"), h("h1", {}, A.title(inc.event_type), " ", A.sevChip(inc.severity), " ", A.stateChip(inc.status), " ", A.classChip(inc.data_class))), h("div", { class: "mono muted" }, inc.incident_id)),
        inc.data_class !== "operational" ? A.banner("test", `NON-PRODUCTION DATA (${inc.data_class}). This incident came from ${inc.data_class === "simulated" ? "a scripted simulation, not real footage" : "demo/recorded input"}; nothing here is a real enforcement matter.`) : null,
        inc.time_basis === "processing_clock" ? A.banner("notice", "The absolute event time is NOT known: this recording had no declared start time, so UTC times below are processing-clock times. The video offset is authoritative.") : null,
        h("div", { class: "grid g2" },
          h("div", { class: "card" }, h("h2", {}, "Evidence"), evTabs, evBox, inc.source_video ? h("a", { href: `#/live/${inc.session_id}` }, "Open source video replay for this session") : h("div", { class: "muted small" }, "Source video not available for replay (live or removed).")),
          h("div", { class: "col" },
            h("div", { class: "card" }, h("h2", {}, "Observed event"), A.kv({ summary: (inc.timeline.find((t) => t.kind === "detected") || {}).summary, camera: `${inc.camera || inc.camera_id} (${inc.camera_id})`, location: inc.location.label ? `${inc.location.label} (configured camera location${inc.location.latitude != null ? ", " + inc.location.latitude + ", " + inc.location.longitude : ""})` : "no location configured",
              started: `video ${A.fmtOff(inc.start_offset_s)} - ${A.fmtUtc(inc.start_utc)}`, detected: `video ${A.fmtOff(inc.detected_offset_s)} - ${A.fmtUtc(inc.detected_utc)}`, ended: inc.end_offset_s === null ? "still open / not observed to end" : `video ${A.fmtOff(inc.end_offset_s)}`,
              severity_basis: inc.severity_basis, confidence: `${inc.confidence_tier} - ${inc.confidence_basis}`, evidence_completeness: inc.evidence_completeness, alert_status: A.title(inc.alert_status),
              detector: `${inc.detector} v${inc.detector_version}`, motion_space: inc.motion_space + " / geometry " + inc.geometry_validity + (inc.speed_basis ? " / " + inc.speed_basis : "") })),
            h("div", { class: "card" }, h("h2", {}, "Uncertainty and limits"), h("ul", {}, inc.uncertainty.map((u) => h("li", {}, u))), caps && caps.real_footage_note && inc.data_class !== "simulated" ? h("div", { class: "muted small" }, "Detector validation: " + caps.real_footage_note) : (inc.data_class === "simulated" ? h("div", { class: "muted small" }, "Scripted SIMULATED scenario: it shows how the rule behaves, not how it performs on real footage.") : null)))),
        h("div", { class: "grid g2" },
          h("div", { class: "card" }, h("h2", {}, "Vehicles"), inc.involved.length ? h("table", {}, h("thead", {}, h("tr", {}, ["Track (session-scoped)", "Class", "Role", "Note", "History"].map((c) => h("th", {}, c)))),
            h("tbody", {}, inc.involved.map((v) => h("tr", {}, h("td", { class: "mono" }, v.track_id), h("td", {}, v.vehicle_class), h("td", {}, A.chip(v.role, v.role === "potentially_involved" ? "warn" : "")), h("td", { class: "small" }, v.note),
              h("td", { class: "small" }, h("a", { href: "#/incidents?q=" + encodeURIComponent(v.track_id) }, "events with this track")))))) : A.empty("No vehicle is attributed (e.g. a congestion episode)."),
            h("div", { class: "muted small" }, "Roles are 'observed' or 'potentially involved'. Neither establishes cause. A track id names one uninterrupted tracker run in this session, not a vehicle: if the tracker lost the vehicle and found it again it has another id, so this history can be incomplete."),
            h("h3", {}, "Established cause"), inc.established_cause ? h("div", {}, h("pre", { class: "json" }, inc.established_cause), h("div", { class: "muted small" }, "recorded by " + inc.established_cause_by)) : h("div", { class: "muted" }, "None recorded. Detectors never set this; only an authorised officer can, with an evidence basis.")),
          h("div", { class: "card" }, h("h2", {}, "Candidate causes"), inc.candidate_causes.length ? h("div", {}, inc.candidate_causes.map((c) => c.hypothesis_id ? h("div", { class: "row sp" }, h("span", {}, h("b", {}, c.hypothesis_id), " ", A.chip(c.outcome, "")), h("span", { class: "small muted" }, `belief ${c.belief}, plausibility ${c.plausibility}, ${c.confidence_tier}`)) : h("div", { class: "muted" }, "Ranking outcome: " + A.title(c.outcome))),
            h("div", { class: "muted small" }, "Dempster-Shafer belief/plausibility over hypotheses - NOT probabilities and NOT established causes.")) : h("div", { class: "muted" }, "No candidate-cause ranking for this event type."),
            inc.related.length ? h("div", {}, h("h3", {}, "Related incidents (temporal association only)"), inc.related.map((r) => h("div", { class: "small" }, h("a", { href: "#/incidents/" + r.incident_id }, r.incident_id), ` - ${A.title(r.relation)}, ${r.gap_s}s apart; no causal claim`))) : null,
            Object.keys(inc.indicators).length ? h("div", {}, h("h3", {}, "Indicators"), A.json(inc.indicators)) : null)),
        plateCard(inc, load), alertCard(inc, load), actionsCard(inc, offences, load),
        h("div", { class: "card" }, h("h2", {}, "Timeline"), h("ul", { class: "tl" }, inc.timeline.map((t) => h("li", { class: t.actor === "system" ? "system" : "" }, h("b", {}, A.title(t.kind)), " ", t.summary, h("div", { class: "muted small" }, `${A.fmtUtc(t.ts_utc)}${t.offset_s !== null ? " - video " + A.fmtOff(t.offset_s) : ""} - ${t.actor}`)))))));
      drawEv();
    }
    await load();
  }

  function plateCard(inc, reload) {
    const verif = A.can("case.verify_plate");
    const heads = inc.current_plates, byId = {}; inc.plates.forEach((p) => { byId[p.reading_id] = p; });
    const track = inc.track_ids[0];
    const status = (p) => A.chip(p.status, p.verified ? "ok" : (p.status === "unreadable" ? "bad" : "warn"));
    return h("div", { class: "card col" }, h("h2", {}, "Vehicle identification (plate)"),
      A.banner("notice", "OCR is not running in this build (no OCR engine or plate-detector weights are installed), so plates are entered by an officer from the evidence frames. An unreviewed OCR reading would never count as verified; OCR confidence is never registry verification."),
      inc.plates.length ? h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, ["Reading", "Source", "Status", "Format", "By", "Note", ""].map((c) => h("th", {}, c)))),
        h("tbody", {}, inc.plates.map((p) => h("tr", {}, h("td", {}, h("b", { class: "mono" }, p.normalized_text || "-"), p.raw_text && p.raw_text !== p.normalized_text ? h("div", { class: "muted small" }, "as entered: " + p.raw_text) : null, p.confidence !== null ? h("div", { class: "muted small" }, "OCR conf " + p.confidence) : null),
          h("td", {}, A.title(p.source)), h("td", {}, status(p), heads.some((x) => x.reading_id === p.reading_id) ? null : A.chip("superseded", "")), h("td", {}, p.format_valid ? "valid" : A.chip("invalid", "bad")), h("td", { class: "small" }, p.created_by + (p.supersedes ? " (supersedes " + p.supersedes.slice(-6) + ")" : "")), h("td", { class: "small" }, p.note),
          h("td", {}, heads.some((x) => x.reading_id === p.reading_id) && !p.verified && p.status !== "unreadable" ? h("div", { class: "row" }, verif ? h("button", { onclick: () => A.confirmNote("Verify plate " + p.normalized_text, "Verification note", false, async (note) => { await A.act(() => A.post(`/plates/${p.reading_id}/verify`, { note, allow_nonstandard: !p.format_valid }), "Plate verified"); reload(); }) }, "Verify") : null,
            A.can("plate.enter") ? h("button", { onclick: () => correct(p, reload) }, "Correct") : null) : null)))))) : A.empty("No plate reading yet."),
      A.can("plate.enter") ? h("div", { class: "row" }, h("button", { onclick: () => enterPlate(inc, track, reload) }, "Enter plate manually"),
        h("button", { onclick: async () => { const r = await A.act(() => A.post(`/incidents/${inc.incident_id}/plates/ocr`, { track_id: track })); if (!r) return; if (r.status === "unavailable") A.modal("OCR unavailable", h("div", { class: "col" }, h("ul", {}, r.reasons.map((x) => h("li", {}, x))), h("div", {}, "Next step: " + r.next_step), h("div", { class: "muted small" }, "Setup: install easyocr with its recognition weights, and place a plate-detector at models/plate_detector.pt. ANVESH never downloads these on its own."))); else { A.toast(`OCR: ${r.status}`, "ok"); reload(); } } }, "Run OCR on evidence")) : null);
  }
  function enterPlate(inc, track, reload) {
    const t = h("input", { placeholder: "e.g. MH12AB1234", style: "text-transform:uppercase" });
    const close = A.modal("Enter plate (officer reading)", h("div", { class: "col" }, h("div", { class: "field" }, h("label", {}, "Plate as read from the evidence frame"), t),
      h("div", { class: "muted small" }, "Recorded with source 'manual_entry'. A different reviewer must verify it before it counts as verified."),
      h("button", { class: "primary", onclick: async () => { const r = await A.act(() => A.post(`/incidents/${inc.incident_id}/plates`, { text: t.value, track_id: track }), "Plate recorded"); if (r) { close(); reload(); } } }, "Save")));
  }
  function correct(p, reload) {
    const t = h("input", { value: p.normalized_text || "" }), n = h("textarea", { placeholder: "why is the original wrong? (required)" });
    const close = A.modal("Correct plate reading", h("div", { class: "col" }, h("div", { class: "muted" }, "The original reading '" + (p.raw_text || "") + "' is preserved; this adds a new reading that supersedes it."), h("div", { class: "field" }, h("label", {}, "Corrected plate"), t), n,
      h("button", { class: "primary", onclick: async () => { const r = await A.act(() => A.post(`/plates/${p.reading_id}/correct`, { text: t.value, note: n.value }), "Correction recorded"); if (r) { close(); reload(); } } }, "Save correction")));
  }

  function alertCard(inc, reload) {
    return h("div", { class: "card" }, h("h2", {}, "Alerts to authorised recipients"),
      inc.alerts.length ? h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, ["Rule / recipient", "Channel", "Status", "Attempts", "Delivery", ""].map((c) => h("th", {}, c)))),
        h("tbody", {}, inc.alerts.map((a) => h("tr", {}, h("td", {}, a.channel, h("div", { class: "muted small" }, a.recipient)), h("td", {}, A.chip(a.adapter_kind, a.is_test ? "warn" : "")), h("td", {}, A.chip(a.status, a.status === "sent" ? "ok" : (a.status === "pending" ? "warn" : "bad")), a.last_error ? h("div", { class: "small muted" }, a.last_error) : null),
          h("td", {}, `${a.attempts}/${a.max_attempts}`), h("td", { class: "small" }, a.notice, a.confirmation ? h("div", { class: "muted" }, a.confirmation) : null),
          h("td", {}, a.status !== "sent" && A.can("alert.retry") && !(a.last_error || "").startsWith("blocked") ? h("button", { onclick: async () => { await A.act(() => A.post(`/alerts/${a.delivery_id}/retry`), "Retry queued"); reload(); } }, "Retry") : null)))))) : h("div", { class: "muted" }, "No alert was routed (severity below every rule's threshold, or no rules)."),
      h("div", { class: "muted small" }, "Only the test adapter is available unless a webhook/email channel is configured on the server; ANVESH never claims a recipient was notified unless the delivery is confirmed."));
  }

  function actionsCard(inc, offences, reload) {
    const note = h("textarea", { placeholder: "note (required to dismiss or close)" });
    const assignee = h("input", { placeholder: "assign to (username)", value: inc.assigned_to || "" });
    const applicable = offences.items.filter((o) => o.event_types.includes(inc.event_type));
    const offence = h("select", {}, applicable.map((o) => h("option", { value: o.code }, `${o.code} - ${o.description}`)));
    const cause = h("textarea", { placeholder: "established cause" }), basis = h("textarea", { placeholder: "evidence basis for that cause" });
    return h("div", { class: "grid g2" },
      h("div", { class: "card col" }, h("h2", {}, "Investigation"), h("div", { class: "row" }, A.stateChip(inc.status), inc.assigned_to ? h("span", { class: "muted" }, "assigned to " + inc.assigned_to) : null),
        A.can("incident.update") ? h("div", { class: "col" }, note, h("div", { class: "row" }, inc.allowed_status.map((st) => h("button", { class: st === "dismissed" ? "warn" : "", onclick: async () => { const r = await A.act(() => A.patch("/incidents/" + inc.incident_id, { status: st, note: note.value }), "Status: " + A.title(st)); if (r) reload(); } }, "Mark " + A.title(st))),
          h("button", { onclick: async () => { if (!note.value.trim()) return A.toast("Write a note first.", "bad"); const r = await A.act(() => A.post(`/incidents/${inc.incident_id}/notes`, { note: note.value }), "Note added"); if (r) reload(); } }, "Add note")),
          h("div", { class: "row" }, assignee, h("button", { onclick: async () => { const r = await A.act(() => A.patch("/incidents/" + inc.incident_id, { assigned_to: assignee.value }), "Assigned"); if (r) reload(); } }, "Assign"))) : h("div", { class: "muted" }, "Your role cannot change incidents."),
        inc.investigation_notes ? h("pre", { class: "json" }, inc.investigation_notes) : null,
        A.can("incident.establish_cause") ? h("details", {}, h("summary", {}, "Record an established cause (officer finding)"), h("div", { class: "col" }, cause, basis, h("button", { onclick: async () => { const r = await A.act(() => A.post(`/incidents/${inc.incident_id}/cause`, { cause: cause.value, basis: basis.value }), "Cause recorded"); if (r) reload(); } }, "Record with my name"))) : null),
      h("div", { class: "card col" }, h("h2", {}, "Violation cases"), inc.cases.length ? inc.cases.map((c) => h("div", { class: "row sp" }, h("a", { href: "#/cases/" + c.case_id }, c.case_id), h("span", {}, c.offence_code, " ", A.stateChip(c.state)))) : h("div", { class: "muted" }, "No case opened."),
        A.can("case.create") ? (applicable.length ? h("div", { class: "col" }, offence, h("button", { class: "primary", onclick: async () => { const r = await A.act(() => A.post(`/incidents/${inc.incident_id}/cases`, { offence_code: offence.value }), "Case opened"); if (r) A.go("#/cases/" + r.case_id); } }, "Open violation case from this incident"))
          : h("div", { class: "muted small" }, "No offence in the catalogue applies to this event type; it is investigated as an incident only.")) : null));
  }

  // ---------------------------------------------------------------- Cases
  A.route("cases", (main, params) => {
    if (params[0]) return caseDetail(main, params[0]);
    const f = { state: "", offence_code: "", data_class: "", q: "" };
    const q = new URLSearchParams((location.hash.split("?")[1]) || ""); for (const [k, v] of q) if (k in f) f[k] = v;
    let offset = 0; const box = h("div", {});
    main.appendChild(h("h1", {}, "Violations and e-challan centre"));
    main.appendChild(A.banner("notice", "ANVESH issues no challan. Cases move through evidence verification, plate verification, officer approval and a hand-off (sandbox simulator, evidence-package export, or an official gateway that is NOT connected). Test references are labelled non-production."));
    main.appendChild(h("div", { class: "card row" }, h("input", { placeholder: "search case, incident or plate", value: f.q, onkeydown: (e) => { if (e.key === "Enter") { f.q = e.target.value; offset = 0; load(); } } }),
      select(CASE_STATES, f.state, (v) => { f.state = v; offset = 0; load(); }, "all states"), select(["operational", "demo", "simulated"], f.data_class, (v) => { f.data_class = v; offset = 0; load(); }, "all data classes")));
    main.appendChild(box);
    async function load() {
      const r = await A.act(() => A.get("/cases" + A.qs(Object.assign({ limit: 25, offset }, f)))); if (!r) return;
      A.clear(box).appendChild(h("div", { class: "card" }, r.items.length ? h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, ["Case", "Offence", "State", "Plate", "Updated", "Data"].map((c) => h("th", {}, c)))),
        h("tbody", {}, r.items.map((c) => h("tr", { class: "click", onclick: () => A.go("#/cases/" + c.case_id) }, h("td", {}, h("b", { class: "mono" }, c.case_id), h("div", { class: "muted small" }, c.incident_id)), h("td", {}, c.offence_code), h("td", {}, A.stateChip(c.state)),
          h("td", {}, c.plate ? h("span", {}, h("b", { class: "mono" }, c.plate.text), " ", A.chip(c.plate.verified ? "verified" : c.plate.status, c.plate.verified ? "ok" : "warn")) : h("span", { class: "muted" }, "none")), h("td", { class: "small" }, A.fmtUtc(c.updated_at)), h("td", {}, A.classChip(c.data_class))))))) : A.empty("No cases match. Open one from an incident."),
        A.pager(r.total, r.limit, r.offset, (o) => { offset = o; load(); })));
    }
    load();
  });

  const MAIN_PATH = ["detected", "awaiting_evidence_verification", "awaiting_plate_verification", "ready_for_review", "approved_for_submission", "submitted", "challan_reference_received", "closed"];
  const ACTION_LABEL = { submit_for_evidence_verification: "Submit for evidence verification", verify_evidence: "Verify evidence", verify_plate: "Verify plate", approve: "Approve for submission", reject: "Reject", request_more: "Request more evidence / review",
    resume_review: "Resume review", retry_submission: "Retry submission", record_reference: "Record external reference", close: "Close case" };

  async function caseDetail(main, cid) {
    const root = h("div", { class: "col" }); main.appendChild(root);
    async function load() {
      let c, audit, offences, sys;
      try { [c, audit, offences, sys] = await Promise.all([A.get("/cases/" + cid), A.can("audit.view") ? A.get(`/audit?entity_type=case&entity_id=${cid}&limit=100`) : Promise.resolve(null), A.get("/offences"), A.get("/system/status")]); }
      catch (e) { A.clear(root).appendChild(A.banner("bad", e.message)); return; }
      const inc = c.incident, assets = c.evidence, clip = assets.find((a) => a.kind === "clip");
      const idx = MAIN_PATH.indexOf(c.state);
      const nonProd = c.labels.non_production;
      const doAction = (action) => {
        const needNote = ["reject", "request_more", "close", "record_reference"].includes(action);
        if (action === "record_reference") { A.confirmNote("Record external reference", "How was it obtained (e.g. portal confirmation)", true, async (note, x) => { const r = await A.act(() => A.post(`/cases/${cid}/actions`, { action, note, external_reference: x.ref }), "Reference recorded"); if (r) load(); }, () => { const i = h("input", { placeholder: "reference returned by the official system" }); return { el: h("div", { class: "field" }, h("label", {}, "External reference"), i), get: () => ({ ref: i.value }) }; }); return; }
        if (action === "approve") { A.confirmNote("Approve for submission", "Approval note (optional)", false, async (note) => { const r = await A.act(() => A.post(`/cases/${cid}/actions`, { action, note }), "Approved - you are recorded as the approving officer"); if (r) load(); }); return; }
        A.confirmNote(ACTION_LABEL[action], needNote ? "Reason" : "Note (optional)", needNote, async (note) => { const r = await A.act(() => A.post(`/cases/${cid}/actions`, { action, note }), ACTION_LABEL[action] + ": done"); if (r) load(); });
      };
      const handoffState = Object.fromEntries(sys.integrations.challan.map((x) => [x.key, x]));
      const lastHandoff = c.handoffs[0];
      A.clear(root).appendChild(h("div", { class: "col" },
        h("div", { class: "row sp" }, h("div", {}, h("a", { href: "#/cases" }, "< Cases"), h("h1", {}, c.case_id, " ", A.stateChip(c.state), " ", A.classChip(c.data_class))), h("div", { class: "row" },
          A.can("export.case") ? h("a", { href: `/api/v1/cases/${cid}/report.html`, target: "_blank", rel: "noopener" }, h("button", {}, "Printable report")) : null,
          A.can("export.case") ? h("a", { href: `/api/v1/cases/${cid}/package` }, h("button", {}, "Download evidence package")) : null)),
        nonProd ? A.banner("test", `NON-PRODUCTION CASE (${c.data_class}). Registry results and challan references on this case are TEST data with no legal effect.`) : null,
        h("div", { class: "card" }, h("div", { class: "steps" }, MAIN_PATH.map((s, i) => h("span", { class: "step " + (s === c.state ? "cur" : (idx > i ? "done" : "")) }, A.title(s))),
          ["rejected", "needs_further_review", "submission_failed"].includes(c.state) ? h("span", { class: "step cur" }, A.title(c.state)) : null)),
        h("div", { class: "grid g2" },
          h("div", { class: "card col" }, h("h2", {}, "Officer actions"), c.allowed_actions.length ? h("div", { class: "row" }, c.allowed_actions.filter((a) => a !== "verify_plate" || true).map((a) => h("button", { class: a === "approve" ? "primary" : (a === "reject" ? "danger" : (a === "request_more" ? "warn" : "")), onclick: () => doAction(a) }, ACTION_LABEL[a]))) : h("div", { class: "muted" }, "No action is available to your role in this state."),
            h("h3", {}, "Readiness for approval"), c.readiness.map((r) => h("div", { class: "check " + (r.ok ? "ok" : "no") }, h("span", { class: "mark" }, r.ok ? "OK" : "X"), h("span", {}, h("b", {}, A.title(r.item)), " - ", r.detail))),
            h("div", { class: "muted small" }, "Approval re-checks every item server-side; a missing item blocks it. The approving officer must differ from the evidence and plate verifiers and the case creator.")),
          h("div", { class: "card col" }, h("h2", {}, "Case"), A.kv({ offence: `${c.offence_code} - ${c.offence_description}`, rule_reference: c.rule_reference || "not configured (department to configure in Administration)", created_by: `${c.created_by} at ${A.fmtUtc(c.created_at)}`,
            evidence_verified_by: c.evidence_verified_by, plate_verified_by: c.plate_verified_by, approved_by: c.approved_by, external_reference: c.external_reference ? `${c.external_reference} (${A.title(c.external_reference_kind)})` : null, version: c.version }),
            c.labels.reference_notice ? A.banner("test", c.labels.reference_notice) : null,
            h("h3", {}, "Reason for classification"), h("div", {}, c.reason), h("h3", {}, "Known limitations"), h("pre", { class: "json" }, c.limitations || "-"),
            ["detected", "awaiting_evidence_verification", "awaiting_plate_verification", "ready_for_review", "needs_further_review"].includes(c.state) && (A.can("case.create") || A.can("case.review")) ? h("button", { onclick: () => editCase(c, offences, load) }, "Edit offence / reason") : null)),
        h("div", { class: "grid g2" },
          h("div", { class: "card col" }, h("h2", {}, "Event and evidence"), A.kv({ incident: h("a", { href: "#/incidents/" + inc.incident_id }, inc.incident_id), event: `${A.title(inc.event_type)} (${inc.severity})`, camera: inc.camera_id, when: `video ${A.fmtOff(inc.start_offset_s)} - ${A.fmtUtc(inc.start_utc)} [${A.title(inc.time_basis)}]`, location: inc.location.label || "-", confidence: `${inc.confidence_tier}: ${inc.confidence_basis}`, detector: inc.detector }),
            clip ? h("video", { class: "clip", controls: true, src: evUrl(clip.asset_id) }) : A.empty("No clip."),
            h("div", { class: "gallery" }, assets.filter((a) => a.kind === "frame_annotated").map((a) => h("img", { src: evUrl(a.asset_id), alt: a.meta.label, onclick: () => A.lightbox(evUrl(a.asset_id), a.meta.label) })))),
          h("div", { class: "col" },
            h("div", { class: "card col" }, h("h2", {}, "Plate"), c.current_plate ? h("div", { class: "col" }, h("div", { class: "row" }, h("b", { class: "mono", style: "font-size:18px" }, c.current_plate.normalized_text), A.chip(c.current_plate.status, c.current_plate.verified ? "ok" : "warn"), c.current_plate.format_valid ? null : A.chip("invalid format", "bad")),
              h("div", { class: "muted small" }, `source ${A.title(c.current_plate.source)}; ${c.current_plate.confidence !== null ? "OCR confidence " + c.current_plate.confidence : "not from OCR"}; by ${c.current_plate.created_by}`), h("a", { href: "#/incidents/" + inc.incident_id }, "Manage plate readings on the incident"))
              : h("div", {}, A.empty("No plate reading."), h("a", { href: "#/incidents/" + inc.incident_id }, "Enter the plate on the incident page"))),
            h("div", { class: "card col" }, h("h2", {}, "Registry verification"), c.registry_lookups.length ? c.registry_lookups.map((l) => h("div", { class: "col card tight" }, h("div", { class: "row" }, A.chip(l.status, l.status === "found" ? "ok" : "warn"), l.is_simulated ? A.chip("SANDBOX - SYNTHETIC", "sim") : null, h("span", { class: "muted small" }, `${l.source_label} - ${A.fmtUtc(l.retrieved_at)} by ${l.requested_by}`)),
              l.error ? h("div", { class: "small muted" }, l.error) : null, Object.keys(l.data).length ? A.kv(l.data) : null, l.owner_fields_hidden ? h("div", { class: "muted small" }, "Owner details are hidden for your role.") : null, l.notice ? h("div", { class: "small muted" }, l.notice) : null))
              : h("div", { class: "muted" }, "No lookup yet. Registry details are never inferred from OCR."),
              A.can("registry.lookup") && c.current_plate ? h("button", { onclick: async () => { const r = await A.act(() => A.post(`/cases/${cid}/registry-lookup`, {})); if (r) { A.toast("Lookup: " + r.status, r.status === "found" ? "ok" : "bad"); load(); } } }, "Look up registration" + (nonProd ? " (sandbox)" : "")) : null,
              c.registry_override_reason ? h("div", { class: "small" }, "LEGACY registry override (abolished 2026-10-21; this is not a registry result): " + c.registry_override_reason) : null,
              (handoffState.official && h("div", { class: "muted small" }, "Production registry: " + (sys.integrations.registry.find((x) => x.key === "production").state === "connected" ? "connected" : "NOT CONNECTED - no authorised registry access is configured.")))))),
        h("div", { class: "grid g2" },
          h("div", { class: "card col" }, h("h2", {}, "Hand-off / submission"), lastHandoff ? h("div", { class: "col" }, A.kv({ adapter: lastHandoff.adapter, status: lastHandoff.status, reference: lastHandoff.external_reference, error: lastHandoff.error, when: A.fmtUtc(lastHandoff.created_at), by: lastHandoff.submitted_by }), lastHandoff.is_simulated ? A.banner("test", "SANDBOX / TEST submission - no real challan exists.") : null,
            lastHandoff.status === "exported" ? A.banner("notice", "Evidence package exported. Submit it through the department's approved channel, then record the reference it returns.") : null) : h("div", { class: "muted" }, "Not submitted."),
            c.state === "approved_for_submission" && A.can("case.submit") ? h("div", { class: "col" }, h("div", { class: "row" },
              h("button", { onclick: async () => { const r = await A.act(() => A.post(`/cases/${cid}/handoff`, { adapter: "export" }), "Evidence package exported"); if (r) load(); } }, "Export evidence package (manual submission)"),
              nonProd ? h("button", { class: "primary", onclick: async () => { const r = await A.act(() => A.post(`/cases/${cid}/handoff`, { adapter: "sandbox" }), "Sandbox submission done (TEST reference)"); if (r) load(); } }, "Submit to SANDBOX simulator") : null),
              h("div", { class: "muted small" }, "Official gateway: " + handoffState.official.detail)) : null,
            h("div", { class: "muted small" }, "Idempotent: repeating a submission cannot create a second one.")),
          h("div", { class: "card" }, h("h2", {}, "State history"), h("ul", { class: "tl" }, c.transitions.map((t) => h("li", {}, h("b", {}, A.title(t.action)), ` ${t.from_state ? A.title(t.from_state) + " -> " : ""}${A.title(t.to_state)}`, h("div", { class: "muted small" }, `${A.fmtUtc(t.ts)} - ${t.actor} (${t.actor_role})${t.note ? " - " + t.note : ""}`)))))),
        auditCard(audit)));
    }
    await load();
  }
  function auditCard(audit) {
    if (!audit) return null;
    const head = h("tr", {}, ["Time", "Actor", "Action", "Detail"].map((x) => h("th", {}, x)));
    const rows = audit.items.map((a) => h("tr", {}, h("td", { class: "small nowrap" }, A.fmtUtc(a.ts)),
      h("td", {}, a.actor, a.actor_role ? h("span", { class: "muted small" }, " (" + a.actor_role + ")") : null), h("td", {}, a.action),
      h("td", { class: "small mono" }, JSON.stringify(a.details).slice(0, 160))));
    return h("div", { class: "card" }, h("h2", {}, "Audit trail for this case"), h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, head), h("tbody", {}, rows))));
  }
  function editCase(c, offences, reload) {
    const applicable = offences.items.filter((o) => o.event_types.includes(c.incident.event_type));
    const off = h("select", {}, applicable.map((o) => h("option", { value: o.code, selected: o.code === c.offence_code }, `${o.code} - ${o.description}`)));
    const reason = h("textarea", {}, c.reason), lim = h("textarea", {}, c.limitations || "");
    const close = A.modal("Edit case", h("div", { class: "col" }, h("div", { class: "field" }, h("label", {}, "Offence"), off), h("div", { class: "field" }, h("label", {}, "Reason for classification"), reason), h("div", { class: "field" }, h("label", {}, "Known limitations"), lim),
      h("button", { class: "primary", onclick: async () => { const r = await A.act(() => A.patch("/cases/" + c.case_id, { offence_code: off.value, reason: reason.value, limitations: lim.value }), "Case updated"); if (r) { close(); reload(); } } }, "Save")));
  }
})(window.ANV);
