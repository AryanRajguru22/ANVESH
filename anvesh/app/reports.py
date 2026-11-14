"""Incident / case reports and the evidence package.

The HTML report is self-contained and printable (browser "Save as PDF"). It states, on every page,
what the data is: a candidate detection awaiting human review, with its confidence tier (a heuristic
label, not a probability of guilt), its uncertainty, the time basis, and -- for non-operational data --
a prominent TEST banner. Owner/registration details appear only if the requesting user may see them.

The evidence PACKAGE is a zip: report.html, case.json, audit.json, the evidence files and a manifest
with a SHA-256 for each file, so a recipient can check nothing changed after export.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import zipfile
from pathlib import Path

from anvesh.app.timeutil import now_iso, to_local_label

_CSS = """body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#111;max-width:980px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;border-bottom:1px solid #999;padding-bottom:3px;margin-top:22px}
.banner{padding:8px 12px;margin:10px 0;font-weight:600;border:2px solid}.test{border-color:#b45309;background:#fef3c7;color:#78350f}
.notice{border-color:#1e3a8a;background:#eff6ff;color:#1e3a8a;font-weight:400}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #bbb;padding:4px 6px;text-align:left;vertical-align:top}
th{background:#f3f4f6}img{max-width:100%;border:1px solid #888}.fig{display:inline-block;width:48%;vertical-align:top;margin:0 1% 10px 0}
.small{font-size:12px;color:#444}ul{margin:4px 0 4px 18px;padding:0}@media print{.noprint{display:none}}"""


def _e(v) -> str:
    return html.escape("" if v is None else str(v))


def _rows(pairs) -> str:
    return "".join(f"<tr><th style='width:30%'>{_e(k)}</th><td>{v}</td></tr>" for k, v in pairs)


def case_report_html(case: dict, incident: dict, assets: list, images: dict, audit_rows: list, utc_offset: int = 330,
                     clip_href: str = None, generated_for: str = "") -> str:
    non_prod = case["data_class"] != "operational"
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>ANVESH case report {_e(case['case_id'])}</title><style>{_CSS}</style></head><body>"]
    parts.append(f"<h1>ANVESH case report - {_e(case['case_id'])}</h1><div class='small'>Generated {now_iso()} UTC for {_e(generated_for)}</div>")
    if non_prod:
        parts.append(f"<div class='banner test'>NON-PRODUCTION DATA ({_e(case['data_class']).upper()}): this case was built from demo/simulated input. "
                     "Any registry result or challan reference in it is test data and has no legal effect.</div>")
    parts.append("<div class='banner notice'>Candidate detection produced by an automated system for human review. The confidence tier is a heuristic label, "
                 "not a probability of guilt. The reviewing officer is the final authority; no challan is issued by this system.</div>")
    parts.append("<h2>Case</h2><table>" + _rows([
        ("Case / state", f"{_e(case['case_id'])} / <b>{_e(case['state'])}</b>"), ("Candidate offence", f"{_e(case['offence_code'])} - {_e(case['offence_description'])}"),
        ("Rule reference", _e(case["rule_reference"] or "not configured")), ("Reason for classification", _e(case["reason"])),
        ("Created", f"{_e(case['created_by'])} at {_e(case['created_at'])}"), ("Evidence verified by", _e(case["evidence_verified_by"] or "-")),
        ("Plate verified by", _e(case["plate_verified_by"] or "-")), ("Approved by", _e(case["approved_by"] or "-")),
        ("External reference", _e(case["external_reference"] or "-") + (f" ({_e(case['external_reference_kind'])})" if case["external_reference_kind"] else "")),
    ]) + "</table>")
    parts.append("<h2>Event</h2><table>" + _rows([
        ("Incident", _e(incident["incident_id"])), ("Type / severity", f"{_e(incident['event_type'])} / {_e(incident['severity'])} ({_e(incident['severity_basis'])})"),
        ("Camera / source", f"{_e(incident['camera_id'])} (session {_e(incident['session_id'])})"),
        ("Location (configured camera location)", _e(json.dumps(incident["location"]))),
        ("Event start", f"video t={incident['start_offset_s']:.1f}s - {_e(to_local_label(incident['start_utc'], utc_offset))} [{_e(incident['time_basis'])}]"),
        ("Detected", f"video t={incident['detected_offset_s']:.1f}s - {_e(to_local_label(incident['detected_utc'], utc_offset))}"),
        ("Detector", f"{_e(incident['detector'])} v{_e(incident['detector_version'])}; config {_e(json.dumps(incident['detector_config']))}"),
        ("Confidence", f"{_e(incident['confidence_tier'])} - {_e(incident['confidence_basis'])}"),
        ("Motion space / geometry", f"{_e(incident['motion_space'])} / {_e(incident['geometry_validity'])}" + (f" / speed basis {_e(incident['speed_basis'])}" if incident["speed_basis"] else "")),
        ("Indicators", f"<pre class='small'>{_e(json.dumps(incident['indicators'], indent=1)[:1800])}</pre>"),
    ]) + "</table>")
    parts.append("<h2>Vehicles</h2><table><tr><th>Track</th><th>Role</th><th>Class</th><th>Note</th></tr>" + "".join(
        f"<tr><td>{_e(v['track_id'])}</td><td>{_e(v['role'])}</td><td>{_e(v['vehicle_class'])}</td><td>{_e(v['note'])}</td></tr>" for v in incident["involved"]) + "</table>"
        "<div class='small'>Roles: <i>observed</i> / <i>potentially_involved</i> only. No vehicle is established as the cause by this system.</div>")
    plate = case.get("current_plate")
    if plate:
        parts.append("<h2>Plate</h2><table>" + _rows([
            ("Reading", f"<b>{_e(plate['normalized_text'])}</b> (original text '{_e(plate['raw_text'])}')"), ("Source / status", f"{_e(plate['source'])} / {_e(plate['status'])}"),
            ("OCR confidence", _e(plate["confidence"] if plate["confidence"] is not None else "n/a (not from OCR)")), ("Format valid", _e(plate["format_valid"])),
            ("Verified", _e(plate["verified"]))]) + "</table>")
    else:
        parts.append("<h2>Plate</h2><div class='small'>No plate reading.</div>")
    parts.append("<h2>Registry verification</h2>")
    if case["registry_lookups"]:
        for l in case["registry_lookups"]:
            data = "".join(f"<li>{_e(k)}: {_e(v)}</li>" for k, v in l["data"].items())
            parts.append(f"<p><b>{_e(l['status'])}</b> via {_e(l['source_label'])} at {_e(l['retrieved_at'])} by {_e(l['requested_by'])}"
                         f"{' - <b>SANDBOX / TEST DATA</b>' if l['is_simulated'] else ''}{' (owner fields hidden for this role)' if l['owner_fields_hidden'] else ''}</p><ul>{data}</ul>")
    else:
        parts.append("<div class='small'>No registry lookup performed. Registry details are never inferred from OCR.</div>")
    if case.get("registry_override_reason"):
        parts.append(f"<p class='small'><b>LEGACY registry override</b> (recorded before 2026-10-21, when overrides were abolished; it is not a registry "
                     f"result): {_e(case['registry_override_reason'])}</p>")
    parts.append("<h2>Known limitations / uncertainty</h2><ul>" + "".join(f"<li>{_e(u)}</li>" for u in incident["uncertainty"]) + "</ul>")
    parts.append("<h2>Evidence</h2>")
    for a in assets:
        if a["kind"] == "frame_annotated" and a["asset_id"] in images:
            parts.append(f"<div class='fig'><img src='{images[a['asset_id']]}' alt='evidence frame'><div class='small'>{_e(a['meta'].get('label'))} - "
                         f"t={a['offset_s']:.2f}s - sha256 {a['sha256'][:16]}...</div></div>")
    clips = [a for a in assets if a["kind"] == "clip"]
    for c in clips:
        link = f" - <a href='{_e(clip_href)}'>play</a>" if clip_href and not c.get("purged_at") else ""
        gone = f" - FILE PURGED under the retention policy at {_e(c['purged_at'])}; hash retained" if c.get("purged_at") else ""
        parts.append(f"<p class='small'>Clip: {_e(c['asset_id'])} ({_e(c['mime'])}, {c['size_bytes']} bytes, sha256 {c['sha256']}){link}{gone}</p>")
    unavailable = [a for a in assets if a["kind"] == "frame_annotated" and a["asset_id"] not in images]
    if unavailable:
        parts.append("<p class='small'>" + _e(len(unavailable)) + " annotated frame(s) are not shown: file purged under the retention policy, missing, "
                     "or failed its integrity check (see the evidence list for the registered hashes).</p>")
    parts.append("<h2>Review and audit history</h2><table><tr><th>Time (UTC)</th><th>Actor</th><th>Action</th><th>Detail</th></tr>" + "".join(
        f"<tr><td>{_e(r['ts'])}</td><td>{_e(r['actor'])}{(' (' + _e(r['actor_role']) + ')') if r.get('actor_role') else ''}</td><td>{_e(r['action'])}</td>"
        f"<td class='small'>{_e(json.dumps(r['details'])[:300])}</td></tr>" for r in reversed(audit_rows)) + "</table>")
    parts.append("<h2>Case transitions</h2><table><tr><th>Time</th><th>From</th><th>To</th><th>Action</th><th>By</th><th>Note</th></tr>" + "".join(
        f"<tr><td>{_e(t['ts'])}</td><td>{_e(t['from_state'])}</td><td>{_e(t['to_state'])}</td><td>{_e(t['action'])}</td><td>{_e(t['actor'])} ({_e(t['actor_role'])})</td><td>{_e(t['note'])}</td></tr>"
        for t in case["transitions"]) + "</table></body></html>")
    return "".join(parts)


def build_case_package(services, case_id: str, user, out_dir: Path) -> Path:
    """Zip the case: report.html, case.json, audit.json, evidence files, manifest.json."""
    case = services.cases.get(case_id, user)
    inc = services.incidents.get(case["incident_id"])
    cam = services.cameras.get(inc["camera_id"]) or {}
    assets = services.evidence.list(inc["incident_id"])
    # Every file is checked against the SHA-256 registered when it was captured. A package must never carry a file that differs
    # from the recorded evidence (the manifest below would otherwise hash the altered bytes and look consistent), so a mismatch
    # or an unexpectedly missing file stops the export. Files purged under the retention policy are skipped and listed.
    from anvesh.app.cases import CaseError

    blobs, purged, problems = {}, [], []
    for a in assets:
        data, state = services.evidence.read_verified(a)
        if state == "ok":
            blobs[a["asset_id"]] = data
        elif state == "purged":
            purged.append({"asset_id": a["asset_id"], "kind": a["kind"], "sha256": a["sha256"]})
        else:
            problems.append(f"{a['kind']} {a['asset_id']}: {state}")
    if problems:
        raise CaseError("the evidence package was not produced because evidence files failed verification", 409, problems)
    images = {a["asset_id"]: "data:image/jpeg;base64," + base64.b64encode(blobs[a["asset_id"]]).decode()
              for a in assets if a["kind"] == "frame_annotated" and a["asset_id"] in blobs}
    audit_rows = []
    for et, eid in (("case", case_id), ("incident", inc["incident_id"])):
        rows, _ = services.audit.query(entity_type=et, entity_id=eid, limit=500)
        audit_rows += rows
    audit_rows.sort(key=lambda r: r["seq"])
    report = case_report_html(case, inc, assets, images, audit_rows, cam.get("utc_offset_minutes", 330), generated_for=user.username)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{case_id}-v{case['version']}.zip"
    tmp = path.with_suffix(".tmp")
    files = {"report.html": report.encode("utf-8"),
             "case.json": json.dumps(case, indent=2, default=str).encode("utf-8"),
             "audit.json": json.dumps(audit_rows, indent=2, default=str).encode("utf-8")}
    for a in assets:
        if a["asset_id"] in blobs:
            files[f"evidence/{Path(a['relpath']).name}"] = blobs[a["asset_id"]]
    manifest = {"case_id": case_id, "case_version": case["version"], "generated_at": now_iso(), "generated_by": user.username,
                "data_class": case["data_class"], "non_production": case["data_class"] != "operational",
                "registered_evidence_sha256": {a["asset_id"]: a["sha256"] for a in assets},
                "purged_under_retention": purged,
                "notice": "Candidate detection for human review; not a challan." + (" NON-PRODUCTION TEST DATA." if case["data_class"] != "operational" else ""),
                "files": {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}}
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    tmp.replace(path)
    return path
