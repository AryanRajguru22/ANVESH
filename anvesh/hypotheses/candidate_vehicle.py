"""Ranking -> candidate-vehicle resolver (forensic audit §5.5/§8.3, plan §11
Commit 7).

The rest of the pipeline ranks *cause categories* (H1-H6, `hypotheses/
ranking.py`'s `CandidateCauseHypothesis`) -- it never names a vehicle.
`Evidence.supporting_track_refs` (`storage/schemas.py`) is the only place a
real `track_id` enters the evidence model at all, and only H2/H3 ever
populate it (`evidence/evidence_builder.py`'s `_evidence_h2`/`_evidence_h3`);
H1/H4/H5/H6 describe traffic/system conditions, not one vehicle, and never
carry a track reference. There is a real "hop" missing between "the
highest-ranked hypothesis" and "the vehicle its evidence names," because
`RankedHypothesisEntry.supporting_evidence_refs` stores `evidence_id`
strings, not `Evidence` objects or `track_id`s directly.

This module is that hop, and nothing else. It does not re-score evidence,
does not re-rank hypotheses, does not touch fusion or propagation, and does
not decide which vehicle is "responsible" for anything -- it only asks,
given a ranking that already exists: "does the evidence that produced the
TOP-ranked hypothesis name a specific vehicle, and if so, which one?"

Design rules (never violated, all directly testable):
  - Never hard-code, guess, or otherwise fabricate a `track_id`. Every
    `track_id` this module can ever return already appears in some
    `Evidence.supporting_track_refs` for the top-ranked hypothesis.
  - Never select a vehicle by a generic property (closest, fastest,
    stopped, first-in-list, highest ID, most visually prominent) -- the
    ONLY thing this module reads is which track_ids the top hypothesis's
    OWN supporting evidence already names.
  - Only the TOP-ranked entry (`ranked_list[0]`) is ever consulted. A
    lower-ranked hypothesis's vehicle reference is never surfaced, even if
    the top hypothesis has none -- the honest answer there is "no vehicle
    for this cause," not "here's a vehicle from a different cause."
  - `belief`/`plausibility` are carried through unchanged from the ranking
    (Dempster-Shafer quantities, never converted into a probability of
    vehicle responsibility -- see `ranking.py`'s own module docstring for
    why this project never does that conversion).
  - The result communicates "highest-ranked candidate supported by current
    evidence," never "confirmed responsible vehicle" -- hence an explicit
    status on every result, never a bare `track_id`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from anvesh.storage.schemas import CandidateCauseHypothesis, CandidateOutcome


class CandidateVehicleStatus(str, Enum):
    """Explicit outcome of resolving a ranking to a vehicle. Never collapse
    these into each other or into a boolean -- each names a genuinely
    different situation (see module docstring)."""

    RESOLVED = "resolved"  # exactly one track_id in the top hypothesis's supporting evidence
    AMBIGUOUS = "ambiguous"  # more than one, equally supported -- picking one would be a fabrication
    NO_VEHICLE_FOR_HYPOTHESIS = "no_vehicle_for_hypothesis"  # top hypothesis names no vehicle (expected for H1/H5/H6)
    NOT_RANKED = "not_ranked"  # the ranking itself abstained (HIGH_CONFLICT/INSUFFICIENT_EVIDENCE) or is empty


@dataclass(frozen=True)
class CandidateVehicleResolution:
    """Non-schema companion result -- the same pattern as `EvidenceItem`,
    `StoppedVehicleClassification`, `CorridorStateAssembly`: a derived
    diagnostic that references existing frozen schema objects rather than
    being folded into one. Never persisted/schema-validated itself."""

    status: CandidateVehicleStatus
    hypothesis_id: object  # str, or None when status == NOT_RANKED
    belief: object  # float, or None when status == NOT_RANKED (evidential quantity, unchanged from ranking)
    plausibility: object  # float, or None when status == NOT_RANKED (ditto)
    supporting_evidence_refs: tuple  # the top entry's own supporting_evidence_refs, unchanged
    candidate_track_ids: tuple  # every distinct track_id the top hypothesis's evidence names, sorted
    track_id: object  # str, only when status == RESOLVED; else None


_NOT_RANKED = CandidateVehicleResolution(
    status=CandidateVehicleStatus.NOT_RANKED,
    hypothesis_id=None,
    belief=None,
    plausibility=None,
    supporting_evidence_refs=(),
    candidate_track_ids=(),
    track_id=None,
)


def _evidence_index(evidence_by_camera: dict) -> dict:
    """evidence_id -> EvidenceItem, flattened across every camera. Mirrors
    `ranking.py`'s own flattening of `evidence_by_camera` when it built
    `supporting_evidence_refs` in the first place."""
    index = {}
    for camera_evidence in evidence_by_camera.values():
        for item in camera_evidence:
            index[item.evidence.evidence_id] = item
    return index


def resolve_candidate_vehicle(
    ranking: CandidateCauseHypothesis,
    evidence_by_camera: dict,
) -> CandidateVehicleResolution:
    """Resolve `ranking`'s top-ranked hypothesis to a vehicle, if its own
    supporting evidence names one.

    `evidence_by_camera` must be the SAME `camera_id -> tuple[EvidenceItem]`
    mapping (or a superset of it) that produced `ranking` via
    `hypotheses.ranking.build_ranking` -- this is what lets
    `supporting_evidence_refs` (evidence_id strings) be traced back to
    `Evidence.supporting_track_refs` (real track_id strings).
    """
    if ranking.outcome != CandidateOutcome.RANKED or not ranking.ranked_list:
        return _NOT_RANKED

    top = ranking.ranked_list[0]
    evidence_index = _evidence_index(evidence_by_camera)

    track_ids: list = []
    seen: set = set()
    for evidence_id in top.supporting_evidence_refs:
        item = evidence_index.get(evidence_id)
        if item is None:
            continue  # cannot resolve what isn't in the supplied evidence -- never guess
        for track_id in item.evidence.supporting_track_refs:
            if track_id not in seen:
                seen.add(track_id)
                track_ids.append(track_id)
    track_ids = tuple(sorted(track_ids))

    if not track_ids:
        status = CandidateVehicleStatus.NO_VEHICLE_FOR_HYPOTHESIS
        resolved_track_id = None
    elif len(track_ids) == 1:
        status = CandidateVehicleStatus.RESOLVED
        resolved_track_id = track_ids[0]
    else:
        status = CandidateVehicleStatus.AMBIGUOUS
        resolved_track_id = None

    return CandidateVehicleResolution(
        status=status,
        hypothesis_id=top.hypothesis_id,
        belief=top.belief,
        plausibility=top.plausibility,
        supporting_evidence_refs=tuple(top.supporting_evidence_refs),
        candidate_track_ids=track_ids,
        track_id=resolved_track_id,
    )
