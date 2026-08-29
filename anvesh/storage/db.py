"""Minimal SQLite persistence (blueprint Part 4/13: `storage/db.py`).

V1 scope: persist enough to reproduce and inspect one corridor-state
decision -- the `CorridorState` itself, its per-camera segment
`TrafficState`s, and the `ReliabilityScore`/provenance that fed the
decision -- as one JSON payload per row, behind a small save/get/list
interface (`CorridorStateStore`). M4 added `RankingStore` alongside it;
M5 adds `FeedbackRecordStore` the same way -- same file, same SQLite
connection pattern, one more small table each time. NOT a second
database system, an ORM, or a repository layer: still one save/get/list
interface per concern, no migrations framework, no query builder. Swap
any store out later if SQLite's concurrency limits are actually hit
(blueprint Part 2, LATER) -- callers only depend on these classes'
methods, not on SQLite specifics.

`CorridorState`/`TrafficState`/`CandidateCauseHypothesis`/
`HypothesisUpdate`/`PropagationPrediction`/`PropagationObservation` carry
nested `Measurement`/enum values that `json` cannot serialize directly,
so this module hand-writes explicit to-dict/from-dict conversions rather
than a generic serializer -- fewer surprises than teaching `json` about
enums, and the schemas are small and fixed enough that this stays a
handful of lines per entity.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from anvesh.evidence.reliability import ReliabilityScore
from anvesh.feedback.update_engine import FeedbackResult
from anvesh.storage.schemas import (
    CandidateCauseHypothesis,
    CorridorState,
    Discrepancy,
    HypothesisUpdate,
    Measurement,
    PropagationObservation,
    PropagationPrediction,
    RankedHypothesisEntry,
    TrafficState,
)
from anvesh.world.corridor_state import CorridorStateAssembly

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS corridor_state_records (
    corridor_state_id TEXT PRIMARY KEY,
    corridor_id TEXT NOT NULL,
    window_start REAL NOT NULL,
    window_end REAL NOT NULL,
    fused_congestion_level TEXT NOT NULL,
    active_ranking_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


def _measurement_to_dict(m: Measurement) -> dict:
    return {"value": m.value, "error": m.error}


def _measurement_from_dict(d: dict) -> Measurement:
    return Measurement(value=d["value"], error=d["error"])


def _traffic_state_to_dict(ts: TrafficState) -> dict:
    return {
        "camera_id": ts.camera_id,
        "window_start": ts.window_start,
        "window_end": ts.window_end,
        "occupancy": ts.occupancy,
        "mean_speed": _measurement_to_dict(ts.mean_speed),
        "vehicle_count": ts.vehicle_count,
        "flow_rate": ts.flow_rate,
        "density": ts.density,
        "congestion_level": ts.congestion_level.value,
        "motion_space": ts.motion_space.value,
        "schema_version": ts.schema_version,
    }


def _traffic_state_from_dict(d: dict) -> TrafficState:
    return TrafficState(
        camera_id=d["camera_id"],
        window_start=d["window_start"],
        window_end=d["window_end"],
        occupancy=d["occupancy"],
        mean_speed=_measurement_from_dict(d["mean_speed"]),
        vehicle_count=d["vehicle_count"],
        flow_rate=d["flow_rate"],
        density=d["density"],
        congestion_level=d["congestion_level"],
        motion_space=d["motion_space"],
        schema_version=d.get("schema_version", "1.0.0"),
    )


def _corridor_state_to_dict(cs: CorridorState) -> dict:
    return {
        "corridor_state_id": cs.corridor_state_id,
        "window": list(cs.window),
        "segment_states": [_traffic_state_to_dict(s) for s in cs.segment_states],
        "fused_congestion_level": cs.fused_congestion_level.value,
        "active_ranking_id": cs.active_ranking_id,
        "schema_version": cs.schema_version,
    }


def _corridor_state_from_dict(d: dict) -> CorridorState:
    return CorridorState(
        corridor_state_id=d["corridor_state_id"],
        window=tuple(d["window"]),
        segment_states=[_traffic_state_from_dict(s) for s in d["segment_states"]],
        fused_congestion_level=d["fused_congestion_level"],
        active_ranking_id=d["active_ranking_id"],
        schema_version=d.get("schema_version", "1.0.0"),
    )


def _reliability_to_dict(r: ReliabilityScore) -> dict:
    return {
        "camera_id": r.camera_id,
        "window_start": r.window_start,
        "window_end": r.window_end,
        "score": r.score,
        "factors": dict(r.factors),
        "method": r.method,
    }


def _reliability_from_dict(d: dict) -> ReliabilityScore:
    return ReliabilityScore(
        camera_id=d["camera_id"],
        window_start=d["window_start"],
        window_end=d["window_end"],
        score=d["score"],
        factors=d["factors"],
        method=d.get("method", "weighted_heuristic_v1"),
    )


class CorridorStateStore:
    """Save/get/list interface over one SQLite table of CorridorStateAssembly records."""

    def __init__(self, db_path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute(_SCHEMA_SQL)
        self._conn.commit()

    def save(self, corridor_id: str, assembly: CorridorStateAssembly) -> None:
        payload = {
            "corridor_state": _corridor_state_to_dict(assembly.corridor_state),
            "reliability_by_camera": {
                cid: _reliability_to_dict(score) for cid, score in assembly.reliability_by_camera.items()
            },
            "cameras_with_data": list(assembly.cameras_with_data),
            "cameras_missing": list(assembly.cameras_missing),
        }
        cs = assembly.corridor_state
        self._conn.execute(
            """
            INSERT INTO corridor_state_records
                (corridor_state_id, corridor_id, window_start, window_end,
                 fused_congestion_level, active_ranking_id, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(corridor_state_id) DO UPDATE SET
                corridor_id=excluded.corridor_id,
                window_start=excluded.window_start,
                window_end=excluded.window_end,
                fused_congestion_level=excluded.fused_congestion_level,
                active_ranking_id=excluded.active_ranking_id,
                payload_json=excluded.payload_json,
                created_at=excluded.created_at
            """,
            (
                cs.corridor_state_id,
                corridor_id,
                cs.window[0],
                cs.window[1],
                cs.fused_congestion_level.value,
                cs.active_ranking_id,
                json.dumps(payload),
                time.time(),
            ),
        )
        self._conn.commit()

    def get(self, corridor_state_id: str):
        row = self._conn.execute(
            "SELECT payload_json FROM corridor_state_records WHERE corridor_state_id = ?",
            (corridor_state_id,),
        ).fetchone()
        if row is None:
            return None
        return self._deserialize(row[0])

    def list_for_corridor(self, corridor_id: str) -> list:
        rows = self._conn.execute(
            "SELECT payload_json FROM corridor_state_records WHERE corridor_id = ? ORDER BY window_start ASC",
            (corridor_id,),
        ).fetchall()
        return [self._deserialize(row[0]) for row in rows]

    def _deserialize(self, payload_json: str) -> CorridorStateAssembly:
        payload = json.loads(payload_json)
        return CorridorStateAssembly(
            corridor_state=_corridor_state_from_dict(payload["corridor_state"]),
            reliability_by_camera={
                cid: _reliability_from_dict(r) for cid, r in payload["reliability_by_camera"].items()
            },
            cameras_with_data=tuple(payload["cameras_with_data"]),
            cameras_missing=tuple(payload["cameras_missing"]),
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CorridorStateStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# M4: CandidateCauseHypothesis (ranking) persistence
# ---------------------------------------------------------------------------

_RANKING_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ranking_records (
    ranking_id TEXT PRIMARY KEY,
    corridor_state_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


def _ranked_entry_to_dict(entry: RankedHypothesisEntry) -> dict:
    return {
        "hypothesis_id": entry.hypothesis_id,
        "belief": entry.belief,
        "plausibility": entry.plausibility,
        "confidence_tier": entry.confidence_tier.value,
        "supporting_evidence_refs": list(entry.supporting_evidence_refs),
        "contradicting_evidence_refs": list(entry.contradicting_evidence_refs),
    }


def _ranked_entry_from_dict(d: dict) -> RankedHypothesisEntry:
    return RankedHypothesisEntry(
        hypothesis_id=d["hypothesis_id"],
        belief=d["belief"],
        plausibility=d["plausibility"],
        confidence_tier=d["confidence_tier"],
        supporting_evidence_refs=d["supporting_evidence_refs"],
        contradicting_evidence_refs=d["contradicting_evidence_refs"],
    )


def _ranking_to_dict(ranking: CandidateCauseHypothesis) -> dict:
    return {
        "ranking_id": ranking.ranking_id,
        "corridor_state_id": ranking.corridor_state_id,
        "outcome": ranking.outcome.value,
        "ranked_list": [_ranked_entry_to_dict(e) for e in ranking.ranked_list],
        "engine_model_id": ranking.engine_model_id,
        "engine_model_version": ranking.engine_model_version,
        "schema_version": ranking.schema_version,
    }


def _ranking_from_dict(d: dict) -> CandidateCauseHypothesis:
    return CandidateCauseHypothesis(
        ranking_id=d["ranking_id"],
        corridor_state_id=d["corridor_state_id"],
        outcome=d["outcome"],
        ranked_list=[_ranked_entry_from_dict(e) for e in d["ranked_list"]],
        engine_model_id=d["engine_model_id"],
        engine_model_version=d["engine_model_version"],
        schema_version=d.get("schema_version", "1.0.0"),
    )


class RankingStore:
    """Save/get/list interface over one SQLite table of CandidateCauseHypothesis records.

    Same file, same connection pattern as `CorridorStateStore` -- a second
    small table in the one V1 SQLite database, not a second database
    system.
    """

    def __init__(self, db_path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute(_RANKING_SCHEMA_SQL)
        self._conn.commit()

    def save(self, ranking: CandidateCauseHypothesis) -> None:
        self._conn.execute(
            """
            INSERT INTO ranking_records (ranking_id, corridor_state_id, outcome, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(ranking_id) DO UPDATE SET
                corridor_state_id=excluded.corridor_state_id,
                outcome=excluded.outcome,
                payload_json=excluded.payload_json,
                created_at=excluded.created_at
            """,
            (
                ranking.ranking_id,
                ranking.corridor_state_id,
                ranking.outcome.value,
                json.dumps(_ranking_to_dict(ranking)),
                time.time(),
            ),
        )
        self._conn.commit()

    def get(self, ranking_id: str):
        row = self._conn.execute(
            "SELECT payload_json FROM ranking_records WHERE ranking_id = ?",
            (ranking_id,),
        ).fetchone()
        if row is None:
            return None
        return _ranking_from_dict(json.loads(row[0]))

    def list_for_corridor_state(self, corridor_state_id: str) -> list:
        rows = self._conn.execute(
            "SELECT payload_json FROM ranking_records WHERE corridor_state_id = ? ORDER BY created_at ASC",
            (corridor_state_id,),
        ).fetchall()
        return [_ranking_from_dict(json.loads(row[0])) for row in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RankingStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# M5: FeedbackResult (prediction + observation + update + revised ranking) persistence
# ---------------------------------------------------------------------------

_FEEDBACK_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS feedback_records (
    update_id TEXT PRIMARY KEY,
    prior_ranking_id TEXT NOT NULL,
    revised_ranking_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


def _propagation_prediction_to_dict(p: PropagationPrediction) -> dict:
    return {
        "prediction_id": p.prediction_id,
        "based_on_ranking_id": p.based_on_ranking_id,
        "predicted_shockwave_speed": _measurement_to_dict(p.predicted_shockwave_speed),
        "predicted_arrival_camera": p.predicted_arrival_camera,
        "predicted_arrival_time": p.predicted_arrival_time,
        "predicted_queue_growth_rate": p.predicted_queue_growth_rate,
        "method": p.method,
        "schema_version": p.schema_version,
    }


def _propagation_prediction_from_dict(d: dict) -> PropagationPrediction:
    return PropagationPrediction(
        prediction_id=d["prediction_id"],
        based_on_ranking_id=d["based_on_ranking_id"],
        predicted_shockwave_speed=_measurement_from_dict(d["predicted_shockwave_speed"]),
        predicted_arrival_camera=d["predicted_arrival_camera"],
        predicted_arrival_time=d["predicted_arrival_time"],
        predicted_queue_growth_rate=d["predicted_queue_growth_rate"],
        method=d["method"],
        schema_version=d.get("schema_version", "1.0.0"),
    )


def _propagation_observation_to_dict(o: PropagationObservation) -> dict:
    return {
        "observation_id": o.observation_id,
        "prediction_id": o.prediction_id,
        "actual_arrival_camera": o.actual_arrival_camera,
        "actual_arrival_time": o.actual_arrival_time,
        "actual_queue_growth_rate": o.actual_queue_growth_rate,
        "data_completeness": o.data_completeness.value,
        "schema_version": o.schema_version,
    }


def _propagation_observation_from_dict(d: dict) -> PropagationObservation:
    return PropagationObservation(
        observation_id=d["observation_id"],
        prediction_id=d["prediction_id"],
        actual_arrival_camera=d["actual_arrival_camera"],
        actual_arrival_time=d["actual_arrival_time"],
        actual_queue_growth_rate=d["actual_queue_growth_rate"],
        data_completeness=d["data_completeness"],
        schema_version=d.get("schema_version", "1.0.0"),
    )


def _discrepancy_to_dict(d: Discrepancy) -> dict:
    return {"location_error": d.location_error, "time_error": d.time_error, "rate_error": d.rate_error}


def _discrepancy_from_dict(d: dict) -> Discrepancy:
    return Discrepancy(location_error=d["location_error"], time_error=d["time_error"], rate_error=d["rate_error"])


def _hypothesis_update_to_dict(u: HypothesisUpdate) -> dict:
    return {
        "update_id": u.update_id,
        "prior_ranking_id": u.prior_ranking_id,
        "propagation_observation_id": u.propagation_observation_id,
        "outcome": u.outcome.value,
        "revised_ranking_id": u.revised_ranking_id,
        "discrepancy": _discrepancy_to_dict(u.discrepancy) if u.discrepancy is not None else None,
        "schema_version": u.schema_version,
    }


def _hypothesis_update_from_dict(d: dict) -> HypothesisUpdate:
    return HypothesisUpdate(
        update_id=d["update_id"],
        prior_ranking_id=d["prior_ranking_id"],
        propagation_observation_id=d["propagation_observation_id"],
        outcome=d["outcome"],
        revised_ranking_id=d["revised_ranking_id"],
        discrepancy=_discrepancy_from_dict(d["discrepancy"]) if d["discrepancy"] is not None else None,
        schema_version=d.get("schema_version", "1.0.0"),
    )


class FeedbackRecordStore:
    """Save/get/list interface over one SQLite table of `FeedbackResult`
    records -- enough to reconstruct the prior ranking, the propagation
    prediction, the observed propagation evidence, the feedback outcome,
    and the revised ranking for any single feedback event.

    Same file, same connection pattern as `CorridorStateStore`/
    `RankingStore` -- one more small table, not a second database system.
    """

    def __init__(self, db_path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute(_FEEDBACK_SCHEMA_SQL)
        self._conn.commit()

    def save(self, result: FeedbackResult) -> None:
        payload = {
            "prior_ranking": _ranking_to_dict(result.prior_ranking),
            "prediction": _propagation_prediction_to_dict(result.prediction),
            "observation": _propagation_observation_to_dict(result.observation),
            "update": _hypothesis_update_to_dict(result.update),
            "revised_ranking": _ranking_to_dict(result.revised_ranking),
            "reason": result.reason,
        }
        self._conn.execute(
            """
            INSERT INTO feedback_records
                (update_id, prior_ranking_id, revised_ranking_id, outcome, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(update_id) DO UPDATE SET
                prior_ranking_id=excluded.prior_ranking_id,
                revised_ranking_id=excluded.revised_ranking_id,
                outcome=excluded.outcome,
                payload_json=excluded.payload_json,
                created_at=excluded.created_at
            """,
            (
                result.update.update_id,
                result.update.prior_ranking_id,
                result.update.revised_ranking_id,
                result.update.outcome.value,
                json.dumps(payload),
                time.time(),
            ),
        )
        self._conn.commit()

    def get(self, update_id: str) -> FeedbackResult:
        row = self._conn.execute(
            "SELECT payload_json FROM feedback_records WHERE update_id = ?",
            (update_id,),
        ).fetchone()
        if row is None:
            return None
        return self._deserialize(row[0])

    def list_for_prior_ranking(self, prior_ranking_id: str) -> list:
        rows = self._conn.execute(
            "SELECT payload_json FROM feedback_records WHERE prior_ranking_id = ? ORDER BY created_at ASC",
            (prior_ranking_id,),
        ).fetchall()
        return [self._deserialize(row[0]) for row in rows]

    def _deserialize(self, payload_json: str) -> FeedbackResult:
        payload = json.loads(payload_json)
        return FeedbackResult(
            prior_ranking=_ranking_from_dict(payload["prior_ranking"]),
            prediction=_propagation_prediction_from_dict(payload["prediction"]),
            observation=_propagation_observation_from_dict(payload["observation"]),
            update=_hypothesis_update_from_dict(payload["update"]),
            revised_ranking=_ranking_from_dict(payload["revised_ranking"]),
            reason=payload["reason"],
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "FeedbackRecordStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
