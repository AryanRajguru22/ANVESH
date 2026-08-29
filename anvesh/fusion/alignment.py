"""Temporal alignment (blueprint Part 3, module 6): synchronize two
cameras' observation streams onto a shared timeline via nearest-timestamp
matching within a tolerance window.

This is alignment only -- no cross-camera *evidence* fusion (Dempster-
Shafer combination, blueprint Part 6) happens here. The output is a
sequence of `AlignedObservationPair`s suitable for M3 (corridor-state
assembly) and M4 (fusion) to consume.

Clock-offset correction is a caller-supplied constant (blueprint Part 16's
"camera synchronization" mitigation: a one-time manual landmark-event
check at recording time), not something this module estimates on its own
-- matching M2's "manual, not automatic" scope for calibration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlignedObservationPair:
    """One aligned instant: an observation from each camera, or `None` on
    whichever side had nothing within `max_offset_seconds` of the other."""

    camera_a_observation: object
    camera_b_observation: object
    reference_timestamp: float
    time_offset_seconds: object  # float, or None if one side is missing

    def __post_init__(self) -> None:
        if self.camera_a_observation is None and self.camera_b_observation is None:
            raise ValueError("an AlignedObservationPair must have at least one observation")

    @property
    def is_complete(self) -> bool:
        return self.camera_a_observation is not None and self.camera_b_observation is not None


def _default_get_timestamp(observation) -> float:
    return observation.frame_timestamp


def align_observations(
    camera_a_observations: list,
    camera_b_observations: list,
    max_offset_seconds: float,
    clock_offset_seconds: float = 0.0,
    get_timestamp=None,
) -> list:
    """Greedy nearest-timestamp alignment of two per-camera observation streams.

    - `clock_offset_seconds` is added to camera B's timestamps before
      matching (a known/measured clock skew correction; see module
      docstring). Positive means camera B's clock reads ahead of camera A's.
    - Every observation from both streams appears in exactly one output
      pair; an observation with no partner within `max_offset_seconds`
      appears alone (its side of the pair is `None`) rather than being
      silently dropped -- this is how "missing frames" and "different
      frame rates" stay visible to M3/M4 instead of disappearing here.
    - Requires `max_offset_seconds >= 0`; there is no built-in default,
      matching the "no fake precision" rule -- the caller must pick a
      tolerance appropriate to their cameras' actual frame rates.

    `get_timestamp` defaults to reading `.frame_timestamp` (matching
    `anvesh.storage.schemas.CameraObservation`), but accepts any callable
    so this also works against plain synthetic/test objects.
    """
    if max_offset_seconds < 0:
        raise ValueError("max_offset_seconds must be >= 0")

    get_timestamp = get_timestamp or _default_get_timestamp

    a_sorted = sorted(camera_a_observations, key=get_timestamp)
    b_sorted = sorted(camera_b_observations, key=get_timestamp)
    b_timestamps = [get_timestamp(obs) + clock_offset_seconds for obs in b_sorted]

    pairs = []
    used_b_indices = set()

    for a_obs in a_sorted:
        ts_a = get_timestamp(a_obs)
        best_idx = None
        best_diff = None
        for idx, ts_b in enumerate(b_timestamps):
            if idx in used_b_indices:
                continue
            diff = abs(ts_b - ts_a)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_idx = idx

        if best_idx is not None and best_diff <= max_offset_seconds:
            used_b_indices.add(best_idx)
            pairs.append(
                AlignedObservationPair(
                    camera_a_observation=a_obs,
                    camera_b_observation=b_sorted[best_idx],
                    reference_timestamp=ts_a,
                    time_offset_seconds=best_diff,
                )
            )
        else:
            pairs.append(
                AlignedObservationPair(
                    camera_a_observation=a_obs,
                    camera_b_observation=None,
                    reference_timestamp=ts_a,
                    time_offset_seconds=None,
                )
            )

    for idx, b_obs in enumerate(b_sorted):
        if idx in used_b_indices:
            continue
        pairs.append(
            AlignedObservationPair(
                camera_a_observation=None,
                camera_b_observation=b_obs,
                reference_timestamp=get_timestamp(b_obs),  # reported in B's own clock, uncorrected
                time_offset_seconds=None,
            )
        )

    pairs.sort(key=lambda p: p.reference_timestamp)
    return pairs
