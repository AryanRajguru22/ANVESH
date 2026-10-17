"""Single-camera pipeline orchestration: wires ingestion -> tracking ->
track management -> schema conversion (+ optional visualization) together.

Deliberately thin -- each stage lives in its own, independently testable
module (ingestion.py, tracking.py, track_manager.py, motion.py,
schema_conversion.py, visualize.py). This module only sequences them; it
is not itself the entry point (see scripts/run_pipeline.py).

`calibration_profile` is optional (M2): when its geometry is VALIDATED
metric (`GeometryValidity.WORLD_VALIDATED`), tracks come out in world-space
(metres); otherwise -- including a calibrated-but-unvalidated profile -- this
camera's tracks stay image-space. See `schema_conversion.build_vehicle_track`.

STREAMING (M-A): this recorded-file batch replay is built on
`anvesh.streaming.engine.StreamingPerceptionEngine` in ARCHIVE mode (keep every
observation, never trim) -- the same detect/track/identity/store code a live
stream runs. Track ids in the result are session-scoped identities
(`"<session>-t<serial>"`), not raw tracker ids; `on_frame_tracked` and the
overlay receive identity-labelled detections.

`on_frame_tracked` (productization pass, video-demo integration): an
optional `(frame_index, timestamp, tracked_detections) -> None` callback
invoked once per frame, right after this frame's `tracked_detections`
are computed. Purely additive -- omitted, it changes nothing about this
function's existing behavior. It exists because `M1PipelineResult` does
not retain per-frame track-to-bounding-box association (only final
per-track summaries in `vehicle_tracks`, and untracked detections in
`camera_observations`); a caller that needs to render a real,
frame-synchronized bounding-box overlay (e.g. a UI export script) needs
exactly the same `tracked_detections` this loop already computes for
`track_manager.update()` and `draw_frame()` -- this hook exposes that
without duplicating this module's own orchestration loop elsewhere.

`M1PipelineResult.track_states` (single-camera lane/speed integration):
the raw `track_manager.TrackState` objects for every finished/active
track, in the same order as `vehicle_tracks` were built from them --
additive, always populated, never read by this module itself. It exists
because `vehicle_tracks` (frozen `storage.schemas.VehicleTrack` objects)
carry no `ground_point_trajectory`, which `perception.speed_estimation.
estimate_vehicle_speed` requires; a caller needing per-track approximate
speed uses this list instead of re-tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from anvesh.perception.ingestion import VideoFileFrameSource
from anvesh.perception.schema_conversion import build_vehicle_track
from anvesh.perception.visualize import VideoWriter, draw_frame
from anvesh.perception.track_manager import derive_run_id
from anvesh.streaming.engine import RUN_ID_AUTO, StreamingPerceptionEngine


@dataclass
class M1PipelineResult:
    camera_observations: list = field(default_factory=list)
    vehicle_tracks: list = field(default_factory=list)
    frame_count: int = 0
    track_states: list = field(default_factory=list)  # raw TrackState objects (see module docstring)
    engine_stats: dict = field(default_factory=dict)  # streaming-engine counters (identity/lifecycle diagnostics)


def run_m1_pipeline(
    video_path,
    camera_id: str,
    tracker,
    model_id: str,
    model_version: str,
    max_missed_frames: int = None,
    partial_after_missed_frames: int = 1,
    visualization_output_path=None,
    calibration_profile=None,
    on_frame_tracked=None,
    session_id: str = None,
    run_id: str = RUN_ID_AUTO,
    frame_source=None,
) -> M1PipelineResult:
    """Batch replay of one recorded file through the streaming engine.

    `max_missed_frames` is retained for caller/config compatibility and is
    interpreted as a MINIMUM occlusion horizon in frames at the source fps;
    the engine lifts it to the tracker's own lost-buffer horizon when that is
    longer, so this manager can never finalize a track the tracker would still
    revive (see `track_manager`'s LIFECYCLE note). `partial_after_missed_frames`
    is accepted and ignored: PARTIAL now begins as soon as a track is not
    observed in an update.
    """
    frame_source = frame_source or VideoFileFrameSource(video_path)
    fps = frame_source.fps()
    if run_id == RUN_ID_AUTO:
        # A recorded file has a stable identity, so its default run scope is
        # DETERMINISTIC (same recording -> same ids; a different recording ->
        # different ids) rather than random: batch results stay reproducible
        # while ids from different recordings of one camera can no longer collide.
        p = Path(video_path) if video_path is not None else None
        run_id = derive_run_id(camera_id, p.name, p.stat().st_size) if (p is not None and p.is_file()) else RUN_ID_AUTO
    del partial_after_missed_frames  # semantic now time-based; kept for signature compatibility

    last_identified: list = []

    def _on_frame(frame_index, timestamp, identified):
        last_identified[:] = identified
        if on_frame_tracked is not None:
            on_frame_tracked(frame_index, timestamp, identified)

    engine = StreamingPerceptionEngine(
        camera_id=camera_id,
        tracker=tracker,
        fps=fps,
        windows=None,
        calibration_profile=calibration_profile,
        session_id=session_id,
        run_id=run_id,
        min_occlusion_s=max_missed_frames / fps if max_missed_frames else 0.0,
        archive=True,
        model_id=model_id,
        model_version=model_version,
        on_frame=_on_frame,
    )
    result = M1PipelineResult()

    writer = None
    try:
        for frame in frame_source:
            engine.process_frame(frame)
            result.frame_count += 1

            if visualization_output_path is not None:
                if writer is None:
                    height, width = frame.image.shape[:2]
                    writer = VideoWriter(visualization_output_path, fps=fps, frame_size=(width, height))
                annotated = draw_frame(
                    frame.image,
                    frame.index,
                    frame.timestamp,
                    list(last_identified),
                    engine.store.manager.active_tracks,
                )
                writer.write(annotated)
    finally:
        if writer is not None:
            writer.release()

    engine.finish()
    result.camera_observations = engine.camera_observations
    result.track_states = list(engine.store.manager.all_tracks())
    result.vehicle_tracks = [
        build_vehicle_track(camera_id, state, calibration_profile=calibration_profile)
        for state in result.track_states
    ]
    result.engine_stats = engine.stats()
    return result
