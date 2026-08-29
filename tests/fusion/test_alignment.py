from dataclasses import dataclass

import pytest

from anvesh.fusion.alignment import AlignedObservationPair, align_observations


@dataclass
class FakeObservation:
    obs_id: str
    frame_timestamp: float


def _ids(pairs, side):
    getter = (lambda p: p.camera_a_observation) if side == "a" else (lambda p: p.camera_b_observation)
    return [getter(p).obs_id if getter(p) is not None else None for p in pairs]


def test_equal_timestamps_pair_exactly():
    a = [FakeObservation("a0", 0.0), FakeObservation("a1", 1.0), FakeObservation("a2", 2.0)]
    b = [FakeObservation("b0", 0.0), FakeObservation("b1", 1.0), FakeObservation("b2", 2.0)]

    pairs = align_observations(a, b, max_offset_seconds=0.01)

    assert len(pairs) == 3
    assert all(p.is_complete for p in pairs)
    assert all(p.time_offset_seconds == 0.0 for p in pairs)
    assert _ids(pairs, "a") == ["a0", "a1", "a2"]
    assert _ids(pairs, "b") == ["b0", "b1", "b2"]


def test_small_clock_offset_within_tolerance():
    a = [FakeObservation("a0", 0.0), FakeObservation("a1", 1.0)]
    b = [FakeObservation("b0", 0.05), FakeObservation("b1", 1.05)]

    pairs = align_observations(a, b, max_offset_seconds=0.1)

    assert all(p.is_complete for p in pairs)
    assert pairs[0].time_offset_seconds == pytest.approx(0.05)
    assert pairs[1].time_offset_seconds == pytest.approx(0.05)


def test_clock_offset_correction_recovers_exact_match():
    """Camera B's clock reads systematically 0.5s ahead; passing the known
    offset as a correction should realign them to an exact match."""
    a = [FakeObservation("a0", 0.0), FakeObservation("a1", 1.0), FakeObservation("a2", 2.0)]
    b = [FakeObservation("b0", 0.5), FakeObservation("b1", 1.5), FakeObservation("b2", 2.5)]

    uncorrected = align_observations(a, b, max_offset_seconds=0.1)
    assert not any(p.is_complete for p in uncorrected)  # 0.5s gap exceeds tolerance without correction

    corrected = align_observations(a, b, max_offset_seconds=0.01, clock_offset_seconds=-0.5)
    assert all(p.is_complete for p in corrected)
    assert all(p.time_offset_seconds == pytest.approx(0.0) for p in corrected)


def test_missing_frame_leaves_the_other_side_none():
    a = [FakeObservation("a0", 0.0), FakeObservation("a1", 1.0), FakeObservation("a2", 2.0)]
    b = [FakeObservation("b0", 0.0), FakeObservation("b2", 2.0)]  # frame at t=1 missing

    pairs = align_observations(a, b, max_offset_seconds=0.1)

    assert len(pairs) == 3
    missing_pair = next(p for p in pairs if p.reference_timestamp == 1.0)
    assert missing_pair.camera_a_observation.obs_id == "a1"
    assert missing_pair.camera_b_observation is None
    assert missing_pair.is_complete is False
    assert missing_pair.time_offset_seconds is None


def test_different_frame_rates_no_observation_is_silently_dropped():
    # camera A at 10fps, camera B at 5fps over the same 1-second window
    a = [FakeObservation(f"a{i}", i * 0.1) for i in range(10)]
    b = [FakeObservation(f"b{i}", i * 0.2) for i in range(5)]

    pairs = align_observations(a, b, max_offset_seconds=0.05)

    assert len(pairs) == 10  # every A observation appears; B's are a timestamp subset of A's
    matched = [p for p in pairs if p.is_complete]
    unmatched = [p for p in pairs if not p.is_complete]
    assert len(matched) == 5
    assert len(unmatched) == 5
    assert {p.camera_b_observation.obs_id for p in matched} == {"b0", "b1", "b2", "b3", "b4"}


def test_extra_unmatched_camera_b_observations_are_kept_not_dropped():
    a = [FakeObservation("a0", 0.0)]
    b = [FakeObservation("b0", 0.0), FakeObservation("b1", 5.0)]  # b1 has no A partner nearby

    pairs = align_observations(a, b, max_offset_seconds=0.1)

    assert len(pairs) == 2
    orphan = next(p for p in pairs if p.camera_a_observation is None)
    assert orphan.camera_b_observation.obs_id == "b1"
    assert orphan.is_complete is False


def test_input_order_does_not_matter_output_is_sorted():
    a = [FakeObservation("a2", 2.0), FakeObservation("a0", 0.0), FakeObservation("a1", 1.0)]
    b = [FakeObservation("b1", 1.0), FakeObservation("b0", 0.0), FakeObservation("b2", 2.0)]

    pairs = align_observations(a, b, max_offset_seconds=0.01)

    assert [p.reference_timestamp for p in pairs] == [0.0, 1.0, 2.0]
    assert _ids(pairs, "a") == ["a0", "a1", "a2"]


def test_negative_max_offset_rejected():
    with pytest.raises(ValueError):
        align_observations([], [], max_offset_seconds=-1.0)


def test_pair_with_no_observations_at_all_is_rejected():
    with pytest.raises(ValueError):
        AlignedObservationPair(
            camera_a_observation=None,
            camera_b_observation=None,
            reference_timestamp=0.0,
            time_offset_seconds=None,
        )


def test_end_to_end_two_camera_alignment_combined_scenario():
    """Combines equal timestamps, a missing frame, a small offset, and an
    extra unmatched observation in one run -- the full M2 alignment path."""
    a = [
        FakeObservation("a0", 0.0),
        FakeObservation("a1", 1.0),
        FakeObservation("a2", 2.0),  # will have no B partner (missing on B)
        FakeObservation("a3", 3.02),  # small clock drift vs its B partner
    ]
    b = [
        FakeObservation("b0", 0.0),
        FakeObservation("b1", 1.0),
        FakeObservation("b3", 3.0),
        FakeObservation("b_extra", 10.0),  # orphan, no A partner
    ]

    pairs = align_observations(a, b, max_offset_seconds=0.05)

    assert len(pairs) == 5  # 4 from A + 1 orphan from B
    by_ts = {p.reference_timestamp: p for p in pairs}

    assert by_ts[0.0].is_complete and by_ts[0.0].camera_b_observation.obs_id == "b0"
    assert by_ts[1.0].is_complete and by_ts[1.0].camera_b_observation.obs_id == "b1"
    assert by_ts[2.0].is_complete is False and by_ts[2.0].camera_a_observation.obs_id == "a2"
    assert by_ts[3.02].is_complete and by_ts[3.02].camera_b_observation.obs_id == "b3"
    assert by_ts[10.0].camera_a_observation is None and by_ts[10.0].camera_b_observation.obs_id == "b_extra"
