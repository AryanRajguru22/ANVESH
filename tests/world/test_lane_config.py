import pytest

from anvesh.world.lane_config import (
    DEFAULT_LANE_CONFIG_DIR,
    LaneCameraConfig,
    default_lane_config_path,
    load_lane_config,
)
from anvesh.world.lane_geometry import LaneAssignmentStatus, LaneRegion, assign_lane

C006_PATH = DEFAULT_LANE_CONFIG_DIR / "s02_c006.toml"
C009_PATH = DEFAULT_LANE_CONFIG_DIR / "s02_c009.toml"


@pytest.mark.parametrize("path", [C006_PATH, C009_PATH])
def test_config_loads_successfully(path):
    config = load_lane_config(path)
    assert isinstance(config, LaneCameraConfig)
    assert len(config.regions) >= 1


def test_c006_expected_lane_ids():
    config = load_lane_config(C006_PATH)
    assert config.camera_id == "s02_c006"
    # Commit 9A.1: c006 now models two real near-side lanes, L1 and the
    # PROVISIONAL L2 (see configs/lanes/s02_c006.toml REVISION HISTORY).
    assert [r.lane_id for r in config.regions] == ["L1", "L2"]


def test_c009_expected_lane_ids():
    config = load_lane_config(C009_PATH)
    assert config.camera_id == "s02_c009"
    assert [r.lane_id for r in config.regions] == ["L1"]


@pytest.mark.parametrize("path", [C006_PATH, C009_PATH])
def test_regions_are_valid_lane_region_instances(path):
    """Polygon validity is enforced by `LaneRegion.__post_init__` itself
    (>=3 vertices, finite coordinates, non-degenerate area) -- loading
    without error is itself proof each configured polygon passed those
    checks."""
    config = load_lane_config(path)
    for region in config.regions:
        assert isinstance(region, LaneRegion)
        assert len(region.vertices) >= 3


@pytest.mark.parametrize("path", [C006_PATH, C009_PATH])
def test_vertices_are_within_image_bounds(path):
    config = load_lane_config(path)
    for region in config.regions:
        for x, y in region.vertices:
            assert 0.0 <= x <= config.image_width, f"{region.lane_id} vertex x={x} outside [0, {config.image_width}]"
            assert 0.0 <= y <= config.image_height, f"{region.lane_id} vertex y={y} outside [0, {config.image_height}]"


@pytest.mark.parametrize("path", [C006_PATH, C009_PATH])
def test_image_resolution_matches_documented_source(path):
    config = load_lane_config(path)
    assert (config.image_width, config.image_height) == (1920, 1080)


@pytest.mark.parametrize("path", [C006_PATH, C009_PATH])
def test_repeated_loading_is_deterministic(path):
    first = load_lane_config(path)
    for _ in range(5):
        again = load_lane_config(path)
        assert again.camera_id == first.camera_id
        assert again.image_width == first.image_width
        assert again.image_height == first.image_height
        assert [r.lane_id for r in again.regions] == [r.lane_id for r in first.regions]
        assert [r.vertices for r in again.regions] == [r.vertices for r in first.regions]


def test_default_lane_config_path_matches_convention():
    assert default_lane_config_path("s02_c006") == C006_PATH
    assert default_lane_config_path("s02_c009") == C009_PATH


def test_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_lane_config(DEFAULT_LANE_CONFIG_DIR / "does_not_exist.toml")


def test_duplicate_lane_id_in_config_raises_value_error(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(
        """
[camera]
camera_id = "bad_cam"
source_video = "x.avi"
image_width = 100
image_height = 100

[[lane]]
lane_id = "L1"
vertices = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]

[[lane]]
lane_id = "L1"
vertices = [[20.0, 0.0], [30.0, 0.0], [30.0, 10.0], [20.0, 10.0]]
"""
    )
    with pytest.raises(ValueError):
        load_lane_config(bad)


def test_invalid_polygon_in_config_raises_value_error(tmp_path):
    bad = tmp_path / "bad_polygon.toml"
    bad.write_text(
        """
[camera]
camera_id = "bad_cam"
source_video = "x.avi"
image_width = 100
image_height = 100

[[lane]]
lane_id = "L1"
vertices = [[0.0, 0.0], [10.0, 0.0]]
"""
    )
    with pytest.raises(ValueError):
        load_lane_config(bad)


def test_loaded_regions_are_usable_with_assign_lane():
    """Confirms the loaded config plugs directly into Commit 2's
    `assign_lane` -- still no wiring into the pipeline, just proof the two
    commits compose."""
    config = load_lane_config(C006_PATH)
    interior_point = (705.0, 300.0)  # inside L1's polygon by construction (Commit 9A.1 geometry)
    result = assign_lane(interior_point, config.regions)
    assert result.status == LaneAssignmentStatus.ASSIGNED
    assert result.lane_id == "L1"

    far_outside_point = (10.0, 1000.0)
    result_outside = assign_lane(far_outside_point, config.regions)
    assert result_outside.status == LaneAssignmentStatus.UNKNOWN_OUTSIDE_REGIONS
