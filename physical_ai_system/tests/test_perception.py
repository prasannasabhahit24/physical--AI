import math

import numpy as np
import pytest

from physical_ai.calibration import CalibrationError, calibrate_from_markers
from physical_ai.perception import detect_scene
from physical_ai.simulator import World, make_default_world


def test_calibration_recovers_camera_model():
    w = make_default_world(seed=3)
    calib = calibrate_from_markers(w.render())
    for x, y in [(100, 20), (150, -60), (200, 100), (70, -130)]:
        u, v = w.to_pixel(x, y)
        rx, ry = calib.pixel_to_robot(u, v)
        assert math.hypot(rx - x, ry - y) < 1.0


def test_calibration_fails_without_markers():
    img = np.full((480, 640, 3), 200, np.uint8)
    with pytest.raises(CalibrationError):
        calibrate_from_markers(img)


def test_detects_blocks_trays_positions_and_sizes():
    w = make_default_world(seed=1)
    calib = calibrate_from_markers(w.render())
    scene = detect_scene(w.render(), calib)
    assert sorted(b.color for b in scene.blocks) == ["blue", "red", "yellow"]
    assert [t.name for t in scene.trays] == ["left tray", "right tray"]
    for b in scene.blocks:
        truth = w.block(b.color)
        assert math.hypot(b.xy[0] - truth.x, b.xy[1] - truth.y) < 2.5
        assert abs(b.size_mm - truth.size) < 3.0
        assert b.confidence > 0.8
    assert scene.trays[0].xy[1] > 0 > scene.trays[1].xy[1]        # left tray has positive y


def test_block_lying_in_tray_does_not_break_tray_detection():
    w = make_default_world(seed=1)
    calib = calibrate_from_markers(w.render())
    blue = w.block("blue")
    blue.x, blue.y = w.tray("right").x, w.tray("right").y
    scene = detect_scene(w.render(), calib)
    assert len(scene.trays) == 2 and len(scene.blocks) == 3
    t = scene.trays[1]
    assert abs(t.xy[0] - 150) < 3 and abs(t.xy[1] + 100) < 3


def test_absent_colour_is_not_detected():
    w = make_default_world()
    calib = calibrate_from_markers(w.render())
    assert "green" not in [b.color for b in detect_scene(w.render(), calib).blocks]
