import threading

import numpy as np
import pytest

from physical_ai.calibration import calibrate_from_markers
from physical_ai.camera import SimulatedCamera
from physical_ai.simulator import SimulatedRobot, make_default_world
from physical_ai.system import PhysicalAISystem


def build(seed=0, grasp_fail_prob=0.0):
    world = make_default_world(seed=seed)
    cam = SimulatedCamera(world)
    robot = SimulatedRobot(world, grasp_fail_prob=grasp_fail_prob, rng=np.random.default_rng(seed))
    return PhysicalAISystem(cam, robot, calibrate_from_markers(cam.capture())), world, robot


def test_blue_block_to_right_tray():
    system, world, robot = build()
    r = system.handle("Put the blue block in the right tray.")
    assert r.status == "success", r.message
    assert world.inside(world.block("blue"), world.tray("right"))
    assert world.collisions == 0 and robot.min_z_seen > 5
    assert r.elapsed_s > 0


def test_red_block_to_left_tray_and_others_untouched():
    system, world, _ = build()
    before = {b.color: (b.x, b.y) for b in world.blocks}
    r = system.handle("Pick up the red block and place it in the tray on the left.")
    assert r.status == "success", r.message
    assert world.inside(world.block("red"), world.tray("left"))
    for c in ("blue", "yellow"):
        assert (world.block(c).x, world.block(c).y) == before[c]


def test_largest_object_to_centre():
    system, world, _ = build()
    r = system.handle("Move the largest object to the centre.")
    assert r.status == "success", r.message
    y = world.block("yellow")
    assert abs(y.x - 150) < 20 and abs(y.y) < 20


def test_beside():
    system, world, _ = build()
    r = system.handle("Put the blue block beside the red block.")
    assert r.status == "success", r.message
    b, red = world.block("blue"), world.block("red")
    assert max(abs(b.x - red.x), abs(b.y - red.y)) < 60       # adjacent
    assert max(abs(b.x - red.x), abs(b.y - red.y)) >= (b.size + red.size) / 2   # not overlapping


def test_absent_object_is_reported_not_guessed():
    system, world, robot = build()
    before = [(b.x, b.y) for b in world.blocks]
    r = system.handle("Pick up the green block.")
    assert r.status == "not_found" and "green" in r.message
    assert [(b.x, b.y) for b in world.blocks] == before and robot.sim_time == 0.0


def test_ambiguous_object_asks_for_clarification_without_moving():
    system, world, robot = build()
    r = system.handle("Move that object to the tray.")
    assert r.status == "clarify" and "Which" in r.message
    assert robot.sim_time == 0.0


def test_missing_destination_asks():
    system, _, robot = build()
    r = system.handle("Pick up the red block.")
    assert r.status == "clarify" and "Where" in r.message and robot.sim_time == 0.0


def test_ambiguous_tray_asks_which():
    system, _, robot = build()
    r = system.handle("Put the red block in the tray.")
    assert r.status == "clarify" and "tray" in r.message and robot.sim_time == 0.0


def test_unreachable_target_is_refused():
    system, world, robot = build()
    world.block("red").x, world.block("red").y = 245.0, 0.0        # beyond the safe area
    r = system.handle("Put the red block in the left tray.")
    assert r.status == "unsafe" and robot.sim_time == 0.0


def test_crowded_neighbour_is_refused():
    system, world, robot = build()
    world.block("blue").x, world.block("blue").y = 170.0, 5.0       # right next to the red block
    r = system.handle("Put the red block in the left tray.")
    assert r.status == "unsafe" and "too close" in r.message and robot.sim_time == 0.0


def test_grasp_retries_after_slip():
    system, world, _ = build(seed=5, grasp_fail_prob=0.5)
    ok = 0
    for seed in range(6):
        system, world, _ = build(seed=seed, grasp_fail_prob=0.5)
        r = system.handle("Put the blue block in the right tray.")
        ok += r.status == "success"
    assert ok >= 5      # P(all 3 attempts fail) = 12.5% per trial; retries should rescue most


def test_total_grasp_failure_is_reported_and_arm_parks():
    system, world, robot = build(grasp_fail_prob=1.0)
    r = system.handle("Put the blue block in the right tray.")
    assert r.status == "failed" and "grasp_failed" in r.message and r.grasp_attempts == 3
    assert not world.inside(world.block("blue"), world.tray("right"))


def test_stop_command_cuts_motion_and_blocks_further_commands():
    system, world, robot = build()
    assert system.handle("stop").status == "stopped"
    r = system.handle("Put the blue block in the right tray.")
    assert r.status == "stopped" and robot.sim_time == 0.0
    system.reset_emergency_stop()
    assert system.handle("Put the blue block in the right tray.").status == "success"


def test_estop_during_motion():
    system, world, robot = build()
    original = robot.move_joints
    calls = {"n": 0}

    def trip(joints, speed):
        calls["n"] += 1
        if calls["n"] == 20:
            system.estop.set()
        original(joints, speed)

    robot.move_joints = trip
    r = system.handle("Put the blue block in the right tray.")
    assert r.status == "stopped"
    assert calls["n"] <= 21


def test_low_confidence_needs_confirmation_and_respects_answer():
    system, world, robot = build()
    scene = system.perceive()
    from physical_ai.perception import DetectedObject
    original = system.perceive

    def dim_perceive():
        s = original()
        for b in s.blocks:
            b.confidence = 0.7
        return s

    system.perceive = dim_perceive
    r = system.handle("Put the blue block in the right tray.")
    assert r.status == "needs_confirmation" and robot.sim_time == 0.0
    system.confirm = lambda q: True
    assert system.handle("Put the blue block in the right tray.").status == "success"


def test_path_never_dips_below_table_or_sweeps_blocks():
    system, world, robot = build()
    system.handle("Put the red block in the left tray.")
    system.handle("Put the yellow block in the right tray.")
    assert world.collisions == 0
    assert robot.min_z_seen > 5
