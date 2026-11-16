import itertools

import pytest

from physical_ai import config
from physical_ai.kinematics import forward, from_servo, solve_ik, to_servo


@pytest.mark.parametrize("x,y,z", [(150, 0, 20), (120, 60, 15), (200, -80, 70), (110, 0, 83), (0, 110, 110)])
def test_ik_fk_round_trip(x, y, z):
    j = solve_ik(x, y, z)
    assert j is not None
    fx, fy, fz = forward(j)
    assert (fx, fy, fz) == pytest.approx((x, y, z), abs=1e-6)


def test_gripper_stays_vertical():
    j = solve_ik(160, 30, 20)
    assert j["shoulder"] + j["elbow"] + j["wrist"] == pytest.approx(-90.0, abs=1e-6)


def test_unreachable_returns_none():
    assert solve_ik(400, 0, 20) is None
    assert solve_ik(-50, 0, 20) is None


def test_whole_workspace_is_reachable_at_grasp_and_transit_heights():
    w = config.WORKSPACE
    bad = []
    for x, y in itertools.product(range(int(w["x_min"]), int(w["x_max"]) + 1, 10),
                                  range(int(w["y_min"]), int(w["y_max"]) + 1, 10)):
        import math
        if not config.REACH_MIN <= math.hypot(x, y) <= config.REACH_MAX:
            continue
        for z in (13, 85):
            if solve_ik(x, y, z) is None:
                bad.append((x, y, z))
    assert not bad, f"unreachable inside the declared workspace: {bad[:5]}"


def test_servo_round_trip_and_range():
    j = solve_ik(150, 40, 20)
    s = to_servo(j)
    assert all(0 <= v <= 180 for v in s.values())
    back = from_servo(s)
    assert all(back[k] == pytest.approx(j[k], abs=1e-6) for k in j)
