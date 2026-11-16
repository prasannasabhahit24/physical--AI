"""Safety checks that run BEFORE any motion. The robot never moves if one of these fails.

Also see: the emergency stop (system.py / robot_arm.py), joint limits (kinematics.py),
speed scaling (config.SPEED_SCALE) and the hardware E-stop + power relay in the firmware.
"""
import math
from dataclasses import dataclass, field
from typing import List

from . import config
from .grounding import Task
from .kinematics import solve_ik
from .perception import Scene
from .planner import PickPlacePlan


@dataclass
class SafetyReport:
    ok: bool
    issues: List[str] = field(default_factory=list)
    needs_confirmation: bool = False
    confirmation_reason: str = ""


def _reachable_xy(x: float, y: float) -> bool:
    w = config.WORKSPACE
    r = math.hypot(x, y)
    return (w["x_min"] <= x <= w["x_max"] and w["y_min"] <= y <= w["y_max"]
            and config.REACH_MIN <= r <= config.REACH_MAX)


def check_task(task: Task, plan: PickPlacePlan, scene: Scene) -> SafetyReport:
    issues: List[str] = []
    t = task.target

    # 1. confidence of the detection
    if t.confidence < config.MIN_CONFIDENCE:
        issues.append(f"I'm not sure about the {t.name} (confidence {t.confidence:.0%}).")

    # 2. inside the operating area
    if not _reachable_xy(*plan.pick_xy):
        issues.append(f"The {t.name} is outside the robot's safe working area.")
    if not _reachable_xy(*plan.place_xy):
        issues.append(f"The destination ({task.dest_label}) is outside the robot's safe working area.")

    # 3. object must fit the gripper
    if t.size_mm > config.MAX_GRASP_MM:
        issues.append(f"The {t.name} is {t.size_mm:.0f} mm wide, wider than the gripper can hold ({config.MAX_GRASP_MM:.0f} mm).")
    if t.size_mm < config.MIN_GRASP_MM:
        issues.append(f"The {t.name} is too small to grasp safely ({t.size_mm:.0f} mm).")

    # 4. neighbours must leave room for the fingers
    for b in scene.blocks:
        if b is t:
            continue
        cheb = max(abs(b.xy[0] - t.xy[0]), abs(b.xy[1] - t.xy[1]))
        if cheb < (b.size_mm + t.size_mm) / 2 + config.JAW_CLEARANCE:
            issues.append(f"The {b.name} is too close to the {t.name}; the gripper could hit it.")

    # 5. every pose on the path must be reachable within joint limits
    for name, (x, y, z) in plan.key_poses():
        if solve_ik(x, y, z) is None:
            issues.append(f"The arm cannot reach the '{name}' pose at ({x:.0f}, {y:.0f}, {z:.0f}) mm.")
            break

    report = SafetyReport(ok=not issues, issues=issues)

    # 6. human confirmation for uncertain detections
    if report.ok and t.confidence < config.CONFIRM_CONFIDENCE:
        report.needs_confirmation = True
        report.confirmation_reason = (f"I'm only {t.confidence:.0%} sure about the {t.name}. "
                                      f"Move it to {task.dest_label} anyway?")
    return report
