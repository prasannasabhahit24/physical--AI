"""Executor: runs a PickPlacePlan on the robot, with grasp verification and retries.

Sequence: open -> above pick -> descend -> close -> verify grip (retry) -> lift ->
carry at safe height -> descend -> open -> lift -> park.
"""
import math
import threading
from dataclasses import dataclass
from typing import Optional

from . import config
from .kinematics import Joints, forward, solve_ik
from .planner import PickPlacePlan
from .robot_arm import EmergencyStopped, RobotArm, RobotError


class PlanError(RuntimeError):
    pass


@dataclass
class ExecResult:
    ok: bool
    reason: str = ""
    grasp_attempts: int = 0


class Executor:
    def __init__(self, robot: RobotArm, speed: float = config.SPEED_SCALE,
                 estop: Optional[threading.Event] = None):
        self.robot = robot
        self.speed = speed
        self.estop = estop or threading.Event()

    # ---- low-level moves
    def _check_stop(self) -> None:
        if self.estop.is_set() or self.robot.is_stopped:
            raise EmergencyStopped("emergency stop")

    def _tool(self):
        return forward(self.robot.get_joints())

    def _line(self, x: float, y: float, z: float) -> None:
        """Straight-line (Cartesian) move of the fingertip, solved step by step."""
        x0, y0, z0 = self._tool()
        n = max(1, math.ceil(math.dist((x0, y0, z0), (x, y, z)) / config.CARTESIAN_STEP_MM))
        for i in range(1, n + 1):
            f = i / n
            p = (x0 + (x - x0) * f, y0 + (y - y0) * f, z0 + (z - z0) * f)
            joints = solve_ik(*p)
            if joints is None:
                raise PlanError(f"pose ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) is not reachable")
            self._check_stop()
            self.robot.move_joints(joints, self.speed)

    def _joint_move(self, target: Joints, steps: int = 14) -> None:
        """Joint-space move; used for the big swings between the park pose and the work area."""
        start = self.robot.get_joints()
        for i in range(1, steps + 1):
            f = i / steps
            self._check_stop()
            self.robot.move_joints({n: start[n] + (target[n] - start[n]) * f for n in config.JOINT_NAMES}, self.speed)

    def park(self) -> None:
        target = solve_ik(*config.HOME_XYZ)
        z_now = self._tool()[2]
        if z_now < 70:                                    # never swing sideways while low
            x, y, _ = self._tool()
            self._line(x, y, 80.0)
        self._joint_move(target)

    # ---- the task
    def execute(self, plan: PickPlacePlan) -> ExecResult:
        attempts = 0
        px, py = plan.pick_xy
        qx, qy = plan.place_xy
        try:
            self.robot.set_gripper(closed=False)
            above_pick = solve_ik(px, py, plan.z_transit)
            if above_pick is None:
                raise PlanError("pick position unreachable")
            self._joint_move(above_pick)

            holding = False
            while attempts < config.MAX_GRASP_ATTEMPTS and not holding:
                attempts += 1
                self._line(px, py, plan.z_grasp)
                self.robot.set_gripper(closed=True)
                holding = self.robot.gripper_holding()
                if not holding:                           # slipped / missed: back off and retry
                    self._line(px, py, plan.z_transit)
                    self.robot.set_gripper(closed=False)
            if not holding:
                self.park()
                return ExecResult(False, "grasp_failed", attempts)

            self._line(px, py, plan.z_transit)
            if not self.robot.gripper_holding():
                self.park()
                return ExecResult(False, "object_lost_while_lifting", attempts)

            self._line(qx, qy, plan.z_transit)
            self._line(qx, qy, plan.z_place)
            self.robot.set_gripper(closed=False)
            self._line(qx, qy, plan.z_transit)
            self.park()
            return ExecResult(True, "", attempts)
        except EmergencyStopped:
            return ExecResult(False, "emergency_stop", attempts)
        except (PlanError, RobotError) as e:
            try:
                self.park()                               # never release a held object mid-air on an error
            except RobotError:
                pass
            return ExecResult(False, f"error: {e}", attempts)
