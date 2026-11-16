"""The full Physical AI loop: language -> perception -> reasoning -> motion -> feedback."""
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from . import config
from .calibration import Calibration
from .camera import Camera
from .command_parser import ParsedCommand, parse_command
from .executor import Executor
from .grounding import GroundingResult, Task, ground
from .perception import Scene, detect_scene
from .planner import make_plan
from .robot_arm import RobotArm
from .safety import check_task


@dataclass
class CommandResult:
    status: str            # success | clarify | not_found | unsafe | needs_confirmation | failed | stopped | unknown
    message: str
    parsed: Optional[ParsedCommand] = None
    task: Optional[Task] = None
    elapsed_s: float = 0.0           # wall time + simulated robot time
    grasp_attempts: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "success"


def verify_placement(task: Task, scene_after: Scene) -> bool:
    """Camera check: is a block of the right colour where we meant to put it?"""
    same = [b for b in scene_after.blocks if b.color == task.target.color]
    if not same:
        return False
    nearest = min(same, key=lambda b: math.dist(b.xy, task.dest_xy))
    if math.dist(nearest.xy, task.dest_xy) > config.MATCH_TOLERANCE_MM:
        return False
    if task.dest_tray is not None:
        t = task.dest_tray
        return max(abs(nearest.xy[0] - t.xy[0]), abs(nearest.xy[1] - t.xy[1])) <= t.half
    return True


class PhysicalAISystem:
    def __init__(self, camera: Camera, robot: RobotArm, calibration: Calibration,
                 parser: Callable[[str], ParsedCommand] = parse_command,
                 confirm: Optional[Callable[[str], bool]] = None,
                 speed: float = config.SPEED_SCALE):
        self.camera, self.robot, self.calibration = camera, robot, calibration
        self.parser, self.confirm = parser, confirm
        self.estop = threading.Event()
        self.executor = Executor(robot, speed, self.estop)
        self.last_scene: Optional[Scene] = None

    def perceive(self) -> Scene:
        self.last_scene = detect_scene(self.camera.capture(), self.calibration)
        return self.last_scene

    def emergency_stop(self) -> None:
        self.estop.set()
        self.robot.emergency_stop()

    def reset_emergency_stop(self) -> None:
        self.robot.reset_emergency_stop()
        self.estop.clear()

    def handle(self, text: str) -> CommandResult:
        t0 = time.perf_counter()
        sim0 = getattr(self.robot, "sim_time", 0.0)

        def done(status, message, **kw) -> CommandResult:
            elapsed = time.perf_counter() - t0 + (getattr(self.robot, "sim_time", 0.0) - sim0)
            return CommandResult(status, message, parsed=cmd, elapsed_s=elapsed, **kw)

        cmd = self.parser(text)
        if cmd.action == "stop":
            self.emergency_stop()
            return done("stopped", "Emergency stop. Motor power is disabled until you reset it.")
        if self.estop.is_set() or self.robot.is_stopped:
            return done("stopped", "The emergency stop is active. Reset it before giving new commands.")
        if cmd.action == "unknown":
            return done("unknown", "I didn't understand that. Try: 'Put the blue block in the right tray.'")
        if cmd.obj is None:
            return done("clarify", "Which object should I move?")

        scene = self.perceive()
        g: GroundingResult = ground(cmd, scene)
        if g.status != "ok":
            return done(g.status, g.message)
        task = g.task
        plan = make_plan(task, scene)

        report = check_task(task, plan, scene)
        if not report.ok:
            return done("unsafe", "I won't do that: " + " ".join(report.issues), task=task)
        if report.needs_confirmation and not (self.confirm and self.confirm(report.confirmation_reason)):
            return done("needs_confirmation", report.confirmation_reason, task=task)

        result = self.executor.execute(plan)
        if result.reason == "emergency_stop":
            return done("stopped", "Stopped by emergency stop during the move.", task=task,
                        grasp_attempts=result.grasp_attempts)
        if not result.ok:
            return done("failed", f"I could not complete the move ({result.reason}).", task=task,
                        grasp_attempts=result.grasp_attempts)

        after = self.perceive()
        if verify_placement(task, after):
            return done("success", f"Done. The {task.target.name} is now in {task.dest_label}.", task=task,
                        grasp_attempts=result.grasp_attempts)
        return done("failed", f"I moved the {task.target.name}, but the camera does not show it in "
                              f"{task.dest_label}.", task=task, grasp_attempts=result.grasp_attempts)
