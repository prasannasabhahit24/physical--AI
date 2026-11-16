"""A small physics-light simulator so the whole pipeline runs and can be tested without hardware.

World:      tabletop with coloured cubes, dark trays and four magenta calibration dots.
Camera:     virtual top-down camera (slightly rotated, so calibration really matters),
            with pixel noise and brightness variation.
Robot:      SimulatedRobot implements RobotArm using the same kinematics as the real arm.
            It models jaw-width tolerance (a mis-located grasp fails), random grasp slips,
            servo position noise and sweeping collisions with other blocks.
"""
import math
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from . import config
from .kinematics import Joints, forward, solve_ik
from .robot_arm import EmergencyStopped, RobotArm, RobotError

BGR = {"red": (30, 30, 210), "blue": (200, 80, 20), "green": (60, 170, 50), "yellow": (30, 210, 230)}
TABLE_BGR = (196, 203, 210)
TRAY_BGR = (40, 40, 40)
MARKER_BGR = (200, 0, 200)

# True camera model (unknown to the software, which must recover it by calibration)
CAM_SCALE = 2.0                 # px per mm
CAM_PHI = math.radians(2.5)     # slight rotation of the camera about the vertical axis
CAM_CENTER_ROBOT = (145.0, 0.0)
CAM_CENTER_PX = (320.0, 240.0)


@dataclass
class SimBlock:
    color: str
    x: float
    y: float
    size: float
    yaw: float = 0.0


@dataclass
class SimTray:
    side: str
    x: float
    y: float
    size: float = 90.0


class World:
    def __init__(self, blocks: List[SimBlock], trays: List[SimTray], rng: Optional[np.random.Generator] = None,
                 noise_sigma: float = 2.0, brightness: float = 1.0):
        self.blocks, self.trays = blocks, trays
        self.rng = rng or np.random.default_rng(0)
        self.noise_sigma, self.brightness = noise_sigma, brightness
        self.held: Optional[SimBlock] = None
        self.events: List[str] = []
        self.collisions = 0

    # ---- camera model
    @staticmethod
    def to_pixel(x: float, y: float):
        c, s = math.cos(CAM_PHI), math.sin(CAM_PHI)
        dx = x - CAM_CENTER_ROBOT[0]
        return (CAM_CENTER_PX[0] + CAM_SCALE * (-s * dx - c * y),
                CAM_CENTER_PX[1] + CAM_SCALE * (-c * dx + s * y))

    def _poly(self, cx, cy, size, yaw_deg=0.0):
        a = math.radians(yaw_deg)
        h = size / 2
        pts = []
        for sx, sy in ((-h, -h), (h, -h), (h, h), (-h, h)):
            rx, ry = sx * math.cos(a) - sy * math.sin(a), sx * math.sin(a) + sy * math.cos(a)
            pts.append(self.to_pixel(cx + rx, cy + ry))
        return np.round(np.array(pts)).astype(np.int32)

    def render(self) -> np.ndarray:
        w, h = config.IMAGE_SIZE
        img = np.empty((h, w, 3), np.uint8)
        img[:] = TABLE_BGR
        for mx, my in config.CAL_MARKERS_ROBOT:
            u, v = self.to_pixel(mx, my)
            cv2.circle(img, (int(round(u)), int(round(v))), 14, MARKER_BGR, -1)
        for t in self.trays:
            cv2.fillConvexPoly(img, self._poly(t.x, t.y, t.size), TRAY_BGR)
        for b in sorted(self.blocks, key=lambda b: b is self.held):
            cv2.fillConvexPoly(img, self._poly(b.x, b.y, b.size, b.yaw), BGR[b.color])
        out = img.astype(np.float32) * self.brightness
        if self.noise_sigma > 0:
            out += self.rng.normal(0, self.noise_sigma, out.shape)
        return np.clip(out, 0, 255).astype(np.uint8)

    # ---- ground truth helpers (used by tests / evaluation only)
    def block(self, color: str) -> SimBlock:
        return next(b for b in self.blocks if b.color == color)

    def tray(self, side: str) -> SimTray:
        return next(t for t in self.trays if t.side == side)

    @staticmethod
    def inside(b: SimBlock, t: SimTray) -> bool:
        return abs(b.x - t.x) <= t.size / 2 - b.size / 2 + 3 and abs(b.y - t.y) <= t.size / 2 - b.size / 2 + 3


class SimulatedRobot(RobotArm):
    def __init__(self, world: World, grasp_fail_prob: float = 0.0, joint_noise_deg: float = 0.25,
                 rng: Optional[np.random.Generator] = None):
        self.world = world
        self.grasp_fail_prob = grasp_fail_prob
        self.joint_noise = joint_noise_deg
        self.rng = rng or np.random.default_rng(1)
        self.joints: Joints = solve_ik(*config.HOME_XYZ)
        self.closed = False
        self.sim_time = 0.0
        self._stopped = False
        self.min_z_seen = 1e9

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...

    # ---- motion
    def move_joints(self, joints: Joints, speed: float) -> None:
        if self._stopped:
            raise EmergencyStopped("emergency stop is active")
        for name, (lo, hi) in config.JOINT_LIMITS.items():
            if not lo - 1e-6 <= joints[name] <= hi + 1e-6:
                raise RobotError(f"joint {name}={joints[name]:.1f} outside limits")
        delta = max(abs(joints[n] - self.joints[n]) for n in config.JOINT_NAMES)
        self.sim_time += delta / (config.MAX_JOINT_SPEED_DPS * max(speed, 0.01))
        prev = forward(self.joints)
        self.joints = {n: joints[n] + float(self.rng.normal(0, self.joint_noise)) for n in config.JOINT_NAMES}
        tool = forward(self.joints)
        self.min_z_seen = min(self.min_z_seen, tool[2])
        self._check_sweep(prev, tool)
        if self.world.held is not None:
            self.world.held.x, self.world.held.y = tool[0], tool[1]

    def _check_sweep(self, prev, tool) -> None:
        """Count a collision if the arm moves sideways through another block."""
        if math.hypot(tool[0] - prev[0], tool[1] - prev[1]) < 5.0:
            return                                        # (nearly) vertical move: servo noise, not a sweep
        held = self.world.held
        reach = (held.size / 2) if held else 10.0
        underside = min(prev[2], tool[2]) - (held.size / 2 if held else 0.0)
        for b in self.world.blocks:
            if b is held:
                continue
            if max(abs(prev[0] - b.x), abs(prev[1] - b.y)) < b.size / 2:
                continue                                  # already straddling this block (grasp / retreat)
            if underside < b.size and max(abs(tool[0] - b.x), abs(tool[1] - b.y)) < b.size / 2 + reach:
                self.world.collisions += 1
                self.world.events.append(f"collision with {b.color} block")
                return

    def get_joints(self) -> Joints:
        return dict(self.joints)

    # ---- gripper
    def set_gripper(self, closed: bool) -> None:
        if self._stopped:
            raise EmergencyStopped("emergency stop is active")
        self.sim_time += config.GRIPPER_SETTLE_S
        world, tool = self.world, forward(self.joints)
        if closed and not self.closed and world.held is None:
            best, best_d = None, 1e9
            for b in world.blocks:
                tol = max(1.0, (config.GRIPPER_OPEN_MM - b.size) / 2)
                d = max(abs(tool[0] - b.x), abs(tool[1] - b.y))
                if d <= tol and tool[2] <= b.size and b.size <= config.MAX_GRASP_MM and d < best_d:
                    best, best_d = b, d
            if best is not None and self.rng.random() >= self.grasp_fail_prob:
                world.held = best
            elif best is not None:
                world.events.append("grasp slipped")
        elif not closed and world.held is not None:
            b = world.held
            surface = config.TRAY_FLOOR_Z if any(World.inside(b, t) for t in world.trays) else 0.0
            drop_height = tool[2] - b.size / 2 - surface
            if drop_height > 20.0:
                world.events.append("object dropped from height")
                b.x += float(self.rng.normal(0, 6.0))
                b.y += float(self.rng.normal(0, 6.0))
            b.x, b.y = (b.x if drop_height > 20.0 else tool[0]), (b.y if drop_height > 20.0 else tool[1])
            world.held = None
        self.closed = closed

    def gripper_holding(self) -> bool:
        return self.closed and self.world.held is not None

    # ---- safety
    def emergency_stop(self) -> None:
        self._stopped = True
        self.world.events.append("emergency stop")

    def reset_emergency_stop(self) -> None:
        self._stopped = False

    @property
    def is_stopped(self) -> bool:
        return self._stopped


# ------------------------------------------------------------------ scene factories
def default_trays() -> List[SimTray]:
    return [SimTray("left", 150.0, 100.0), SimTray("right", 150.0, -100.0)]


def make_default_world(seed: int = 0, noise_sigma: float = 2.0) -> World:
    """The proposal's demo: red, blue and a larger yellow block plus left/right trays."""
    blocks = [SimBlock("red", 200, 30, 32, 8), SimBlock("blue", 200, -30, 30, -12),
              SimBlock("yellow", 115, 0, 44, 4)]
    return World(blocks, default_trays(), np.random.default_rng(seed), noise_sigma)


def make_random_world(rng: np.random.Generator, n_blocks: Optional[int] = None, noise_sigma: float = 2.0) -> World:
    colors = ["red", "blue", "green", "yellow"]
    n = n_blocks or int(rng.integers(2, 5))
    chosen = list(rng.choice(colors, size=n, replace=False))
    sizes = list(rng.permutation([26.0, 32.0, 38.0, 44.0])[:n])      # distinct, so "largest" is unambiguous
    blocks: List[SimBlock] = []
    for color, size in zip(chosen, sizes):
        for _ in range(500):
            x, y = float(rng.uniform(102, 208)), float(rng.uniform(-30, 30))
            if all(max(abs(x - b.x), abs(y - b.y)) >= (size + b.size) / 2 + 24 for b in blocks):
                blocks.append(SimBlock(color, x, y, float(size), float(rng.uniform(-20, 20))))
                break
    return World(blocks, default_trays(), rng, noise_sigma, brightness=float(rng.uniform(0.92, 1.08)))
