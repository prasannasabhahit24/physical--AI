"""Forward / inverse kinematics for a 4-DOF arm with a vertical gripper.

Joints: base yaw, shoulder, elbow (all in a vertical plane after yaw) and a wrist
pitch that keeps the gripper pointing straight down. The tool point is the fingertip.
"""
import math
from typing import Dict, Optional, Tuple

from .config import GEOM, JOINT_LIMITS, JOINT_NAMES, SERVO_MAP, ArmGeometry

Joints = Dict[str, float]


def solve_ik(x: float, y: float, z: float, geom: ArmGeometry = GEOM,
             limits: Dict[str, Tuple[float, float]] = JOINT_LIMITS) -> Optional[Joints]:
    """Return joint angles (deg) that put the fingertip at (x, y, z), or None if unreachable."""
    if x < 0:
        return None
    r = math.hypot(x, y)
    base = math.degrees(math.atan2(y, x))
    wrist_z = z + geom.gripper_len - geom.shoulder_height       # relative to the shoulder
    d2 = r * r + wrist_z * wrist_z
    c = (d2 - geom.l1 ** 2 - geom.l2 ** 2) / (2 * geom.l1 * geom.l2)
    if c < -1.0 or c > 1.0:
        return None
    q2 = -math.acos(c)                                          # elbow-up solution
    q1 = math.atan2(wrist_z, r) - math.atan2(geom.l2 * math.sin(q2), geom.l1 + geom.l2 * math.cos(q2))
    q3 = -math.pi / 2 - q1 - q2                                 # gripper vertical
    joints = {"base": base, "shoulder": math.degrees(q1), "elbow": math.degrees(q2),
              "wrist": math.degrees(q3)}
    for name, (lo, hi) in limits.items():
        if not lo - 1e-6 <= joints[name] <= hi + 1e-6:
            return None
    return joints


def forward(joints: Joints, geom: ArmGeometry = GEOM) -> Tuple[float, float, float]:
    """Fingertip position (x, y, z) for the given joint angles (deg)."""
    q0 = math.radians(joints["base"])
    q1 = math.radians(joints["shoulder"])
    q2 = math.radians(joints["elbow"])
    q3 = math.radians(joints["wrist"])
    s = q1 + q2 + q3
    r = geom.l1 * math.cos(q1) + geom.l2 * math.cos(q1 + q2) + geom.gripper_len * math.cos(s)
    z = (geom.shoulder_height + geom.l1 * math.sin(q1) + geom.l2 * math.sin(q1 + q2)
         + geom.gripper_len * math.sin(s))
    return r * math.cos(q0), r * math.sin(q0), z


def is_reachable(x: float, y: float, z: float) -> bool:
    return solve_ik(x, y, z) is not None


def to_servo(joints: Joints) -> Joints:
    """Convert joint-space angles to hobby-servo angles (0-180)."""
    out = {}
    for name in JOINT_NAMES:
        offset, sign = SERVO_MAP[name]
        out[name] = max(0.0, min(180.0, offset + sign * joints[name]))
    return out


def from_servo(servo: Joints) -> Joints:
    out = {}
    for name in JOINT_NAMES:
        offset, sign = SERVO_MAP[name]
        out[name] = (servo[name] - offset) / sign
    return out
