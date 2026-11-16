"""Central configuration for the Physical AI pick-and-place system.

Units: millimetres and degrees unless stated otherwise.
Robot frame: origin at the base of the arm, +x forward (away from the base),
+y to the robot's left, z up from the table surface.
"""
from dataclasses import dataclass
from typing import Dict, Tuple


# ----------------------------------------------------------------- arm geometry
@dataclass(frozen=True)
class ArmGeometry:
    shoulder_height: float = 80.0   # shoulder axis height above the table
    l1: float = 120.0               # shoulder -> elbow
    l2: float = 120.0               # elbow -> wrist
    gripper_len: float = 60.0       # wrist pivot -> fingertip (gripper points straight down)


GEOM = ArmGeometry()

# Joint limits in joint space (degrees). Measure these on your own arm.
JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    "base": (-90.0, 90.0),
    "shoulder": (0.0, 120.0),
    "elbow": (-150.0, -10.0),
    "wrist": (-100.0, 60.0),
}
JOINT_NAMES = ("base", "shoulder", "elbow", "wrist")

# servo_angle = offset + sign * joint_angle. Tune for how the servos are mounted.
SERVO_MAP: Dict[str, Tuple[float, float]] = {
    "base": (90.0, 1.0),
    "shoulder": (0.0, 1.0),
    "elbow": (180.0, 1.0),
    "wrist": (100.0, 1.0),
}

# Gripper
GRIPPER_OPEN_DEG = 20.0
GRIPPER_CLOSED_DEG = 120.0
GRIPPER_OPEN_MM = 60.0       # jaw opening when open
MAX_GRASP_MM = 50.0          # largest object the gripper may pick
MIN_GRASP_MM = 12.0

# ----------------------------------------------------------------- workspace
WORKSPACE = dict(x_min=60.0, x_max=230.0, y_min=-150.0, y_max=150.0)
REACH_MIN = 90.0             # radial distance limits from the base axis
REACH_MAX = 222.0
CENTER_XY = (150.0, 0.0)     # "the centre" of the work surface
ZONE_Y = 100.0               # |y| for "left"/"right" zones when no tray is named
EDGE_MARGIN = 12.0           # keep placements this far inside the workspace

# Home / park pose: outside the camera's view so it never occludes the scene.
HOME_XYZ = (0.0, 110.0, 110.0)

# ----------------------------------------------------------------- motion
SPEED_SCALE = 0.35           # 0..1; keep low during early testing
MAX_JOINT_SPEED_DPS = 120.0  # at SPEED_SCALE = 1
CARTESIAN_STEP_MM = 12.0     # interpolation step for straight-line moves
TRANSIT_CLEARANCE = 60.0     # carried object's underside stays this far above the table
TRAY_FLOOR_Z = 5.0
PLACE_DROP_MM = 2.0          # release this far above the surface
MAX_GRASP_ATTEMPTS = 3
GRIPPER_SETTLE_S = 0.4

# ----------------------------------------------------------------- vision
IMAGE_SIZE = (640, 480)
MIN_BLOCK_AREA_PX = 700
MIN_TRAY_AREA_PX = 8000
MIN_CONFIDENCE = 0.60        # below this the system refuses to act on a detection
CONFIRM_CONFIDENCE = 0.80    # between MIN and this the user must confirm
HSV_RANGES = {
    # name: list of (lower, upper) HSV ranges (OpenCV H: 0-179)
    "red": [((0, 120, 100), (8, 255, 255)), ((172, 120, 100), (179, 255, 255))],
    "yellow": [((18, 120, 120), (35, 255, 255))],
    "green": [((40, 90, 70), (85, 255, 255))],
    "blue": [((100, 120, 70), (130, 255, 255))],
}
MARKER_HSV = [((140, 100, 80), (165, 255, 255))]   # magenta calibration dots
TRAY_MAX_VALUE = 80          # trays are dark: HSV value below this

# Calibration markers in robot coordinates, ordered
# [far-left, far-right, near-left, near-right] = image [top-left, top-right, bottom-left, bottom-right]
CAL_MARKERS_ROBOT = [(225.0, 135.0), (225.0, -135.0), (70.0, 135.0), (70.0, -135.0)]

# ----------------------------------------------------------------- safety
JAW_CLEARANCE = 10.0         # free space needed beside a block to grasp it
PLACE_CLEARANCE = 8.0        # gap kept between a placed block and its neighbours
MATCH_TOLERANCE_MM = 18.0    # post-move verification tolerance

# ----------------------------------------------------------------- language
LLM_MODEL = "claude-sonnet-5-5"   # used only by the optional LLM parser
