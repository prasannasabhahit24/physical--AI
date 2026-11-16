"""Camera-to-robot calibration.

The camera looks straight down at the table. Four magenta dots sit at known
robot-frame positions (CAL_MARKERS_ROBOT). Detecting them gives a homography that
maps any image pixel on the table plane to robot (x, y) in millimetres.
"""
import json
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from . import config


class CalibrationError(RuntimeError):
    pass


class Calibration:
    def __init__(self, homography: np.ndarray):
        self.H = np.asarray(homography, dtype=np.float64)

    def pixel_to_robot(self, u: float, v: float) -> Tuple[float, float]:
        p = cv2.perspectiveTransform(np.array([[[u, v]]], dtype=np.float64), self.H)[0, 0]
        return float(p[0]), float(p[1])

    def mm_per_pixel(self, u: float, v: float) -> float:
        a = np.array(self.pixel_to_robot(u, v))
        b = np.array(self.pixel_to_robot(u + 1.0, v))
        c = np.array(self.pixel_to_robot(u, v + 1.0))
        return float((np.linalg.norm(b - a) + np.linalg.norm(c - a)) / 2.0)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"homography": self.H.tolist()}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Calibration":
        with open(path) as f:
            return cls(np.array(json.load(f)["homography"]))


def calibrate_from_points(pixel_pts: Sequence[Tuple[float, float]],
                          robot_pts: Sequence[Tuple[float, float]]) -> Calibration:
    """Fit the homography from >= 4 pixel / robot point pairs.
    On real hardware the robot points can be measured by jogging the gripper to each dot."""
    if len(pixel_pts) < 4 or len(pixel_pts) != len(robot_pts):
        raise CalibrationError("need at least 4 matching point pairs")
    H, _ = cv2.findHomography(np.array(pixel_pts, np.float64), np.array(robot_pts, np.float64), 0)
    if H is None:
        raise CalibrationError("could not compute homography")
    return Calibration(H)


def find_marker_pixels(image: np.ndarray) -> List[Tuple[float, float]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in config.MARKER_HSV:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in contours:
        if cv2.contourArea(c) < 40:
            continue
        m = cv2.moments(c)
        pts.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))
    return pts


def _order_top_left_first(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """[top-left, top-right, bottom-left, bottom-right] by image position."""
    by_v = sorted(pts, key=lambda p: p[1])
    top, bottom = sorted(by_v[:2]), sorted(by_v[2:])
    return [top[0], top[1], bottom[0], bottom[1]]


def calibrate_from_markers(image: np.ndarray) -> Calibration:
    pts = find_marker_pixels(image)
    if len(pts) != 4:
        raise CalibrationError(f"expected 4 calibration markers, found {len(pts)}")
    return calibrate_from_points(_order_top_left_first(pts), config.CAL_MARKERS_ROBOT)
