"""Perception: find coloured blocks and trays in a camera frame and locate them in robot coordinates."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import config
from .calibration import Calibration


@dataclass
class DetectedObject:
    kind: str                      # "block" | "tray"
    color: Optional[str]           # block colour, None for trays
    name: str                      # "red block", "left tray"
    center_px: Tuple[float, float]
    xy: Tuple[float, float]        # robot frame, mm
    size_mm: float                 # side length (blocks are cubes, trays square-ish)
    area_px: float
    confidence: float
    yaw_deg: float = 0.0

    @property
    def half(self) -> float:
        return self.size_mm / 2.0


@dataclass
class Scene:
    blocks: List[DetectedObject] = field(default_factory=list)
    trays: List[DetectedObject] = field(default_factory=list)
    image: Optional[np.ndarray] = None

    @property
    def objects(self) -> List[DetectedObject]:
        return self.blocks + self.trays


def _mask_for_color(hsv: np.ndarray, ranges) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _describe_contour(contour, calib: Calibration):
    area = cv2.contourArea(contour)
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    rect = cv2.minAreaRect(contour)
    rect_area = max(rect[1][0] * rect[1][1], 1.0)
    solidity = min(1.0, area / rect_area)
    mm_px = calib.mm_per_pixel(cx, cy)
    size_mm = float(np.sqrt(area)) * mm_px
    return cx, cy, area, solidity, size_mm, float(rect[2])


def detect_scene(image: np.ndarray, calib: Calibration) -> Scene:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    scene = Scene(image=image)

    # ---- trays: large dark regions (outer contour, so a block lying inside does not split it)
    dark = cv2.inRange(hsv, np.array((0, 0, 0)), np.array((179, 255, config.TRAY_MAX_VALUE)))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    trays = []
    for c in contours:
        info = _describe_contour(c, calib)
        if info is None or info[2] < config.MIN_TRAY_AREA_PX:
            continue
        cx, cy, area, solidity, size_mm, yaw = info
        conf = float(np.clip(solidity, 0, 1))
        trays.append(DetectedObject("tray", None, "tray", (cx, cy), calib.pixel_to_robot(cx, cy),
                                    size_mm, area, conf, yaw))
    trays.sort(key=lambda t: -t.xy[1])               # leftmost (largest y) first
    if len(trays) == 1:
        names = ["tray"]
    elif len(trays) == 2:
        names = ["left tray", "right tray"]
    elif len(trays) == 3:
        names = ["left tray", "middle tray", "right tray"]
    else:
        names = [f"tray {i + 1}" for i in range(len(trays))]
    for t, n in zip(trays, names):
        t.name = n
    scene.trays = trays

    # ---- coloured blocks
    for color, ranges in config.HSV_RANGES.items():
        mask = _mask_for_color(hsv, ranges)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            info = _describe_contour(c, calib)
            if info is None or info[2] < config.MIN_BLOCK_AREA_PX:
                continue
            cx, cy, area, solidity, size_mm, yaw = info
            conf = float(np.clip(solidity, 0, 1) * min(1.0, area / (1.5 * config.MIN_BLOCK_AREA_PX)))
            scene.blocks.append(DetectedObject("block", color, f"{color} block", (cx, cy),
                                               calib.pixel_to_robot(cx, cy), size_mm, area, conf, yaw))
    return scene


def annotate(scene: Scene) -> np.ndarray:
    """Return a copy of the image with detections drawn on it (for debugging / demos)."""
    img = scene.image.copy()
    for o in scene.objects:
        u, v = map(int, o.center_px)
        r = int(o.size_mm)  # rough drawing radius in px (scale ~2 px/mm -> half size)
        cv2.rectangle(img, (u - r, v - r), (u + r, v + r), (0, 255, 255), 1)
        cv2.putText(img, f"{o.name} {o.confidence:.2f}", (u - r, v - r - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img
