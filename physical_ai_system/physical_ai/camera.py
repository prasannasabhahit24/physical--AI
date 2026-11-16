"""Camera sources: a real USB camera, or the simulator's virtual top-down camera."""
from abc import ABC, abstractmethod

import numpy as np

from . import config


class Camera(ABC):
    @abstractmethod
    def capture(self) -> np.ndarray:
        """Return one BGR frame."""


class OpenCVCamera(Camera):
    def __init__(self, index: int = 0, size=config.IMAGE_SIZE, warmup_frames: int = 5):
        import cv2
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera {index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self._warmup = warmup_frames

    def capture(self) -> np.ndarray:
        for _ in range(self._warmup):         # let auto-exposure settle
            self._cap.read()
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("camera returned no frame")
        return frame

    def release(self) -> None:
        self._cap.release()


class SimulatedCamera(Camera):
    def __init__(self, world):
        self.world = world

    def capture(self) -> np.ndarray:
        return self.world.render()
