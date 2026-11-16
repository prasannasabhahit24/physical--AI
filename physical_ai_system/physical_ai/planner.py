"""Task / motion planner: turn a grounded Task into the key poses of a pick-and-place."""
from dataclasses import dataclass
from typing import Tuple

from . import config
from .grounding import Task
from .perception import Scene


@dataclass
class PickPlacePlan:
    pick_xy: Tuple[float, float]
    place_xy: Tuple[float, float]
    z_transit: float      # fingertip height while carrying (object underside clears everything)
    z_grasp: float        # fingertip height when grasping
    z_place: float        # fingertip height when releasing
    size_mm: float
    label: str

    def key_poses(self):
        """Every Cartesian pose the arm will visit (used for reachability / safety checks)."""
        px, py = self.pick_xy
        qx, qy = self.place_xy
        return [("above pick", (px, py, self.z_transit)), ("grasp", (px, py, self.z_grasp)),
                ("above place", (qx, qy, self.z_transit)), ("place", (qx, qy, self.z_place))]


def make_plan(task: Task, scene: Scene) -> PickPlacePlan:
    size = task.target.size_mm
    tallest = max([b.size_mm for b in scene.blocks] + [config.TRAY_FLOOR_Z])
    underside = max(config.TRANSIT_CLEARANCE, tallest + 14.0)
    return PickPlacePlan(
        pick_xy=task.target.xy,
        place_xy=task.dest_xy,
        z_transit=underside + size / 2,
        z_grasp=size / 2,
        z_place=task.dest_surface_z + size / 2 + config.PLACE_DROP_MM,
        size_mm=size,
        label=f"{task.target.name} -> {task.dest_label}",
    )
