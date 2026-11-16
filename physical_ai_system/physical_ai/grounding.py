"""Grounding: match the words in a parsed command to objects the camera actually sees.

This is where "the blue block" becomes a specific detected object with coordinates, where
"the tray on the left" becomes a place position, and where ambiguity is caught
("move that object to the tray" -> ask which object) instead of guessed.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import config
from .command_parser import DestRef, ObjectRef, ParsedCommand
from .perception import DetectedObject, Scene


@dataclass
class Task:
    target: DetectedObject
    dest_xy: Tuple[float, float]
    dest_label: str
    dest_surface_z: float = 0.0
    dest_tray: Optional[DetectedObject] = None


@dataclass
class GroundingResult:
    status: str                      # ok | clarify | not_found | unsafe
    message: str = ""
    task: Optional[Task] = None


def describe(o: DetectedObject) -> str:
    x, y = o.xy
    side = "on the left" if y > 40 else "on the right" if y < -40 else "in the middle"
    depth = "near the robot" if x < 125 else "far from the robot" if x > 175 else ""
    return f"{o.name} {side}{' ' + depth if depth else ''}".strip()


# ------------------------------------------------------------------ geometry helpers
def _in_workspace(x: float, y: float, margin: float = config.EDGE_MARGIN) -> bool:
    w = config.WORKSPACE
    r = math.hypot(x, y)
    return (w["x_min"] + margin <= x <= w["x_max"] - margin and w["y_min"] + margin <= y <= w["y_max"] - margin
            and config.REACH_MIN + 5 <= r <= config.REACH_MAX - 5)


def _free_of_blocks(xy, size, scene: Scene, ignore: Optional[DetectedObject]) -> bool:
    for b in scene.blocks:
        if b is ignore:
            continue
        cheb = max(abs(xy[0] - b.xy[0]), abs(xy[1] - b.xy[1]))
        if cheb < (size + b.size_mm) / 2 + config.PLACE_CLEARANCE:
            return False
    return True


def _free_of_trays(xy, size, scene: Scene) -> bool:
    for t in scene.trays:
        cheb = max(abs(xy[0] - t.xy[0]), abs(xy[1] - t.xy[1]))
        if cheb < t.half + size / 2 + 4:
            return False
    return True


def find_free_spot(scene: Scene, size: float, preferred: Tuple[float, float],
                   ignore: Optional[DetectedObject], max_radius: float = 90.0, step: float = 6.0):
    """Nearest table position to `preferred` that is reachable and clear of other objects."""
    n = int(max_radius / step)
    cands = [(i * step, j * step) for i in range(-n, n + 1) for j in range(-n, n + 1)]
    cands.sort(key=lambda c: c[0] ** 2 + c[1] ** 2)
    for dx, dy in cands:
        p = (preferred[0] + dx, preferred[1] + dy)
        if _in_workspace(*p) and _free_of_blocks(p, size, scene, ignore) and _free_of_trays(p, size, scene):
            return p
    return None


def _tray_slot(tray: DetectedObject, size: float, scene: Scene, ignore: DetectedObject):
    inner = tray.half - size / 2 - 4
    if inner < 0:
        return None
    offs = [0.0] if inner < 6 else [0.0, inner, -inner]
    cands = sorted(((tray.xy[0] + dx, tray.xy[1] + dy) for dx in offs for dy in offs),
                   key=lambda p: (p[0] - tray.xy[0]) ** 2 + (p[1] - tray.xy[1]) ** 2)
    for p in cands:
        if _free_of_blocks(p, size, scene, ignore):
            return p
    return None


# ------------------------------------------------------------------ resolving references
def resolve_object(ref: ObjectRef, scene: Scene, exclude: Optional[DetectedObject] = None, role: str = "object"):
    """Returns (DetectedObject | None, GroundingResult | None). The result is set when it fails."""
    blocks = [b for b in scene.blocks if b is not exclude]
    if not blocks:
        return None, GroundingResult("not_found", "I can't see any objects on the table.")
    cands = blocks
    if ref.color:
        cands = [b for b in blocks if b.color == ref.color]
        if not cands:
            other = "another" if exclude is not None and exclude.color == ref.color else "a"
            return None, GroundingResult("not_found", f"I can't see {other} {ref.color} block on the table.")
    if ref.size_rank:
        cands = sorted(cands, key=lambda b: b.size_mm, reverse=(ref.size_rank == "largest"))
        if len(cands) > 1 and abs(cands[0].size_mm - cands[1].size_mm) < 4.0:
            return None, GroundingResult(
                "clarify", f"The {ref.size_rank} object is ambiguous: {describe(cands[0])} and "
                           f"{describe(cands[1])} are about the same size. Which one?")
        return cands[0], None
    if len(cands) == 1:
        return cands[0], None
    if ref.color:
        listing = "; ".join(describe(b) for b in cands)
        return None, GroundingResult("clarify", f"I see {len(cands)} {ref.color} blocks ({listing}). Which one do you mean?")
    listing = "; ".join(describe(b) for b in cands)
    return None, GroundingResult("clarify", f"Which {role} do you mean? I can see: {listing}.")


def _resolve_tray(dest: DestRef, scene: Scene):
    trays = scene.trays
    if not trays:
        return None, GroundingResult("not_found", "I can't see any tray on the table.")
    if dest.side == "left":
        cands = [t for t in trays if t.xy[1] > 20]
    elif dest.side == "right":
        cands = [t for t in trays if t.xy[1] < -20]
    elif dest.side == "center":
        cands = [t for t in trays if abs(t.xy[1]) <= 20]
    else:
        cands = trays
    if not cands:
        return None, GroundingResult("not_found", f"I can't see a tray on the {dest.side}.")
    if len(cands) > 1:
        return None, GroundingResult("clarify", "I can see several trays. Which one: " +
                                     ", ".join(t.name for t in cands) + "?")
    return cands[0], None


# ------------------------------------------------------------------ main entry point
_PROBLEM_MESSAGES = {
    "missing_destination": "Where should I put it? For example: 'in the left tray' or 'to the centre'.",
    "unsupported_destination": "I can only place objects in a tray, at the centre, on the left/right, or beside another block.",
    "unclear_destination": "Beside which object? Please name it (for example 'beside the red block').",
}


def ground(cmd: ParsedCommand, scene: Scene) -> GroundingResult:
    if cmd.obj is None:
        return GroundingResult("clarify", "Which object should I move?")
    target, fail = resolve_object(cmd.obj, scene)
    if fail:
        return fail
    for problem, msg in _PROBLEM_MESSAGES.items():
        if problem in cmd.problems:
            return GroundingResult("clarify", f"I found the {target.name}. {msg}")

    dest = cmd.dest
    if dest.kind == "tray":
        tray, fail = _resolve_tray(dest, scene)
        if fail:
            return fail
        slot = _tray_slot(tray, target.size_mm, scene, target)
        if slot is None:
            return GroundingResult("unsafe", f"The {target.name} does not fit in the {tray.name} "
                                             f"(too large, or the tray is full).")
        return GroundingResult("ok", task=Task(target, slot, f"the {tray.name}", config.TRAY_FLOOR_Z, tray))

    if dest.kind in ("center", "zone", "beside"):
        if dest.kind == "center":
            preferred, label = config.CENTER_XY, "the centre of the table"
        elif dest.kind == "zone":
            preferred = (config.CENTER_XY[0], config.ZONE_Y if dest.side == "left" else -config.ZONE_Y)
            label = f"the {dest.side} side of the table"
        else:
            ref, fail = resolve_object(dest.ref, scene, exclude=target, role="reference object")
            if fail:
                return fail
            gap = (target.size_mm + ref.size_mm) / 2 + config.PLACE_CLEARANCE + 4
            opts = [(ref.xy[0], ref.xy[1] + gap), (ref.xy[0], ref.xy[1] - gap),
                    (ref.xy[0] + gap, ref.xy[1]), (ref.xy[0] - gap, ref.xy[1])]
            opts.sort(key=lambda p: math.dist(p, target.xy))
            preferred = next((p for p in opts if _in_workspace(*p)
                              and _free_of_blocks(p, target.size_mm, scene, target)
                              and _free_of_trays(p, target.size_mm, scene)), opts[0])
            label = f"beside the {ref.name}"
        spot = find_free_spot(scene, target.size_mm, preferred, target)
        if spot is None:
            return GroundingResult("unsafe", f"I can't find a free spot {label.replace('the ', '', 1)}.")
        return GroundingResult("ok", task=Task(target, spot, label, 0.0, None))

    return GroundingResult("clarify", _PROBLEM_MESSAGES["unsupported_destination"])
