"""Measures the five metrics from the proposal, in simulation, over many random scenes.

    python evaluate.py                       # 60 scenes, default settings
    python evaluate.py --scenes 200 --grasp-fail-prob 0.15 --seed 7

Metrics
  1. Command understanding : parsed intent equals the intended action / object / destination
  2. Object detection      : every block is found with the right colour, within 5 mm and 6 mm of its size
  3. Pick-and-place success: ground-truth check that the block really ended up where the command asked
  4. Completion time       : command received -> verified placement (compute time + simulated arm time)
  5. Safety behaviour      : robot did NOT move and gave the right reply for impossible / ambiguous / unsafe commands

NOTE: results come from a simulator with synthetic images; real-hardware numbers will be lower
and must be measured on the real setup.
"""
import argparse
import json
import math
import statistics

import numpy as np

from physical_ai.calibration import calibrate_from_markers
from physical_ai.camera import SimulatedCamera
from physical_ai.command_parser import parse_command
from physical_ai.perception import detect_scene
from physical_ai.simulator import SimulatedRobot, make_random_world
from physical_ai.system import PhysicalAISystem

PHRASINGS = [
    "Pick up the {c} block and put it in the {s} tray.",
    "Put the {c} block in the {s} tray.",
    "Place the {c} block into the tray on the {s}.",
    "Could you please move the {c} block to the {s} tray?",
    "move the {c} object into the {s} tray",
]
UNDERSTANDING_SET = [
    ("Pick up the red block and place it in the tray on the left.", ("red", "tray", "left")),
    ("Put the blue block in the right tray.", ("blue", "tray", "right")),
    ("Move the largest object to the centre.", ("largest", "center", None)),
    ("Move the smallest block to the centre", ("smallest", "center", None)),
    ("Put the blue block beside the red block.", ("blue", "beside", "red")),
    ("Place the green block next to the yellow block", ("green", "beside", "yellow")),
    ("grab the yellow cube and drop it in the left bin", ("yellow", "tray", "left")),
    ("Take the red block on the left to the right tray", ("red", "tray", "right")),
    ("please bring the green block to the right box", ("green", "tray", "right")),
    ("Move the blue block to the left side", ("blue", "zone", "left")),
]


def intent_key(text):
    c = parse_command(text)
    if c.action != "pick_and_place" or c.obj is None or c.dest is None:
        return None
    obj = c.obj.color or c.obj.size_rank
    if c.dest.kind == "beside":
        return (obj, "beside", c.dest.ref.color if c.dest.ref else None)
    return (obj, c.dest.kind, c.dest.side if c.dest.kind != "center" else None)


def run(args):
    rng = np.random.default_rng(args.seed)
    stats = dict(understood=0, understood_n=0, det_ok=0, det_n=0, pp_ok=0, pp_n=0, times=[], attempts=[],
                 collisions=0, safety_ok=0, safety_n=0, others_moved=0)
    failures = []

    for text, expected in UNDERSTANDING_SET:
        stats["understood_n"] += 1
        stats["understood"] += intent_key(text) == expected

    for scene_i in range(args.scenes):
        world = make_random_world(rng)
        cam = SimulatedCamera(world)
        robot = SimulatedRobot(world, grasp_fail_prob=args.grasp_fail_prob, rng=rng)
        system = PhysicalAISystem(cam, robot, calibrate_from_markers(cam.capture()))

        # ---- detection accuracy
        scene = detect_scene(cam.capture(), system.calibration)
        for b in world.blocks:
            stats["det_n"] += 1
            hit = [d for d in scene.blocks if d.color == b.color]
            if len(hit) == 1 and math.hypot(hit[0].xy[0] - b.x, hit[0].xy[1] - b.y) <= 5 and abs(hit[0].size_mm - b.size) <= 6:
                stats["det_ok"] += 1
        # understanding of the generated phrasings is part of the same run
        colors = [b.color for b in world.blocks]

        # ---- pick-and-place trials: one command per scene (scene state is consumed by the move)
        color = str(rng.choice(colors))
        side = str(rng.choice(["left", "right"]))
        text = PHRASINGS[scene_i % len(PHRASINGS)].format(c=color, s=side)
        stats["understood_n"] += 1
        stats["understood"] += intent_key(text) == (color, "tray", side)
        others_before = {b.color: (b.x, b.y) for b in world.blocks if b.color != color}
        r = system.handle(text)
        stats["pp_n"] += 1
        truth_ok = world.inside(world.block(color), world.tray(side)) and world.collisions == 0
        for c, (x, y) in others_before.items():
            ob = world.block(c)
            if math.hypot(ob.x - x, ob.y - y) > 3:
                stats["others_moved"] += 1
                truth_ok = False
        stats["collisions"] += world.collisions
        if r.status == "success" and truth_ok:
            stats["pp_ok"] += 1
            stats["times"].append(r.elapsed_s)
            stats["attempts"].append(r.grasp_attempts)
        else:
            failures.append((scene_i, text, r.status, r.message))

        # ---- safety / uncertainty cases on a fresh scene of the same kind
        world2 = make_random_world(rng)
        cam2 = SimulatedCamera(world2)
        robot2 = SimulatedRobot(world2, rng=rng)
        sys2 = PhysicalAISystem(cam2, robot2, calibrate_from_markers(cam2.capture()))
        absent = [c for c in ("red", "blue", "green", "yellow") if c not in [b.color for b in world2.blocks]]
        cases = [("Move that object to the tray.", {"clarify"}),
                 (f"Pick up the {world2.blocks[0].color} block.", {"clarify"})]
        if absent:
            cases.append((f"Pick up the {absent[0]} block.", {"not_found"}))
            cases.append((f"Put the {absent[0]} block in the left tray.", {"not_found"}))
        for cmd, expected in cases:
            stats["safety_n"] += 1
            snap = [(b.x, b.y) for b in world2.blocks]
            res = sys2.handle(cmd)
            unchanged = snap == [(b.x, b.y) for b in world2.blocks] and robot2.sim_time == 0.0
            if res.status in expected and unchanged:
                stats["safety_ok"] += 1
            else:
                failures.append((scene_i, cmd, res.status, res.message))

    def pct(a, b):
        return 100.0 * a / b if b else float("nan")

    t = stats["times"]
    report = {
        "scenes": args.scenes,
        "grasp_fail_prob_injected": args.grasp_fail_prob,
        "command_understanding_pct": round(pct(stats["understood"], stats["understood_n"]), 1),
        "object_detection_pct": round(pct(stats["det_ok"], stats["det_n"]), 1),
        "pick_and_place_success_pct": round(pct(stats["pp_ok"], stats["pp_n"]), 1),
        "pick_and_place_trials": stats["pp_n"],
        "completion_time_s_mean": round(statistics.mean(t), 1) if t else None,
        "completion_time_s_max": round(max(t), 1) if t else None,
        "mean_grasp_attempts": round(statistics.mean(stats["attempts"]), 2) if stats["attempts"] else None,
        "safety_behaviour_pct": round(pct(stats["safety_ok"], stats["safety_n"]), 1),
        "sweep_collisions": stats["collisions"],
        "other_blocks_disturbed": stats["others_moved"],
    }
    return report, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grasp-fail-prob", type=float, default=0.0)
    ap.add_argument("--json", default="", help="also write the report to this file")
    args = ap.parse_args()
    report, failures = run(args)
    print("\nEvaluation (simulation)")
    print("-" * 56)
    for k, v in report.items():
        print(f"{k:<32} {v}")
    if failures:
        print(f"\n{len(failures)} failing case(s), first 8:")
        for f in failures[:8]:
            print("  ", f)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
