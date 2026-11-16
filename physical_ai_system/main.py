"""Command-line entry point.

Simulation (no hardware needed):
    python main.py --demo                      # run the proposal's example commands
    python main.py                             # interactive: type commands
    python main.py --voice                     # speak commands (needs SpeechRecognition)

Real hardware:
    python main.py --hardware --port /dev/ttyUSB0 --camera 0 [--calibration calibration.json]
"""
import argparse
import os

import cv2
import numpy as np

from physical_ai import config
from physical_ai.calibration import Calibration, calibrate_from_markers
from physical_ai.camera import OpenCVCamera, SimulatedCamera
from physical_ai.perception import annotate, detect_scene
from physical_ai.robot_arm import SerialArmRobot
from physical_ai.simulator import SimulatedRobot, make_default_world
from physical_ai.system import PhysicalAISystem

DEMO_COMMANDS = [
    "Put the blue block in the right tray.",
    "Pick up the red block and place it in the tray on the left.",
    "Move the largest object to the centre.",
    "Pick up the green block.",              # absent -> should say it cannot see it
    "Move that object to the tray.",         # ambiguous -> should ask which object
]


def ask_yes_no(question: str) -> bool:
    return input(f"{question} [y/N] ").strip().lower().startswith("y")


def build_system(args):
    if args.hardware:
        camera = OpenCVCamera(args.camera)
        robot = SerialArmRobot(args.port)
        robot.connect()
        if args.calibration and os.path.exists(args.calibration):
            calib = Calibration.load(args.calibration)
        else:
            calib = calibrate_from_markers(camera.capture())
            if args.calibration:
                calib.save(args.calibration)
        world = None
    else:
        world = make_default_world(seed=args.seed)
        camera = SimulatedCamera(world)
        robot = SimulatedRobot(world, grasp_fail_prob=args.grasp_fail_prob, rng=np.random.default_rng(args.seed))
        calib = calibrate_from_markers(camera.capture())
    system = PhysicalAISystem(camera, robot, calib, confirm=ask_yes_no, speed=args.speed)
    return system, world


def save_view(system, folder, name):
    if folder:
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(os.path.join(folder, name), annotate(system.perceive()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hardware", action="store_true", help="use the real arm + camera instead of the simulator")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--calibration", default="calibration.json")
    ap.add_argument("--demo", action="store_true", help="run the example commands and exit")
    ap.add_argument("--voice", action="store_true", help="take spoken commands")
    ap.add_argument("--llm", action="store_true", help="use the optional LLM parser (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=config.SPEED_SCALE, help="0..1, keep low at first")
    ap.add_argument("--grasp-fail-prob", type=float, default=0.0, help="simulator only")
    ap.add_argument("--save-images", default="", help="folder for annotated before/after camera views")
    args = ap.parse_args()

    system, _ = build_system(args)
    if args.llm:
        from physical_ai.llm_parser import parse_with_llm
        system.parser = parse_with_llm

    def run(text, i=0):
        save_view(system, args.save_images, f"{i:02d}_before.png")
        r = system.handle(text)
        print(f"  [{r.status}] {r.message}  ({r.elapsed_s:.1f}s)")
        save_view(system, args.save_images, f"{i:02d}_after.png")
        return r

    if args.demo:
        for i, text in enumerate(DEMO_COMMANDS, 1):
            print(f"> {text}")
            run(text, i)
        return

    print("Type a command (or 'quit'). Say or type 'stop' for an emergency stop.")
    i = 0
    while True:
        if args.voice:
            from physical_ai.speech_input import listen_once
            try:
                text = listen_once()
                print(f"> {text}")
            except Exception as e:
                print(f"  could not hear that ({e})")
                continue
        else:
            text = input("> ").strip()
        if text.lower() in ("quit", "exit"):
            break
        if text.lower() == "reset":
            system.reset_emergency_stop()
            print("  emergency stop cleared")
            continue
        i += 1
        run(text, i)


if __name__ == "__main__":
    main()
