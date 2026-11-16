# Physical AI System — natural-language-controlled pick-and-place robot

Code for the proposal *"A Natural-Language-Controlled Intelligent Object Handling Robot"*.
A camera-guided arm hears/reads a command such as **"Pick up the red block and place it in the tray on the left"**,
finds the objects, checks it is safe, moves the object, and verifies the result.

> **Status:** everything runs and is tested **in simulation**. The Arduino firmware and the serial driver are
> written and the driver is tested against a fake serial port, but **nothing here has been run on a physical arm**.
> Expect to tune calibration, servo mapping and thresholds on real hardware (see "Moving to real hardware").

## Quick start (no hardware)

```bash
pip install -r requirements.txt
python main.py --demo --save-images out/      # the proposal's example commands; saves annotated camera views
python main.py                                # type your own commands
python -m pytest -q                           # 54 tests
python evaluate.py --scenes 60                # the proposal's five metrics, in simulation
```

## The pipeline (proposal §3 and §6) and where each step lives

| Step | Module |
|---|---|
| 1-3 Receive command (text / voice) | `main.py`, `physical_ai/speech_input.py` |
| 4 Interpret intent → `{action, object, destination}` | `physical_ai/command_parser.py` (rule-based), `physical_ai/llm_parser.py` (optional LLM) |
| 5 Perceive workspace (objects, trays, positions) | `physical_ai/camera.py`, `physical_ai/perception.py` |
| Camera → robot coordinates | `physical_ai/calibration.py`, `calibrate.py` |
| 6 Ground words in vision, detect ambiguity | `physical_ai/grounding.py` |
| 7 Ask for clarification instead of guessing | `physical_ai/grounding.py`, `physical_ai/system.py` |
| 8 Plan grasp pose and path | `physical_ai/planner.py`, `physical_ai/kinematics.py` |
| Safety checks before any motion | `physical_ai/safety.py` |
| 9-10 Move, grasp, verify grip, place, release | `physical_ai/executor.py`, `physical_ai/robot_arm.py` |
| Arm hardware driver | `physical_ai/robot_arm.py` (`SerialArmRobot`), `arduino/arm_controller/arm_controller.ino` |
| 11 Verify by camera, report success/failure | `physical_ai/system.py` (`verify_placement`) |
| Simulator (world, camera, arm) | `physical_ai/simulator.py` |
| Metrics (proposal §9) | `evaluate.py` |
| Tests | `tests/` |

Everything the rest of the system needs from the robot goes through the `RobotArm` interface, so the simulated arm,
the Arduino arm, or a ROS 2 / vendor-SDK arm are interchangeable.

## Safety behaviour (proposal §7)

- Refuses to act when the object is missing, ambiguous, low-confidence, too wide for the gripper, unreachable, outside the working area, or too close to a neighbour for the fingers.
- Every key pose is checked against joint limits before the first movement.
- Carries objects high enough that their underside clears every other block.
- Low default speed (`SPEED_SCALE`), grasp verified by gripper current, up to 3 retries, then stops and reports.
- Software e-stop (`stop` command or `system.emergency_stop()`), checked between every motion step.
- Firmware: hardware E-stop button on D2 and a power relay on D7 that cuts the servo supply. **Do not rely on software alone; wire the physical E-stop.**
- Uncertain detections (confidence 60-80%) need a yes/no confirmation.
- Not implemented: detecting a human hand in the workspace. Use a physical barrier / keep hands out while it runs.

## Moving to real hardware

1. **Parts:** 4-DOF servo arm with gripper, USB webcam fixed above the table looking straight down, Arduino, separate servo power supply (5-6 V, enough amps), relay, E-stop button, current-sense on the gripper supply, coloured cubes (red/blue/green/yellow), two dark trays, four magenta dots.
2. **Arm geometry:** edit `ArmGeometry`, `JOINT_LIMITS` and `SERVO_MAP` in `physical_ai/config.py` to match your arm. Then recompute the `HOME` angles in the `.ino` (`to_servo(solve_ik(*HOME_XYZ))`).
3. **Wiring:** pin list is at the top of `arm_controller.ino`. Flash it with the Arduino IDE.
4. **Camera:** mount it so the robot is at the bottom of the image and the work area fills the frame. Park pose must be outside the camera view.
5. **Calibration:** put the four magenta dots at `CAL_MARKERS_ROBOT` positions (measure from the base axis), run `python calibrate.py --camera 0`. Jog the gripper tip to each dot to confirm it lands within a few mm.
6. **Gripper threshold:** set `hold_current_threshold` in `SerialArmRobot` from real readings (empty-close vs holding a block).
7. Start slow: `python main.py --hardware --port /dev/ttyUSB0 --speed 0.2`, with light soft blocks and your hand on the E-stop.
8. **Colours:** tune `HSV_RANGES` for your lighting. A depth camera is the next upgrade (this version assumes a flat table and a top-down camera).

## Simulator results

`python evaluate.py` (60 random scenes, seed 0): command understanding 100%, detection 100%, pick-and-place 100% (60/60), mean completion 8.7 s (simulated arm time + compute), safety behaviour 100%, 0 collisions.
With 20% injected grasp failures the retry logic still completed 98% of trials.
These are **simulator numbers** with clean synthetic images and idealised physics. They show the logic is correct, not what a real arm will score.

## Why this solution is good

- **Complete loop:** language → perception → reasoning → motion → feedback, with a real action in the world, which is what a Physical AI system requires.
- **Safe by design:** it checks before moving, asks when unsure, verifies after moving, and stops on faults. A wrong guess is never executed silently.
- **Modular:** each stage is a separate, tested module behind a small interface; swap the parser, detector or arm without touching the rest.
- **Testable without hardware:** the simulator lets the whole system be developed and measured before the arm exists.
- **Measurable:** the five metrics from the proposal are computed automatically.
- **Cheap and reproducible:** a hobby arm, a webcam and an Arduino; no GPU needed.
- **Extensible:** depth camera, learned detector, ROS 2/MoveIt 2, mobile base and voice replies fit into the existing interfaces.

## Limits of this version

Top-down camera with a flat table, coloured cubes and dark trays only; rule-based language (the LLM parser is optional and was not tested against the live API); 4-DOF arm cannot rotate the gripper to match a rotated block; no stacking; no hand detection; no ROS 2 node (the `RobotArm` interface is where it would plug in).
