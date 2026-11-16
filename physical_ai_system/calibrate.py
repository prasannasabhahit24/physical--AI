"""Camera-to-robot calibration for real hardware.

1. Fix the camera above the table, looking straight down, with the robot at the bottom of the image.
2. Place four magenta dots at the positions in config.CAL_MARKERS_ROBOT (measure from the arm's base axis).
3. Run:  python calibrate.py --camera 0 --out calibration.json
   It shows the detected dots and saves the pixel->robot mapping. Press any key to close the preview.

Check the result by jogging the gripper tip to each dot: it should land within ~3 mm.
"""
import argparse

import cv2

from physical_ai.calibration import calibrate_from_markers, find_marker_pixels
from physical_ai.camera import OpenCVCamera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default="calibration.json")
    args = ap.parse_args()

    frame = OpenCVCamera(args.camera).capture()
    calib = calibrate_from_markers(frame)
    calib.save(args.out)
    for u, v in find_marker_pixels(frame):
        x, y = calib.pixel_to_robot(u, v)
        cv2.circle(frame, (int(u), int(v)), 18, (0, 255, 255), 2)
        cv2.putText(frame, f"({x:.0f},{y:.0f})", (int(u) + 20, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    print(f"saved {args.out}; mm per pixel at image centre: {calib.mm_per_pixel(320, 240):.3f}")
    cv2.imshow("calibration", frame)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
