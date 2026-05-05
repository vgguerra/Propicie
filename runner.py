# =============================================================================
# runner.py — Main entry point
# =============================================================================
# Initialises shared hardware (Kinect, MediaPipe) and participant data once,
# then delegates to each exercise module in sequence.
# =============================================================================

import cv2
from pykinect2 import PyKinectRuntime, PyKinectV2
import mediapipe as mp

from utils import show_register_screen
from exercicios import sit_and_reach, back_scratch

# ---------------------------------------------------------------------------
# Hardware initialisation
# ---------------------------------------------------------------------------
kinect   = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
holistic = mp.solutions.holistic.Holistic()


def finish():
    """Clean shutdown: destroy all OpenCV windows, release Kinect, exit."""
    cv2.destroyAllWindows()
    kinect.close()
    raise SystemExit(0)


# ---------------------------------------------------------------------------
# Participant registration (runs once for both exercises)
# ---------------------------------------------------------------------------
age, height, weight, gender_raw = show_register_screen()
participant = {
    "age":    age,
    "height": height,
    "weight": weight,
    "gender": "Feminine" if gender_raw.strip().upper() == "F" else "Male",
}

# ---------------------------------------------------------------------------
# Exercise sequence
# ---------------------------------------------------------------------------
print("Starting Sit and Reach...")
sit_and_reach.run(kinect, holistic, participant, finish)

print("Starting Back Scratch...")
back_scratch.run(kinect, holistic, participant, finish)

print("All exercises completed.")
finish()