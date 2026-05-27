# =============================================================================
# runner.py — Main entry point
# =============================================================================

import cv2
from pykinect2 import PyKinectRuntime, PyKinectV2
import mediapipe as mp

print("1 - A importar language_select...")
from ui.language_select import show_language_select
print("2 - A importar menu...")
from ui.menu import show_main_menu
print("3 - A importar locale_setup...")
from locale_setup import set_language
print("4 - A importar utils...")
from utils import show_register_screen
print("5 - A importar exercicios...")
from exercicios import sit_and_reach, back_scratch
print("6 - Todos os imports OK")

# ---------------------------------------------------------------------------
# Language selection (graphical — click on flag)
# ---------------------------------------------------------------------------
lang = show_language_select()
set_language(lang)

# ---------------------------------------------------------------------------
# Hardware initialisation
# ---------------------------------------------------------------------------
kinect   = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
holistic = mp.solutions.holistic.Holistic()


def finish():
    cv2.destroyAllWindows()
    kinect.close()
    raise SystemExit(0)


# ---------------------------------------------------------------------------
# Main menu loop
# ---------------------------------------------------------------------------
while True:
    action = show_main_menu()

    if action == "quit":
        finish()

    elif action == "auto":
        try:
            sit_and_reach.run(kinect, holistic, finish)
            back_scratch.run(kinect, holistic, finish)
        except SystemExit:
            raise
        except BaseException as e:
            import traceback, sys
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            input("Enter...")

    elif action == "sit_and_reach":
        sit_and_reach.run(kinect, holistic, finish)

    elif action == "back_scratch":
        back_scratch.run(kinect, holistic, finish)

    elif action == "view_data":
        # Placeholder — to be implemented
        pass