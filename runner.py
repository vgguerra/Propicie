# =============================================================================
# runner.py — Main entry point
# =============================================================================

import cv2
from camera import OrbbecCamera
import mediapipe as mp

from ui.menu import show_main_menu
from utils import ReturnToMenu
from exercises import sit_and_reach, back_scratch

# ---------------------------------------------------------------------------
# Hardware initialisation
# ---------------------------------------------------------------------------
kinect   = OrbbecCamera()
holistic = mp.solutions.holistic.Holistic()


def finish():
    cv2.destroyAllWindows()
    raise ReturnToMenu()


# ---------------------------------------------------------------------------
# Main menu loop
# ---------------------------------------------------------------------------
while True:
    escolha = show_main_menu()
        
    if escolha == "quit":
        break

    try:
        if escolha == "auto":
            print("A iniciar Modo Automático...")
            sit_and_reach.run(kinect, holistic, finish)
            print("A iniciar o segundo exercício...")
            back_scratch.run(kinect, holistic, finish)
            
        elif escolha == "sit_and_reach":
            print("A iniciar Sit and Reach...")
            sit_and_reach.run(kinect, holistic, finish)
            
        elif escolha == "back_scratch":
            print("A iniciar Back Scratch...")
            back_scratch.run(kinect, holistic, finish)

        elif escolha == "view_data":
            try:
                from ui.view_data import show_data_visualization
                show_data_visualization()
            except Exception as e:
                print(f"[RUNNER] Erro no view_data: {e}", flush=True)
    except ReturnToMenu:
        continue

cv2.destroyAllWindows()
kinect.close()