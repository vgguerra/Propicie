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
    escolha = show_main_menu()
        
    if escolha == "quit":
        break
        
    elif escolha == "auto":
        print("A iniciar Modo Automático...")
        # Corre o primeiro
        sit_and_reach.run(kinect, holistic, finish)
        # Assim que o ecrã final do primeiro fechar (pressionando ESC), corre o segundo
        print("A iniciar o segundo exercício...")
        back_scratch.run(kinect, holistic, finish)
        # Ao acabar o segundo, o loop continua e volta sozinho para o Menu Principal!
        
    elif escolha == "sit_and_reach":
        print("A iniciar Sit and Reach...")
        sit_and_reach.run(kinect, holistic, finish)
        # Quando fechar a tela final, volta para o menu automaticamente
        
    elif escolha == "back_scratch":
        print("A iniciar Back Scratch...")
        back_scratch.run(kinect, holistic, finish)
        # Quando fechar a tela final, volta para o menu automaticamente

    elif escolha == "view_data":
        print("[RUNNER] Opção view_data selecionada no menu.", flush=True)
        try:
            print("[RUNNER] A importar show_data_visualization...", flush=True)
            from ui.view_data import show_data_visualization
            print("[RUNNER] Importação bem-sucedida! A chamar a função...", flush=True)
            
            show_data_visualization()
            
            print("[RUNNER] Função show_data_visualization terminou com sucesso.", flush=True)
        except Exception as e:
            print(f"[RUNNER CRITICAL ERROR] Apanhado no runner: {e}", flush=True)
        except BaseException as be:
            print(f"[RUNNER CRITICAL SYSTEM] Erro de sistema detetado: {be}", flush=True)
finish()