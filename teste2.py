import cv2
import numpy as np
import mediapipe as mp
from pykinect2 import PyKinectRuntime, PyKinectV2

# Inicializa o Kinect
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

# Inicializa MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Fator de conversão de pixels para cm (ajustável conforme necessário)
PIXELS_TO_CM_SCALE = 0.0265

# Define um limite de visibilidade para evitar leituras instáveis
VISIBILITY_THRESHOLD = 0.6

def calculate_distance_2d(point1, point2):
    """Calcula a distância euclidiana entre dois pontos 2D."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

while True:
    if kinect.has_new_color_frame():
        # Obtém o frame RGB
        frame = kinect.get_last_color_frame()
        frame = frame.reshape((1080, 1920, 4))  # Kinect BGRA frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Converte para BGR

        # Converte para RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Processa a pose
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Certifique-se de que os pontos estão visíveis
            if (landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].visibility > VISIBILITY_THRESHOLD and 
                landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility > VISIBILITY_THRESHOLD):

                # Obtém as coordenadas dos pontos desejados
                index_finger = (int(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x * frame.shape[1]),
                                int(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y * frame.shape[0]))

                foot_index = (int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x * frame.shape[1]),
                              int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y * frame.shape[0]))

                # Calcula a distância em pixels
                distance_px = calculate_distance_2d(index_finger, foot_index)

                # Converte pixels para centímetros
                distance_cm = distance_px * PIXELS_TO_CM_SCALE

                # Exibe a distância na tela
                cv2.putText(frame, f'Distance: {distance_cm:.2f} cm', (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Desenha um círculo nos pontos analisados
                cv2.circle(frame, index_finger, 10, (0, 0, 255), -1)  # Vermelho para a mão
                cv2.circle(frame, foot_index, 10, (255, 0, 0), -1)  # Azul para o pé

                # Desenha uma linha conectando os pontos
                cv2.line(frame, index_finger, foot_index, (0, 255, 255), 2)  # Amarelo

        # Exibe o frame processado
        cv2.imshow('Distance Measurement', frame)

    # Pressione "q" para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cv2.destroyAllWindows()
