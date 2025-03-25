import cv2
import mediapipe as mp
from pykinect2 import PyKinectRuntime
from pykinect2 import PyKinectV2

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose()
hands = mp_hands.Hands()

# Inicializar Kinect
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

# Relação aproximada de pixels para cm a 1 metro de distância
PIXEL_TO_CM_RATIO = 0.125  # 1 pixel ≈ 0.125 cm a 1m de distância

while True:
    frame = None  

    # Capturar frame de cor do Kinect
    if kinect.has_new_color_frame():
        color_frame = kinect.get_last_color_frame()
        color_frame = color_frame.reshape((1080, 1920, 4))[:, :, :3]  # Remover canal alfa
        frame = cv2.resize(color_frame, (640, 480))  # Reduzir tamanho para melhor desempenho

    if frame is None:
        continue  # Pular iteração se não houver frame válido

    # Converter para RGB para o MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processar com MediaPipe
    pose_results = pose.process(frame_rgb)
    hands_results = hands.process(frame_rgb)

    hand_x, hand_y = None, None
    foot_x, foot_y = None, None

    # Detectar mão (dedo indicador - landmark[8])
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            landmark = hand_landmarks.landmark[8]
            hand_x, hand_y = int(landmark.x * 640), int(landmark.y * 480)
            cv2.circle(frame, (hand_x, hand_y), 10, (0, 255, 0), -1)

    # Detectar pé (tornozelo direito - landmark[31])
    if pose_results.pose_landmarks:
        landmark = pose_results.pose_landmarks.landmark[31]
        foot_x, foot_y = int(landmark.x * 640), int(landmark.y * 480)
        cv2.circle(frame, (foot_x, foot_y), 10, (0, 0, 255), -1)

    # Calcular distância 2D e converter para cm
    if None not in (hand_x, hand_y, foot_x, foot_y):
        dist_pixels = ((hand_x - foot_x) ** 2 + (hand_y - foot_y) ** 2) ** 0.5
        dist_cm = dist_pixels * PIXEL_TO_CM_RATIO  # Converter pixels para cm

        cv2.putText(frame, f"Dist: {dist_cm:.2f} cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Mostrar frame
    cv2.imshow("Kinect + MediaPipe (2D em cm)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
