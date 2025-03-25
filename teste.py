import cv2
import mediapipe as mp
import numpy as np
import math
from pykinect2 import PyKinectRuntime, PyKinectV2

# Approximate ratio of pixels to cm at 1 meter distance
PIXEL_TO_CM_RATIO = 0.533333  # 1 pixel ≈ 0.125 cm at 1m distance


# Kinect initialization
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

# Media Pipe Holistic initialization
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Function to calculate Euclidean distance in 2D (x, y only)
def calculate_distance_2d(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to calculate angle between three points    
def calculate_angle(a, b, c):
    # Certifique-se de que 'a', 'b' e 'c' são objetos numpy.ndarray ou pontos do MediaPipe Landmark
    # Então, acesse as coordenadas x e y diretamente
    vetor_1 = (a[0] - b[0], a[1] - b[1])  # Vetor do ponto b para o ponto a
    vetor_2 = (c[0] - b[0], c[1] - b[1])  # Vetor do ponto b para o ponto c

    # Produto escalar
    produto_escalar = vetor_1[0] * vetor_2[0] + vetor_1[1] * vetor_2[1]

    # Normas (módulos)
    norma_vetor_1 = np.sqrt(vetor_1[0]**2 + vetor_1[1]**2)
    norma_vetor_2 = np.sqrt(vetor_2[0]**2 + vetor_2[1]**2)

    # Cálculo do cosseno do ângulo
    cos_angulo = produto_escalar / (norma_vetor_1 * norma_vetor_2)

    # Garantir que o valor de cos_angulo esteja no intervalo [-1, 1] para evitar erros de precisão
    cos_angulo = max(-1, min(1, cos_angulo))

    # Ângulo em radianos
    angulo_radianos = math.acos(cos_angulo)

    # Converter para graus
    angulo_graus = math.degrees(angulo_radianos)

    return angulo_graus

# Function to calculate average distance
def average_distance(distances):
    return sum(distances) / len(distances)

# Function to draw dynamic angle arc on image
def draw_dynamic_angle_arc(image, p1, p2, p3, angle):
    # Draw lines between points
    cv2.line(image, p1, p2, (255, 255, 0), 2)
    cv2.line(image, p2, p3, (255, 255, 0), 2)
    
    # Calculate the arc points
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    # Determine the arc center point
    arc_center = tuple(p2.astype(int))
    
    # Calculate radius and draw arc dynamically based on angle
    radius = int(np.linalg.norm(p1 - p2) / 2)
    
    # Draw filled arc
    axes = (radius, radius)
    cv2.ellipse(image, arc_center, axes, 0, 0, angle, (0, 255, 0), -1)
    cv2.ellipse(image, arc_center, axes, 0, 0, angle, (0, 0, 255), 2)

while True:
    if kinect.has_new_color_frame():  
        frame = kinect.get_last_color_frame()
        frame = frame.reshape((1080, 1920, 4))  # Kinect BGRA frame dimensions
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  
        
        # Convertendo para RGB para o MediaPipe
        rgb_frame = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = holistic.process(rgb_frame)

        rgb_frame.flags.writeable = True
        image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Desenhar os landmarks de pose e mão direita
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            landmark = results.pose_landmarks.landmark[31]
            foot = int(landmark.x * 640), int(landmark.y * 480)
        
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            #
            landmark = results.right_hand_landmarks.landmark[12]  
            hand = int(landmark.x * 640), int(landmark.y * 480) 


            dist_pixels = calculate_distance_2d(hand,foot)
            dist_cm = dist_pixels * PIXEL_TO_CM_RATIO  
            cv2.putText(image, f"Dist: {dist_cm - 2:.2f} cm", (1080, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        
        cv2.imshow('MediaPipe Holistic', image)

       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
kinect.close()
