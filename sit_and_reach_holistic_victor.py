from pykinect2 import PyKinectRuntime, PyKinectV2
from openpyxl import load_workbook
import mediapipe as mp
import numpy as np
import time
import math
import cv2

# Approximate ratio of pixels to cm at 1 meter distance
PIXEL_TO_CM_RATIO = 0.533333  # 1 pixel ≈ 0.125 cm at 1m distance


# Kinect initialization
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

# Media Pipe Holistic initialization
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Function to finish program
def finish_program():
    cv2.destroyAllWindows()
    kinect.close()
    exit()

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

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    vetor_1 = (a[0] - b[0], a[1] - b[1]) 
    vetor_2 = (c[0] - b[0], c[1] - b[1])  

    produto_escalar = vetor_1[0] * vetor_2[0] + vetor_1[1] * vetor_2[1]

    norma_vetor_1 = math.sqrt(vetor_1[0]**2 + vetor_1[1]**2)
    norma_vetor_2 = math.sqrt(vetor_2[0]**2 + vetor_2[1]**2)

    cos_angulo = produto_escalar / (norma_vetor_1 * norma_vetor_2)

    cos_angulo = max(-1, min(1, cos_angulo))

    angulo_radianos = math.acos(cos_angulo)

    angulo_graus = math.degrees(angulo_radianos)

    return angulo_graus

# Load existing spreadsheet
arquivo = "dados.xlsx"
planilha = load_workbook(arquivo)
sheet = planilha.active

progress_calibration1 = 0

while True:

    # variable initialization
    distances = []
    average_over = 6
    final_distance = None
    pose_correct_start_time = None
    calibration_time = None
    pose_held_duration = 8  # seconds, increased for more time before the exercise ends
    calibration_held_duration = 5
    calibration = ""
    progress_calibration = 0

    while True:
        if kinect.has_new_color_frame():  
            frame = kinect.get_last_color_frame()
            frame = frame.reshape((1080, 1920, 4))  # Kinect BGRA frame dimensions
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  
            
            # Converting to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = holistic.process(rgb_frame)

            rgb_frame.flags.writeable = True
            image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # variable initialization
            distance = -1
            rigth_knee_angle = -1
            right_hip_angle = -1
            right_elbow_angle = -1
            pose_correct = "Incorrect"
            progress = 0
            progress_calibration1 = 0

            if calibration != "Ok":
                calibration = "Wrong Position"

            # Draw the pose and right hand landmarks
            if results.pose_landmarks and results.right_hand_landmarks:
                pose_landmarks = results.pose_landmarks.landmark
                hand_landmarks = results.right_hand_landmarks.landmark

                required_pose_landmarks = [
                    pose_landmarks[16], pose_landmarks[20], pose_landmarks[30],
                    pose_landmarks[24], pose_landmarks[26], pose_landmarks[28],
                    pose_landmarks[12], pose_landmarks[14],pose_landmarks[25]
                ]
                
                if all(lm.visibility > 0.0 for lm in required_pose_landmarks):
   
                    # Draws the body landmarks
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                
                    # Draws the hand landmarks
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    right_hip = np.array([pose_landmarks[24].x, pose_landmarks[24].y])
                    right_knee = np.array([pose_landmarks[26].x, pose_landmarks[26].y])
                    right_ankle = np.array([pose_landmarks[28].x, pose_landmarks[28].y])
                    right_shoulder = np.array([pose_landmarks[12].x, pose_landmarks[12].y])
                    right_elbow = np.array([pose_landmarks[14].x, pose_landmarks[14].y])
                    right_wrist = np.array([pose_landmarks[16].x, pose_landmarks[16].y])

                    left_hip = np.array([pose_landmarks[23].x,pose_landmarks[23].y])
                    left_knee = np.array([pose_landmarks[25].x,pose_landmarks[25].y])
                    left_ankle = np.array([pose_landmarks[27].x,pose_landmarks[27].y])

                    rigth_knee_angle = calculate_angle(right_hip,right_knee,right_ankle)
                    left_knee_angle = calculate_angle(left_hip,left_knee,left_ankle)
                    right_hip_angle = calculate_angle(right_shoulder,right_hip,right_knee)
                    right_elbow_angle = calculate_angle(right_shoulder,right_elbow,right_wrist)

                    right_shoulder_coords = tuple(np.multiply(right_shoulder[:2], [frame.shape[1], frame.shape[0]]).astype(int))
                    right_knee_coords = tuple(np.multiply(right_knee[:2], [frame.shape[1], frame.shape[0]]).astype(int))
                    right_hip_coords = tuple(np.multiply(right_hip[:2], [frame.shape[1], frame.shape[0]]).astype(int))
                    right_ankle_coords = tuple(np.multiply(right_ankle[:2], [frame.shape[1], frame.shape[0]]).astype(int))
                    right_elbow_coords = tuple(np.multiply(right_elbow[:2], [frame.shape[1], frame.shape[0]]).astype(int))
                    right_wrist_coords = tuple(np.multiply(right_wrist[:2], [frame.shape[1], frame.shape[0]]).astype(int))

                    left_knee_coords = tuple(np.multiply(left_knee[:2], [frame.shape[1], frame.shape[0]]).astype(int))
                    left_hip_coords = tuple(np.multiply(left_hip[:2], [frame.shape[1], frame.shape[0]]).astype(int))
                    left_ankle_coords = tuple(np.multiply(left_ankle[:2], [frame.shape[1], frame.shape[0]]).astype(int))

                    cv2.putText(image, f'Left Knee Angle: {left_knee_angle:.2f}',(1000,600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)

                    draw_dynamic_angle_arc(image,right_hip_coords, right_knee_coords, right_ankle_coords, rigth_knee_angle)
                    cv2.putText(image, f'Knee Angle: {rigth_knee_angle:.2f}', right_knee_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 230, 0), 2)

                    draw_dynamic_angle_arc(image, right_shoulder_coords, right_hip_coords, right_knee_coords, right_hip_angle)
                    cv2.putText(image, f'Hip Angle: {right_hip_angle:.2f}', right_hip_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)

                    draw_dynamic_angle_arc(image, right_shoulder_coords, right_elbow_coords, right_wrist_coords, right_elbow_angle)
                    cv2.putText(image, f'Elbow Angle: {right_elbow_angle:.2f}', right_elbow_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)
                
                    if 150 < rigth_knee_angle < 180 and 120 < right_hip_angle < 150 and 90 < left_knee_angle < 120 and progress_calibration != 1.0:
                        calibration = "Right Position"
                        if calibration_time is None:
                            calibration_time = time.time()
                        progress_calibration1 = (time.time() - calibration_time) / calibration_held_duration
                        if progress_calibration1 >= 1.0:
                            progress_calibration = 1.0
                            foot_landmark = pose_landmarks[32]
                            foot = int(foot_landmark.x * 640), int((foot_landmark.y * 480) - 2) 
                            calibration = "Ok"
                    else:
                        calibration_time = None
                        progress2 = 0.0

                    if calibration == "Ok":
                        # Capture hand position
                        hand_landmark = hand_landmarks[12]  
                        hand = int(hand_landmark.x * 640), int(hand_landmark.y * 480)

                        # Calculate distance
                        dist_pixels = calculate_distance_2d(hand, foot)
                        distance = dist_pixels * PIXEL_TO_CM_RATIO  

                        distances.append(distance)
                        if len(distances) > average_over:
                            distances.pop(0)
                            distance = average_distance(distances)

                        if 170 < right_elbow_angle < 180 and 100 < right_hip_angle < 140 and 150 < rigth_knee_angle < 180 and 110 < left_knee_angle < 140:
                            pose_correct = "Correct"
                            if pose_correct_start_time is None:
                                pose_correct_start_time = time.time()
                            progress = (time.time() - pose_correct_start_time) / pose_held_duration
                            if progress >= 1.0:
                                progress = 1.0
                                final_distance = distance - 3.974
                                break
                        else:
                            pose_correct_start_time = None
                            progress = 0.0

                        # draw_dynamic_angle_arc(image,left_hip_coords,left_knee_coords,left_ankle_coords,left_knee_angle)
                        cv2.putText(image, f'Left Knee Angle: {left_knee_angle:.2f}',(1000,600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        draw_dynamic_angle_arc(image,right_hip_coords, right_knee_coords, right_ankle_coords, rigth_knee_angle)
                        cv2.putText(image, f'Knee Angle: {rigth_knee_angle:.2f}', right_knee_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        draw_dynamic_angle_arc(image, right_shoulder_coords, right_hip_coords, right_knee_coords, right_hip_angle)
                        cv2.putText(image, f'Hip Angle: {right_hip_angle:.2f}', right_hip_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        draw_dynamic_angle_arc(image, right_shoulder_coords, right_elbow_coords, right_wrist_coords, right_elbow_angle)
                        cv2.putText(image, f'Elbow Angle: {right_elbow_angle:.2f}', right_elbow_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        
                        cv2.putText(image, f"Dist: {distance:.2f} cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                        cv2.putText(image, f'Pose: {pose_correct}', (5, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(image, f'Calibration: {calibration}', (1000, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('MediaPipe Holistic', image)

        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                finish_program()
            
    # Final result visualization
    if final_distance is not None:
        final_frame = np.zeros((500, 800, 3), dtype=np.uint8) 
        
        cv2.putText(final_frame, f'Exercise Completed', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(final_frame, f'Final Distance: {final_distance :.2f} centimeters', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(final_frame,f'Press "r" to restart or "q" to finish the exercise',(50,400),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,0),2)
        
        cv2.imshow('Final Results', final_frame)

        # time.sleep(2)

        # real_value = float(input("Qual o valor real?"))
        # distance = float(distance)

        # erro = np.abs(real_value - distance)
        # nova_linha = [real_value, distance, erro]
        # sheet.append(nova_linha)
        # planilha.save(arquivo)

        print(distances)
        
        # Allow the user to close the final result window with 'q'
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):  # Press 'r' to restart the exercise
                cv2.destroyWindow('Final Results')
                break
            elif key == ord('q'):  # Press 'q' to exit
                finish_program()


    else:
        print("Exercise not performed correctly")
        finish_program()
        
