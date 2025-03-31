from pykinect2 import PyKinectRuntime, PyKinectV2
from openpyxl import load_workbook
import mediapipe as mp
import numpy as np
import time
import math
import cv2

# Approximate ratio of pixels to cm at 1 meter distance
PIXEL_TO_CM_RATIO = 0.533333  # 1 pixel ≈ 0.125 cm

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

# Function to show the final display
def final_visualization(final_distance):
    final_frame = np.zeros((500, 800, 3), dtype=np.uint8) 
    
    cv2.putText(final_frame, f'Exercise Completed', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(final_frame, f'Final Distance: {final_distance :.2f} centimeters', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(final_frame,f'Press "r" to restart or "q" to finish the exercise',(50,400),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,0),2)
    
    cv2.imshow('Final Results', final_frame)

    # real_value = float(input("Qual o valor real?"))
    # distance = float(distance)

    # erro = np.abs(real_value - distance)
    # nova_linha = [real_value, distance, erro]
    # sheet.append(nova_linha)
    # planilha.save(arquivo)
    
    # Allow the user to close the final result window with 'q'
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):  # Press 'r' to restart the exercise
            cv2.destroyWindow('Final Results')
            break
        elif key == ord('q'):  # Press 'q' to exit
            finish_program() 

# Function to process kinect frames
def process_frame(kinect):
    frame = kinect.get_last_color_frame()
    frame = frame.reshape((1080, 1920, 4))  # Kinect BGRA frame dimensions
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  
    
    # Converting to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    
    rgb_frame.flags.writeable = True
    return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR),holistic.process(rgb_frame),frame

# Function to get the landmarks
def get_landmarks(results,repeats): 
    if repeats in [0,1]:
        if not results.pose_landmarks or not results.right_hand_landmarks:
            return None, None 
        verification_hand, hand_landmarks = results.right_hand_landmarks, results.right_hand_landmarks.landmark
    else:
        if not results.pose_landmarks or not results.left_hand_landmarks:
            return None, None 
        verification_hand, hand_landmarks = results.left_hand_landmarks, results.left_hand_landmarks.landmark
    
    pose_landmarks = results.pose_landmarks.landmark

    if results.pose_landmarks and verification_hand:
        return pose_landmarks, hand_landmarks
    return None, None

# Function to check if the calibration is right
def check_calibration(calibration_time, foot, repeats, knee_angle, opposite_knee_angle, hip_angle, progress_calibration, progress_calibration1, calibration_held_duration,pose_landmarks):
    if repeats in [0, 1]: 
        foot_index = 32
    else: 
        foot_index = 31

    if 150 < knee_angle < 180 and 120 < hip_angle < 150 and 90 < opposite_knee_angle < 120 and progress_calibration1 == 0.0:
        if calibration_time is None:
            calibration_time = time.time()
        progress_calibration = (time.time() - calibration_time) / calibration_held_duration
        if progress_calibration >= 1.0:
            foot_landmark = pose_landmarks[foot_index]
            foot = int(foot_landmark.x * 640), int(foot_landmark.y * 480)
            return "Ok", 1.0, calibration_time, foot, 1.0
        return "Right Position", progress_calibration, calibration_time, None, 0.0
    if progress_calibration1 == 0.0:
        return "Wrong Position", 0.0, None, None, 0.0 if progress_calibration1 == 0.0 else ("Ok", 1.0, calibration_time, foot, 1.0)
    return "Ok", 1.0, calibration_time, foot, 1.0

# Function to check if the posture is right
def check_posture(pose_correct_start_time, knee_angle, opposite_knee_angle, hip_angle, elbow_angle, pose_held_duration, progress, distance):
    
    if 170 < elbow_angle < 180 and 70 < hip_angle < 140 and 90 < opposite_knee_angle < 140:
        if pose_correct_start_time is None:
            pose_correct_start_time = time.time()
        progress = (time.time() - pose_correct_start_time) / pose_held_duration
        if progress >= 1.0:
            final_distance = distance - 6.192
            return "Correct", min(progress, 1.0), pose_correct_start_time,final_distance
        return "Correct", min(progress, 1.0), pose_correct_start_time, None
    return "Incorrect", 0.0, None, None

# Function to calculate all the angles needed to perform the exercise
def calculate_angles(repeats, pose_landmarks):

    side = "right" if repeats in [0,1] else "left"

    pose_indices = {
        "right": [12,14,16,24,26,28,23,25,27],
        "left": [11,13,15,23,25,27,24,26,28]
    }

    indices = pose_indices[side]
    
    shoulder = np.array([pose_landmarks[indices[0]].x, pose_landmarks[indices[0]].y])
    elbow = np.array([pose_landmarks[indices[1]].x, pose_landmarks[indices[1]].y])
    wrist = np.array([pose_landmarks[indices[2]].x, pose_landmarks[indices[2]].y])
    hip = np.array([pose_landmarks[indices[3]].x, pose_landmarks[indices[3]].y])
    knee = np.array([pose_landmarks[indices[4]].x, pose_landmarks[indices[4]].y])
    ankle = np.array([pose_landmarks[indices[5]].x, pose_landmarks[indices[5]].y])
    
    opposite_hip = np.array([pose_landmarks[indices[6]].x, pose_landmarks[indices[6]].y])
    opposite_knee = np.array([pose_landmarks[indices[7]].x, pose_landmarks[indices[7]].y])
    opposite_ankle = np.array([pose_landmarks[indices[8]].x, pose_landmarks[indices[8]].y])

    return calculate_angle(hip,knee,ankle),calculate_angle(opposite_hip,opposite_knee,opposite_ankle),calculate_angle(shoulder,hip,knee), calculate_angle(shoulder,elbow,wrist)

# Function to draw all the arcs needed to analyze whether the program is working during the testing phase
def draw_angles_arcs(repeats,knee_angle, opposite_knee_angle, hip_angle, elbow_angle, pose_landmarks, image ,frame):

    side = "right" if repeats in [0,1] else "left"

    pose_indices = {
        "right": [12,14,16,24,26,28],
        "left": [11,13,15,23,25,27]
    }

    indices = pose_indices[side]

    shoulder = np.array([pose_landmarks[indices[0]].x, pose_landmarks[indices[0]].y])
    elbow = np.array([pose_landmarks[indices[1]].x, pose_landmarks[indices[1]].y])
    wrist = np.array([pose_landmarks[indices[2]].x, pose_landmarks[indices[2]].y])
    hip = np.array([pose_landmarks[indices[3]].x, pose_landmarks[indices[3]].y])
    knee = np.array([pose_landmarks[indices[4]].x, pose_landmarks[indices[4]].y])
    ankle = np.array([pose_landmarks[indices[5]].x, pose_landmarks[indices[5]].y])

    shoulder_coords = tuple(np.multiply(shoulder[:2], [frame.shape[1], frame.shape[0]]).astype(int))
    knee_coords = tuple(np.multiply(knee[:2], [frame.shape[1], frame.shape[0]]).astype(int))
    hip_coords = tuple(np.multiply(hip[:2], [frame.shape[1], frame.shape[0]]).astype(int))
    ankle_coords = tuple(np.multiply(ankle[:2], [frame.shape[1], frame.shape[0]]).astype(int))
    elbow_coords = tuple(np.multiply(elbow[:2], [frame.shape[1], frame.shape[0]]).astype(int))
    wrist_coords = tuple(np.multiply(wrist[:2], [frame.shape[1], frame.shape[0]]).astype(int))

    cv2.putText(image, f'Opposite Knee Angle: {opposite_knee_angle:.2f}',(1000,600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)

    draw_dynamic_angle_arc(image,hip_coords, knee_coords, ankle_coords, knee_angle)
    cv2.putText(image, f'Knee Angle: {knee_angle:.2f}', knee_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 230, 0), 2)

    draw_dynamic_angle_arc(image, shoulder_coords, hip_coords, knee_coords, hip_angle)
    cv2.putText(image, f'Hip Angle: {hip_angle:.2f}', hip_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)

    draw_dynamic_angle_arc(image, shoulder_coords, elbow_coords, wrist_coords, elbow_angle)
    cv2.putText(image, f'Elbow Angle: {elbow_angle:.2f}', elbow_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)

# Function to process all required landmarks
def process_landmarks(results, repeats):
    pose_landmarks, hand_landmarks = get_landmarks(results, repeats)
    if pose_landmarks is None or hand_landmarks is None:
        return None, None
    
    side = "right" if repeats in [0,1] else "left"
    pose_indices = {
        "right": [16, 20, 30, 24, 26, 28, 12, 14, 25],
        "left": [15, 19, 29, 23, 25, 27, 11, 13, 26]
    }
    required_pose_landmarks = [pose_landmarks[i] for i in pose_indices[side]]
    
    if all(lm.visibility > 0.0 for lm in required_pose_landmarks):
        return pose_landmarks, hand_landmarks
    return None, None

# Function to draw the process landmarks
def draw_landmarks(image, results, repeats):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    if repeats in [0,1]:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    else:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

# Function to process the exercise Sit and Reach
def process_exercise(repeats):
    
    # variable initialization
    calibration_held_duration = 5
    progress_calibration1 = 0
    progress_calibration = 0
    calibration = "Wrong Position"
    distances = []
    average_over = 6
    final_distance = None
    pose_correct_start_time = None
    calibration_time = None
    pose_held_duration = 8  
    foot = None

    while True:
        if kinect.has_new_color_frame():  
            image,results,frame = process_frame(kinect)

            pose_correct = "Incorrect"
            progress = 0

            pose_landmarks, hand_landmarks = process_landmarks(results, repeats)
                
            if pose_landmarks is not None and hand_landmarks is not None:

                draw_landmarks(image, results, repeats)
                    
                angles = calculate_angles(repeats,pose_landmarks)
                draw_angles_arcs(repeats, *angles, pose_landmarks, image, frame)
            
                calibration,progress_calibration,calibration_time,foot,progress_calibration1 = check_calibration(calibration_time, foot, repeats, *angles[:3],progress_calibration,progress_calibration1, calibration_held_duration, pose_landmarks)

                if calibration == "Ok":
                    # Capture hand position
                    hand_landmark = hand_landmarks[12]  
                    hand = int(hand_landmark.x * 640), int(hand_landmark.y * 480)

                    # Calculate distance
                    dist_pixels = calculate_distance_2d(hand, foot)
                    distance = dist_pixels * PIXEL_TO_CM_RATIO  

                    # Calculate average distance
                    distances.append(distance)
                    if len(distances) > average_over:
                        distances.pop(0)
                        distance = average_distance(distances)

                    pose_correct, progress, pose_correct_start_time,final_distance = check_posture(pose_correct_start_time,*angles, pose_held_duration, progress, distance)

                    if final_distance != None:
                        break

                    cv2.putText(image, f"Dist: {distance:.2f} cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    cv2.putText(image, f'Pose: {pose_correct}', (5, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
            cv2.putText(image, f'Calibration: {calibration}', (1000, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('MediaPipe Holistic', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                finish_program()

    return final_distance

# Load existing spreadsheet
arquivo = "dados.xlsx"
planilha = load_workbook(arquivo)
sheet = planilha.active

repeats = 0

while repeats < 4:
    final_distance = process_exercise(repeats)
    # Final result visualization
    if final_distance is not None:
        final_visualization(final_distance)
        repeats += 1

    else:
        print("Exercise not performed correctly")
        finish_program()

finish_program()