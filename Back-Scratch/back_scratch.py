from pykinect2 import PyKinectRuntime, PyKinectV2
import mediapipe as mp
import numpy as np
import cv2
import time

# Approximate ratio of pixels to cm at 1 meter distance
PIXEL_TO_CM_RATIO = 0.625

# Kinect initialization
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

# Inicializa o MediaPipe Holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Media Pipe Holistic initialization
def finish_program():
    cv2.destroyAllWindows()
    kinect.close()
    exit()

# Function to calculate Euclidean distance in 2D (x, y only)
def calculate_distance_2d(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to process kinect frames
def process_frame(kinect):
    frame = kinect.get_last_color_frame()
    frame = frame.reshape((1080, 1920, 4))  # BGRA
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    rgb_frame = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = holistic.process(rgb_frame)
    rgb_frame.flags.writeable = True
    return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR), results, frame

# Draw hands landmarks
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
    
# Draw only middle finger landmark
def draw_middle_finger_only(image, hand_landmarks, color=(0, 255, 0)):
    middle_indices = [9, 10, 11, 12]
    h, w, _ = image.shape

    for i in middle_indices:
        x = int(hand_landmarks.landmark[i].x * w)
        y = int(hand_landmarks.landmark[i].y * h)
        cv2.circle(image, (x, y), 5, color, -1)

    for i in range(len(middle_indices) - 1):
        x1 = int(hand_landmarks.landmark[middle_indices[i]].x * w)
        y1 = int(hand_landmarks.landmark[middle_indices[i]].y * h)
        x2 = int(hand_landmarks.landmark[middle_indices[i + 1]].x * w)
        y2 = int(hand_landmarks.landmark[middle_indices[i + 1]].y * h)
        cv2.line(image, (x1, y1), (x2, y2), color, 2)

while True:
    distance = 0
    start_time = time.time()

    # Main Loop
    while True:
        
        if kinect.has_new_color_frame():
            elapsed_time = time.time() - start_time

            if elapsed_time >= 10:
                summary_image = np.zeros((500, 800, 3), dtype=np.uint8)
                cv2.putText(summary_image, f"Distância entre as mãos: {distance:.2f} cm", (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cv2.imshow("Resumo após 10 segundos", summary_image)

                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    finish_program()
                elif key == ord('c'):
                    cv2.destroyWindow("Resumo após 10 segundos")
                    break

            image, results, frame = process_frame(kinect)

            if results.left_hand_landmarks and results.right_hand_landmarks:
                draw_landmarks(image, results)

                hand_landmark1 = results.left_hand_landmarks.landmark[12]  
                hand_landmark2 = results.right_hand_landmarks.landmark[12]

                hand1 = int(hand_landmark1.x * 640), int(hand_landmark1.y * 480)
                hand2 = int((hand_landmark2.x * 640)), int(hand_landmark2.y * 480)

                cv2.putText(image, f'Postion X and Y of hand1: {hand1[0]}, {hand1[1]}', (1000, 100),  
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)
                cv2.putText(image, f'Position X and Y of hand2: {hand2[0]}, {hand2[1]}', (1000, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)

                distance_pixel = calculate_distance_2d(hand1, hand2)
                distance = (distance_pixel * PIXEL_TO_CM_RATIO) - .5
                cv2.putText(image, f"Dist: {distance:.2f} cm", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            cv2.imshow('Left Hand Tracking with Kinect and Holistic', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                finish_program
