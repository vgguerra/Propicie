from pykinect2 import PyKinectRuntime, PyKinectV2
import mediapipe as mp
import numpy as np
import cv2
import time

# Approximate ratio of pixels to cm at 1 meter distance
PIXEL_TO_CM_RATIO = 0.625

# variable initialization
AVERAGE_OVER = 5

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

# Function to calculate average distance
def average_distance(distances):
    return sum(distances) / len(distances)

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

# Function to show the performance screen for that attempt
def final_repetition_visualization(distance):

    final_repetition_frame = np.zeros((500, 800, 3), dtype=np.uint8)

    cv2.putText(final_repetition_frame, f'Repetition Completed', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(final_repetition_frame, f"Distance between both hands: {distance} cm", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(final_repetition_frame,f'Press "c" to continue or "q" to finish the exercise',(50,400),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,0),2)
    cv2.imshow("Repetition Results", final_repetition_frame)
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            finish_program()
        elif key == ord('c'):
            cv2.destroyWindow("Repetition Results")
            break  

# Function to show the final display
def final_visualization(left,right):
    final_frame = np.zeros((500,800,3),dtype=np.uint8)

    cv2.putText(final_frame,f'Exercise completed',(200,100),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(final_frame, f'Better result of the right side: {right} cm', (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(final_frame, f'Better result of the left side: {left} cm', (40, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(final_frame,f'Press "q" to finish the exercise',(200,400),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,0),2)

    cv2.imshow("Final results",final_frame)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to exit
            finish_program() 

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

# Function to check if the hands are positioned correctly
def check_distance(distance,start_time):
    if distance < 20:
        if start_time is None:
            start_time = time.time()  
        elapsed_time = time.time() - start_time
    else:
        start_time = None  
        elapsed_time = 0
    return elapsed_time, start_time

def process_exercise(repeats):
    distances = []
    elapsed_time = None
    start_time = None 

    # Main Loop
    while True:
        if kinect.has_new_color_frame():
            image, results, frame = process_frame(kinect)

            if results.left_hand_landmarks and results.right_hand_landmarks:
                draw_landmarks(image, results)

                hand_landmark1 = results.left_hand_landmarks.landmark[12]  
                hand_landmark2 = results.right_hand_landmarks.landmark[12]

                right_hand = int(hand_landmark1.x * 640), int(hand_landmark1.y * 480)
                left_hand = int((hand_landmark2.x * 640)), int(hand_landmark2.y * 480)

                distance_pixel = calculate_distance_2d(right_hand, left_hand)
                distance = (distance_pixel * PIXEL_TO_CM_RATIO) - 0.5
                
                distances.append(distance)
                if len(distances) > AVERAGE_OVER:
                    distances.pop(0)
                    distance = average_distance(distances)

                elapsed_time,start_time = check_distance(distance,start_time) 

                cv2.putText(image, f"Dist: {distance:.2f} cm", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(image, f'Pos Right Hand: {right_hand[0]}, {right_hand[1]}', (1000, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)
                cv2.putText(image, f'Pos Left Hand: {left_hand[0]}, {left_hand[1]}', (1000, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)

                if elapsed_time >= 4:
                    if repeats in [0,1]:
                        if left_hand[1] >= right_hand[1]:
                            distance = -distance
                    else:
                        if left_hand[1] <= right_hand[1]:
                            distance = -distance    
                            
                    return f'{distance:.2f}'
                
            else:
                start_time = None  

            cv2.imshow('Left Hand Tracking with Kinect and Holistic', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                finish_program()
    
repeats = 0

distances_right = []
distances_left = []

while repeats < 4:
    final_distance = process_exercise(repeats)

    if final_distance is not None:
        if repeats in [0,1]: 
            distances_right.append(final_distance)
            side = "right"
        else: 
            distances_left.append(final_distance)
            side = "left"

        repeats += 1

        final_repetition_visualization(final_distance)
    else:
        print("Exercise not performed correctly")
        finish_program()

better_left,better_right = max(distances_left, key=float), max(distances_right, key=float)
final_visualization(better_left,better_right)

finish_program()