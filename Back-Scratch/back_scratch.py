from pykinect2 import PyKinectRuntime, PyKinectV2
import mediapipe as mp
import numpy as np
import cv2

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Function to calculate Euclidean distance in 2D (x, y only)
def calculate_distance_2d(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

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

# Function to draw the process landmarks
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())

distance = 0

while True:
    if kinect.has_new_color_frame():
        image,results,frame = process_frame(kinect)
    
        if results.left_hand_landmarks and results.right_hand_landmarks:

            draw_landmarks(image,results)

            hand_landmark1 = results.left_hand_landmarks.landmark[12]
            hand_landmark2 = results.right_hand_landmarks.landmark[12]

            hand1 = int(hand_landmark1.x * 640), int(hand_landmark1.y * 480)
            hand2 = int((hand_landmark2.x * 640) + 2), int(hand_landmark2.y * 480)

            cv2.putText(image, f'Postion X and Y of foot: {hand1[0]}, {hand1[1]}',(1000,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)
            cv2.putText(image, f'Position X and Y of hand: {hand2[0]}, {hand2[1]}',(1000,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)

            distance = calculate_distance_2d(hand1,hand2)
            cv2.putText(image, f"Dist: {distance :.2f} cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('Left Hand Tracking with Kinect and Holistic', image)

 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kinect.close()
cv2.destroyAllWindows()
