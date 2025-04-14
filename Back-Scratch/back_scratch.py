from pykinect2 import PyKinectRuntime, PyKinectV2
import mediapipe as mp
import numpy as np
import cv2

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

while True:
    if kinect.has_new_color_frame():
        frame = kinect.get_last_color_frame()
        frame = frame.reshape((1080, 1920, 4))
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        rgb_frame = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        results = holistic.process(rgb_frame)
    
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    

        if results.left_hand_landmarks and results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            mp_drawing.draw_landmarks(frame,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            
        cv2.imshow('Left Hand Tracking with Kinect and Holistic', frame)

 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kinect.close()
cv2.destroyAllWindows()
