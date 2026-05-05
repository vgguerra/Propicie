# =============================================================================
# exercises/back_scratch.py
# =============================================================================
# Two repetitions per side (repeats 0-1 → right side, 2-3 → left side).
# Exercise logic (distance threshold, timing) is unchanged.
# =============================================================================

import time

import cv2
import mediapipe as mp
import numpy as np

from config import (
    BACK_SCRATCH_PIXEL_TO_CM,
    BS_DISTANCE_THRESHOLD,
    BS_POSE_HELD_DURATION,
    BS_POSE_NO_HELD_DURATION,
    BS_ERROR,
)
from utils import (
    calculate_distance_2d,
    read_kinect_frame,
    append_to_excel,
    append_to_log,
    show_real_distance_screen,
)

# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic       = mp.solutions.holistic


# =============================================================================
# Private helpers
# =============================================================================

def _draw_landmarks(image, results):
    for hand_lm in (results.left_hand_landmarks, results.right_hand_landmarks):
        mp_drawing.draw_landmarks(
            image, hand_lm, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())


def _draw_middle_finger(image, hand_landmarks, color=(0, 255, 0)):
    """Draw only the middle-finger chain (joints 9-12)."""
    middle_indices = [9, 10, 11, 12]
    h, w, _ = image.shape

    for i in middle_indices:
        x = int(hand_landmarks.landmark[i].x * w)
        y = int(hand_landmarks.landmark[i].y * h)
        cv2.circle(image, (x, y), 5, color, -1)

    for a, b in zip(middle_indices, middle_indices[1:]):
        p1 = (int(hand_landmarks.landmark[a].x * w),
              int(hand_landmarks.landmark[a].y * h))
        p2 = (int(hand_landmarks.landmark[b].x * w),
              int(hand_landmarks.landmark[b].y * h))
        cv2.line(image, p1, p2, color, 2)


def _check_distance_timer(distance, start_time):
    """
    Accumulate time while hands are within threshold.
    Returns (elapsed_seconds, updated_start_time).
    """
    if distance < BS_DISTANCE_THRESHOLD:
        if start_time is None:
            start_time = time.time()
        return time.time() - start_time, start_time
    return 0, None


# =============================================================================
# UI screens
# =============================================================================

def _screen_repetition(distance, real_distance, finish_cb):
    frame = np.zeros((500, 800, 3), dtype=np.uint8)
    cv2.putText(frame, "Repetition Completed",                     (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(frame, f"Distance between hands: {distance} cm",   (50,  200), cv2.FONT_HERSHEY_SIMPLEX, 1,   (0, 255, 0),     2)
    cv2.putText(frame, f"Real Distance: {real_distance} cm",       (50,  250), cv2.FONT_HERSHEY_SIMPLEX, 1,   (0, 255, 0),     2)
    cv2.putText(frame, 'Press "c" to continue or "q" to finish',   (50,  400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0),   2)
    cv2.imshow("Repetition Results", frame)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            finish_cb()
        elif key == ord("c"):
            cv2.destroyWindow("Repetition Results")
            break


def screen_final(best_right, best_left, finish_cb):
    frame = np.zeros((500, 800, 3), dtype=np.uint8)
    cv2.putText(frame, "Exercise Completed",                          (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(frame, f"Best result — right side: {best_right} cm", (40,  200), cv2.FONT_HERSHEY_SIMPLEX, 1,   (0, 255, 0),     2)
    cv2.putText(frame, f"Best result — left  side: {best_left} cm",  (40,  270), cv2.FONT_HERSHEY_SIMPLEX, 1,   (0, 255, 0),     2)
    cv2.putText(frame, 'Press "q" to exit',                           (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0),   2)
    cv2.imshow("Final Results", frame)

    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            finish_cb()


# =============================================================================
# Core exercise loop
# =============================================================================

def run_repetition(repeats, kinect, holistic, finish_cb):
    """
    Run one back-scratch repetition and return the measured distance (str).
    *finish_cb* is called on 'q' keypress or fatal error.
    """
    start_time        = None
    last_detected     = time.time()

    while True:
        if not kinect.has_new_color_frame():
            continue

        image, results, _ = read_kinect_frame(kinect, holistic)

        if results.left_hand_landmarks and results.right_hand_landmarks:
            last_detected = time.time()
            _draw_landmarks(image, results)

            # Middle-finger tip (landmark 12) in scaled pixel space
            lm_left  = results.left_hand_landmarks.landmark[12]
            lm_right = results.right_hand_landmarks.landmark[12]

            left_hand  = (int(lm_left.x  * 640), int(lm_left.y  * 480))
            right_hand = (int(lm_right.x * 640), int(lm_right.y * 480))

            dist_px  = calculate_distance_2d(left_hand, right_hand)
            distance = (dist_px * BACK_SCRATCH_PIXEL_TO_CM) - BS_ERROR

            elapsed, start_time = _check_distance_timer(distance, start_time)

            cv2.putText(image, f"Dist: {distance:.2f} cm",         (50,   50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,   0,   0), 2)
            cv2.putText(image, f"Right hand: {right_hand}",        (1000, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235,   0), 2)
            cv2.putText(image, f"Left  hand: {left_hand}",         (1000, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235,   0), 2)

            if elapsed >= BS_POSE_HELD_DURATION:
                return round(-distance, 2)

        else:
            # Reset timer if hands disappear for long enough
            if time.time() - last_detected >= BS_POSE_NO_HELD_DURATION:
                start_time = None

        cv2.imshow("Back Scratch", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            finish_cb()


# =============================================================================
# Public entry point
# =============================================================================

_EXCEL_PATH = "./arquivos/tabelas_utentes/back_scratch_utentes.xlsx"
_LOG_PATH   = "./arquivos/logs_utentes/logs_back_scratch_utentes"


def run(kinect, holistic, participant, finish_cb):
    """
    Run 4 back-scratch repetitions (2 per side).

    Parameters
    ----------
    kinect       : PyKinectRuntime instance
    holistic     : MediaPipe Holistic instance
    participant  : dict with keys age, height, weight, gender
    finish_cb    : callable — called to terminate the program cleanly
    """
    distances_right, distances_left = [], []

    for rep in range(4):
        dist = run_repetition(rep, kinect, holistic, finish_cb)

        if dist is None:
            print("Exercise not performed correctly.")
            finish_cb()

        real  = show_real_distance_screen()
        error = abs(abs(float(real)) - abs(dist))
        side  = "right" if rep in (0, 1) else "left"

        append_to_excel(_EXCEL_PATH, {
            "Age": participant["age"], "Height": participant["height"],
            "Weight": participant["weight"], "Gender": participant["gender"],
            "Real distance": real, "Calculated distance": dist, "Erro": error,
        })
        append_to_log(_LOG_PATH,
                      participant["age"], participant["height"],
                      participant["weight"], participant["gender"],
                      real, dist, side)

        if side == "right":
            distances_right.append(dist)
        else:
            distances_left.append(dist)

        _screen_repetition(f"{dist:.2f}", real, finish_cb)

    best_right = max(distances_right)
    best_left  = max(distances_left)
    screen_final(f"{best_right:.2f}", f"{best_left:.2f}", finish_cb)