# =============================================================================
# exercises/sit_and_reach.py
# =============================================================================
# Two repetitions per leg (repeats 0-1 → right leg, 2-3 → left leg).
# Landmark indices follow MediaPipe Holistic (see config.py for reference).
# Exercise logic (angles, calibration, posture checks) is unchanged.
# =============================================================================

import time

import cv2
import mediapipe as mp
import numpy as np
from locale_setup import _
from utils import (
    calculate_distance_2d,
    read_kinect_frame,
    append_to_excel,
    append_to_log,
    show_real_distance_screen,
    put_text_utf8,
)

from config import (
    SIT_AND_REACH_PIXEL_TO_CM,
    SAR_CALIB_ELBOW_MIN, SAR_CALIB_ELBOW_MAX,
    SAR_CALIB_HIP_MIN,   SAR_CALIB_HIP_MAX,
    SAR_CALIB_KNEE_MIN,  SAR_CALIB_KNEE_MAX,
    SAR_POSTURE_ELBOW_MIN, SAR_POSTURE_ELBOW_MAX,
    SAR_POSTURE_HIP_MIN,   SAR_POSTURE_HIP_MAX,
    SAR_POSTURE_KNEE_MIN,  SAR_POSTURE_KNEE_MAX,
    SAR_OPP_ELBOW_MIN, SAR_OPP_ELBOW_MAX,
    SAR_OPP_KNEE_MIN,  SAR_OPP_KNEE_MAX,
    SAR_CALIBRATION_DURATION, SAR_POSE_DURATION,
    SAR_AVERAGE_OVER, SAR_ERROR,
)
from utils import (
    calculate_distance_2d, calculate_angle,
    rolling_average, draw_angle_arc,
    read_kinect_frame, append_to_excel, append_to_log,
    show_real_distance_screen,
)

# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic       = mp.solutions.holistic

# Landmark index sets: [shoulder, elbow, wrist, hip, knee, ankle,
#                       opp_hip, opp_knee, opp_ankle,
#                       opp_shoulder, opp_elbow, opp_wrist]
# repeats 0-1 → right leg extended (use left-side arm landmarks)
# repeats 2-3 → left  leg extended (use right-side arm landmarks)
_POSE_INDICES = {
    "right": [11, 13, 15, 23, 25, 27, 24, 26, 28, 12, 14, 16],
    "left":  [12, 14, 16, 24, 26, 28, 23, 25, 27, 11, 13, 15],
}

# Foot landmark used to anchor the reference point
_FOOT_INDEX = {"right": 31, "left": 32}

# Hand offset adjustments (pixels) per side to improve tip accuracy
_HAND_OFFSET = {"right": (+5, +8), "left": (-3, +13)}


# =============================================================================
# Private helpers
# =============================================================================

def _side(repeats):
    return "right" if repeats in (0, 1) else "left"


def _get_landmarks(results, repeats):
    """Return (pose_landmarks, hand_landmarks) or (None, None)."""
    side = _side(repeats)
    if side == "right":
        hand_result = results.left_hand_landmarks
    else:
        hand_result = results.right_hand_landmarks

    if not results.pose_landmarks or not hand_result:
        return None, None
    return results.pose_landmarks.landmark, hand_result.landmark


def _process_landmarks(results, repeats):
    """Return landmarks only when all required joints are visible."""
    pose_lm, hand_lm = _get_landmarks(results, repeats)
    if pose_lm is None:
        return None, None

    side    = _side(repeats)
    indices = _POSE_INDICES[side][:6] + _POSE_INDICES[side][9:]   # exclude opp-hip/knee/ankle for visibility check
    required = [pose_lm[i] for i in _POSE_INDICES[side]]

    if all(lm.visibility > 0.0 for lm in required):
        return pose_lm, hand_lm
    return None, None


def _draw_landmarks(image, results, repeats):
    side = _side(repeats)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    hand_lm = results.left_hand_landmarks if side == "right" else results.right_hand_landmarks
    mp_drawing.draw_landmarks(
        image, hand_lm, mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


def _calculate_angles(repeats, pose_lm):
    """
    Returns (knee, opp_knee, hip, elbow, opp_elbow) angles.
    Untouched from original logic.
    """
    idx = _POSE_INDICES[_side(repeats)]

    def pt(i):
        return np.array([pose_lm[idx[i]].x, pose_lm[idx[i]].y])

    shoulder, elbow, wrist       = pt(0), pt(1), pt(2)
    hip, knee, ankle             = pt(3), pt(4), pt(5)
    opp_hip, opp_knee, opp_ankle = pt(6), pt(7), pt(8)
    opp_shoulder, opp_elbow, opp_wrist = pt(9), pt(10), pt(11)

    return (
        calculate_angle(hip,         knee,      ankle),
        calculate_angle(opp_hip,     opp_knee,  opp_ankle),
        calculate_angle(shoulder,    hip,       knee),
        calculate_angle(shoulder,    elbow,     wrist),
        calculate_angle(opp_shoulder, opp_elbow, opp_wrist),
    )


def _draw_angle_arcs(repeats, knee, opp_knee, hip, elbow, opp_elbow,
                     pose_lm, image, frame):
    """Overlay angle arcs and text on *image*."""
    side = _side(repeats)
    idx  = _POSE_INDICES[side]

    def to_px(i):
        return tuple(np.multiply(
            [pose_lm[idx[i]].x, pose_lm[idx[i]].y],
            [frame.shape[1], frame.shape[0]]
        ).astype(int))

    sh, el, wr     = to_px(0), to_px(1), to_px(2)
    hp, kn, an     = to_px(3), to_px(4), to_px(5)
    o_sh, o_el, o_wr = to_px(9), to_px(10), to_px(11)

    draw_angle_arc(image, hp, kn, an, knee)
    cv2.putText(image, f"{_('Knee Angle')}: {knee:.1f}", kn,    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 230, 0), 2)
    draw_angle_arc(image, sh, hp, kn, hip)
    cv2.putText(image, f"{_('Hip Angle')}: {hip:.1f}",   hp,    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)
    draw_angle_arc(image, sh, el, wr, elbow)
    cv2.putText(image, f"{_('Elbow Angle')}: {elbow:.1f}", el,  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)
    draw_angle_arc(image, o_sh, o_el, o_wr, opp_elbow)
    cv2.putText(image, f"{_('Opp Elbow')}: {opp_elbow:.1f}", o_el, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)
    cv2.putText(image, f"{_('Opp Knee')}: {opp_knee:.1f}", (1000, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)


def _check_calibration(calib_time, foot, repeats,
                        knee, opp_knee, hip, elbow,
                        progress, locked, duration, pose_lm):
    """
    Returns (status, progress, calib_time, foot, locked).
    Untouched calibration logic from original.
    """
    side       = _side(repeats)
    foot_index = _FOOT_INDEX[side]

    in_position = (
        SAR_CALIB_KNEE_MIN  < knee  < SAR_CALIB_KNEE_MAX  and
        SAR_CALIB_HIP_MIN   < hip   < SAR_CALIB_HIP_MAX   and
        SAR_CALIB_ELBOW_MIN < elbow < SAR_CALIB_ELBOW_MAX and
        locked == 0.0
    )

    if in_position:
        if calib_time is None:
            calib_time = time.time()
        progress = (time.time() - calib_time) / duration
        if progress >= 1.0:
            lm   = pose_lm[foot_index]
            foot = (int(lm.x * 640), int(lm.y * 480))
            return "Ok", 1.0, calib_time, foot, 1.0
        return "Right Position", progress, calib_time, None, 0.0

    if locked == 0.0:
        return "Wrong Position", 0.0, None, None, 0.0
    return "Ok", 1.0, calib_time, foot, 1.0


def _check_posture(start_time, knee, opp_knee, hip, elbow, opp_elbow,
                   duration, progress, distance):
    """
    Returns (status, progress, start_time, final_distance_or_None).
    Untouched posture logic from original.
    """
    in_position = (
        SAR_POSTURE_ELBOW_MIN < elbow     < SAR_POSTURE_ELBOW_MAX and
        SAR_OPP_ELBOW_MIN     < opp_elbow < SAR_OPP_ELBOW_MAX     and
        SAR_POSTURE_HIP_MIN   < hip       < SAR_POSTURE_HIP_MAX   and
        SAR_POSTURE_KNEE_MIN  < knee      < SAR_POSTURE_KNEE_MAX
    )

    if in_position:
        if start_time is None:
            start_time = time.time()
        progress = (time.time() - start_time) / duration
        if progress >= 1.0:
            return "Correct", 1.0, start_time, -distance
        return "Correct", min(progress, 1.0), start_time, None

    return "Incorrect", 0.0, None, None


# =============================================================================
# UI screens
# =============================================================================

def _screen_repetition(distance, real_distance, finish_cb):
    frame = np.zeros((500, 800, 3), dtype=np.uint8)
    frame = put_text_utf8(frame, _("Repetition Completed"), (200, 100), font_size=48, color=(255, 255, 255))
    frame = put_text_utf8(frame, f"{_('Final Distance')}: {distance} cm", (100, 200), font_size=32, color=(0, 255, 0))
    frame = put_text_utf8(frame, f"{_('Real Distance')}: {real_distance} cm", (100, 250), font_size=32, color=(0, 255, 0))
    frame = put_text_utf8(frame, f"{_('Press C to continue or Q to finish')}", (50, 400), font_size=26, color=(255, 255, 0))
    cv2.imshow(_("Repetition Results"), frame)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            cv2.destroyWindow(_("Repetition Results"))
            break
        elif key == ord("q"):
            finish_cb()


def screen_final(best_right, best_left, finish_cb):
    frame = np.zeros((500, 800, 3), dtype=np.uint8)
    frame = put_text_utf8(frame, _("Exercise Completed"), (200, 100), font_size=48, color=(255, 255, 255))
    frame = put_text_utf8(frame, f"{_('Best Right Leg')}: {best_right} cm", (40, 200), font_size=32, color=(0, 255, 0))
    frame = put_text_utf8(frame, f"{_('Best Left Leg')}: {best_left} cm", (40, 270), font_size=32, color=(0, 255, 0))
    frame = put_text_utf8(frame, f"{_('C or Q')}", (200, 400), font_size=26, color=(255, 255, 0))
    cv2.imshow(_("Final Results"), frame)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            cv2.destroyWindow(_("Final Results"))
            break
        elif key == ord("q"):
            finish_cb()


# =============================================================================
# Core exercise loop
# =============================================================================

def run_repetition(repeats, kinect, holistic, finish_cb):
    """
    Run one sit-and-reach repetition and return the measured distance (str).
    *finish_cb* is called on 'q' keypress or fatal error.
    """
    calib_time   = None
    calib_locked = 0.0
    calib_prog   = 0.0
    calibration  = "Wrong Position"
    foot         = None
    pose_start   = None
    distances    = []
    final_dist   = None

    while True:
        if not kinect.has_new_color_frame():
            continue

        image, results, frame = read_kinect_frame(kinect, holistic)
        pose_correct = "Incorrect"

        pose_lm, hand_lm = _process_landmarks(results, repeats)

        if pose_lm is not None and hand_lm is not None:
            _draw_landmarks(image, results, repeats)
            angles = _calculate_angles(repeats, pose_lm)
            _draw_angle_arcs(repeats, *angles, pose_lm, image, frame)

            (calibration, calib_prog, calib_time,
             foot, calib_locked) = _check_calibration(
                calib_time, foot, repeats,
                *angles[:4], calib_prog, calib_locked,
                SAR_CALIBRATION_DURATION, pose_lm,
            )

            if calibration == "Ok":
                ox, oy = _HAND_OFFSET[_side(repeats)]
                hand = (int(hand_lm[12].x * 640) + ox,
                        int(hand_lm[12].y * 480) + oy)

                dist_px  = calculate_distance_2d(hand, foot)
                distance = dist_px * SIT_AND_REACH_PIXEL_TO_CM

                distances.append(distance)
                if len(distances) > SAR_AVERAGE_OVER:
                    distances.pop(0)
                    distance = rolling_average(distances)

                pose_correct, _, pose_start, final_dist = _check_posture(
                    pose_start, *angles, SAR_POSE_DURATION, 0, distance
                )

                if final_dist is not None:
                    side = _side(repeats)
                    if side == "right" and hand[0] < foot[0] and distance > 1.2:
                        final_dist = -(final_dist + SAR_ERROR)
                    elif side == "left" and hand[0] > foot[0] and distance > 1.2:
                        final_dist = -(final_dist + SAR_ERROR)
                    break

                cv2.putText(image, f"{_('Foot')}: {foot[0]}, {foot[1]}",  (1000, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)
                cv2.putText(image, f"{_('Hand')}: {hand[0]}, {hand[1]}",  (1000, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)
                cv2.putText(image, f"{_('Dist')}: {distance:.2f} cm",     (50,    50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,   0, 0), 2)
                cv2.putText(image, f"{_('Pose')}: {pose_correct}",        (50,   250), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 0), 2)

        cv2.putText(image, f"{_('Calibration')}: {calibration}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 0), 2)
        cv2.imshow(_("MediaPipe Holistic"), image)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            finish_cb()

    return round(final_dist, 2)


# =============================================================================
# Public entry point
# =============================================================================

_EXCEL_PATH = "./arquivos/tabelas_utentes/sit_and_reach_2_utentes.xlsx"
_LOG_PATH   = "./arquivos/logs_utentes/logs_sit_and_reach_utentes"


def run(kinect, holistic, participant, finish_cb):
    """
    Run 4 sit-and-reach repetitions (2 per leg).

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
            print(_("Exercise not performed correctly."))
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