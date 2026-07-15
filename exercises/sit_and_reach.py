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
from locale_setup import translate
from utils import (
    ReturnToMenu,
    WindowManager,
    calculate_angle,
    calculate_distance_2d,
    draw_angle_arc,
    rolling_average,
    read_kinect_frame,
    append_to_excel,
    append_to_log,
)
from ui.exercise_intro import show_exercise_intro
from ui.forms import (
    show_real_distance_screen_styled,
    show_repetition_result,
    show_exercise_final,    
    show_register_screen_styled,
    _make_base,
)
from ui.draw import put_text, put_text_multi, measure_text
from ui.theme import W, H, HEADER_H

from config import (
    SIT_AND_REACH_PIXEL_TO_CM,
    SAR_CALIB_ELBOW_MIN, SAR_CALIB_ELBOW_MAX,
    SAR_CALIB_HIP_MIN,  SAR_CALIB_HIP_MAX,
    SAR_CALIB_KNEE_MIN,  SAR_CALIB_KNEE_MAX,
    SAR_POSTURE_ELBOW_MIN, SAR_POSTURE_ELBOW_MAX,
    SAR_POSTURE_HIP_MIN,   SAR_POSTURE_HIP_MAX,
    SAR_POSTURE_KNEE_MIN,  SAR_POSTURE_KNEE_MAX,
    SAR_OPP_ELBOW_MIN, SAR_OPP_ELBOW_MAX,
    SAR_OPP_KNEE_MIN,  SAR_OPP_KNEE_MAX,
    SAR_CALIBRATION_DURATION, SAR_POSE_DURATION,
    SAR_AVERAGE_OVER, SAR_ERROR_RIGHT, SAR_ERROR_LEFT, SAR_SIGN_THRESHOLD,
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
# Original: {"right": (+5, +8), "left": (-5, +8)}
_HAND_OFFSET = {"right": (0, 0), "left": (0, 0)}


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
    """Overlay angle arcs and text on *image* using Unicode wrapper.
    Returns the modified image."""
    side = _side(repeats)
    idx  = _POSE_INDICES[side]

    def to_px(i):
        return tuple(np.multiply(
            [pose_lm[idx[i]].x, pose_lm[idx[i]].y],
            [frame.shape[1], frame.shape[0]]
        ).astype(int))

    sh, el, wr = to_px(0), to_px(1), to_px(2)
    hp, kn, an = to_px(3), to_px(4), to_px(5)
    o_sh, o_el, o_wr = to_px(9), to_px(10), to_px(11)

    draw_angle_arc(image, hp, kn, an, knee)
    draw_angle_arc(image, sh, hp, kn, hip)
    draw_angle_arc(image, sh, el, wr, elbow)
    draw_angle_arc(image, o_sh, o_el, o_wr, opp_elbow)

    return put_text_multi(image, [
        (f"{translate('Knee Angle')}: {knee:.1f}",   kn,          24, (0, 230, 0), False),
        (f"{translate('Hip Angle')}: {hip:.1f}",     hp,          24, (0, 235, 0), False),
        (f"{translate('Elbow Angle')}: {elbow:.1f}", el,          24, (0, 235, 0), False),
        (f"{translate('Opp Elbow')}: {opp_elbow:.1f}", o_el,      24, (0, 235, 0), False),
        (f"{translate('Opp Knee')}: {opp_knee:.1f}", (1000, 400), 24, (0, 235, 0), False),
    ])


def _check_calibration(calib_time, foot, repeats, iw, ih,
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
            foot = (int(lm.x * iw), int(lm.y * ih))
            return "Ok", 1.0, calib_time, foot, 1.0
        return translate("Right Position"), progress, calib_time, None, 0.0

    if locked == 0.0:
        return translate("Wrong Position"), 0.0, None, None, 0.0
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
            return translate("Correct"), 1.0, start_time, -distance
        return translate("Correct"), min(progress, 1.0), start_time, None

    return translate("Incorrect"), 0.0, None, None


# =============================================================================
# Core exercise loop
# =============================================================================

def run_repetition(repeats, kinect, holistic, finish_cb):
    """
    Run one sit-and-reach repetition and return the measured distance (str).
    *finish_cb* is called on ESC keypress or fatal error.
    """
    print(f"[run_rep] início repeats={repeats}")

    calib_time   = None
    calib_locked = 0.0
    calib_prog   = 0.0
    calibration  = translate("Wrong Position")
    foot         = None
    pose_start   = None
    distances    = []
    final_dist   = None
    display_dist = 0.0

    wm = WindowManager("Sit and Reach", finish_cb, delay=1, size=(W, H))

    while not wm.should_close:
        distance_str = ""
        if not kinect.has_new_color_frame():
            continue

        image, results, frame = read_kinect_frame(kinect, holistic)
        pose_correct = translate("Incorrect")

        pose_lm, hand_lm = _process_landmarks(results, repeats)

        if pose_lm is not None and hand_lm is not None:
            _draw_landmarks(image, results, repeats)
            angles = _calculate_angles(repeats, pose_lm)
            image = _draw_angle_arcs(repeats, *angles, pose_lm, image, frame)

            # Visual guides: foot and hand measurement points (image coords)
            side = _side(repeats)
            ih, iw = image.shape[:2]
            foot_idx = _FOOT_INDEX[side]
            foot_pt  = (int(pose_lm[foot_idx].x * iw), int(pose_lm[foot_idx].y * ih))
            hand_pt  = (int(hand_lm[12].x * iw), int(hand_lm[12].y * ih))
            cv2.circle(image, foot_pt, 12, (0, 255, 255), -1)
            cv2.circle(image, hand_pt, 12, (0, 255, 255), -1)

            (calibration, calib_prog, calib_time,
             foot, calib_locked) = _check_calibration(
                calib_time, foot, repeats, iw, ih,
                *angles[:4], calib_prog, calib_locked,
                SAR_CALIBRATION_DURATION, pose_lm,
            )

            if calibration == "Ok":
                ox, oy = _HAND_OFFSET[_side(repeats)]
                hand = (int(hand_lm[12].x * iw) + ox,
                        int(hand_lm[12].y * ih) + oy)

                dist_px  = calculate_distance_2d(hand, foot)
                distance = dist_px * SIT_AND_REACH_PIXEL_TO_CM

                distances.append(distance)
                if len(distances) > SAR_AVERAGE_OVER:
                    distances.pop(0)
                    distance = rolling_average(distances)

                pose_correct, _prog, pose_start, final_dist = _check_posture(
                    pose_start, *angles, SAR_POSE_DURATION, 0, distance
                )

                display_dist = distance
                if _side(repeats) == "right" and hand[0] < foot[0] and distance > SAR_SIGN_THRESHOLD:
                    display_dist = -(distance + SAR_ERROR_RIGHT)
                elif _side(repeats) == "left" and hand[0] > foot[0] and distance > SAR_SIGN_THRESHOLD:
                    display_dist = -(distance + SAR_ERROR_LEFT)

                if final_dist is not None:
                    if side == "right" and hand[0] < foot[0] and distance > SAR_SIGN_THRESHOLD:
                        final_dist = -(final_dist + SAR_ERROR_RIGHT)
                    elif side == "left" and hand[0] > foot[0] and distance > SAR_SIGN_THRESHOLD:
                        final_dist = -(final_dist + SAR_ERROR_LEFT)
                    break

                image = put_text(image, f"{translate('Foot')}: {foot[0]}, {foot[1]}", (1000, 100), font_size=24, color=(0, 235, 0))
                image = put_text(image, f"{translate('Hand')}: {hand[0]}, {hand[1]}", (1000, 200), font_size=24, color=(0, 235, 0))


        side_label = translate("right_side_label") if repeats in (0, 1) else translate("left_side_label")
        rep_num    = (repeats % 2) + 1  

        canvas = _make_base(translate("Sit and Reach"), side_label, rep_num, 2) 

        feed_h = H - HEADER_H - 10
        feed_w = int(image.shape[1] * feed_h / image.shape[0])
        feed_x = (W - feed_w) // 2
        resized = cv2.resize(image, (feed_w, feed_h))
        canvas[HEADER_H + 5 : HEADER_H + 5 + feed_h, feed_x : feed_x + feed_w] = resized

        overlay = canvas.copy()
        sar_box_x = (W - 415) // 2
        cv2.rectangle(overlay, (sar_box_x, HEADER_H + 10), (sar_box_x + 415, HEADER_H + 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

        if calibration == "Ok":
            _l1 = f"{translate('Pose')}: {pose_correct}"
            sign_char = "-" if display_dist < 0 else "+"
            _l2 = f"{translate('Dist')}: {sign_char}{abs(display_dist):.2f} cm"
            _tw1, _th = measure_text(_l1, 24)
            canvas = put_text(canvas, _l1, (sar_box_x + (415 - _tw1) // 2, HEADER_H + 15), font_size=24, color=(255, 255, 255))
            _tw2, _th = measure_text(_l2, 24)
            canvas = put_text(canvas, _l2, (sar_box_x + (415 - _tw2) // 2, HEADER_H + 43), font_size=24, color=(255, 255, 255))
        else:
            _cal = f"{translate('Calibration')}: {calibration}"
            _tw, _th = measure_text(_cal, 24)
            canvas = put_text(canvas, _cal, (sar_box_x + (415 - _tw) // 2, HEADER_H + 10 + (70 - _th) // 2), font_size=24, color=(255, 255, 255))

        wm.show(canvas)

        key = wm.poll()
        if key == "close":
            finish_cb()
            return None

    return round(final_dist, 2)


# =============================================================================
# Public entry point
# =============================================================================

#_EXCEL_PATH = "./arquivos/tabelas_utentes/sit_and_reach_2_utentes.xlsx"
_EXCEL_PATH = "./arquivos/tabelas_testes/sit_and_reach_test_orbbec.xlsx"
_LOG_PATH   = "./arquivos/logs_utentes/logs_sit_and_reach_orbbec"


def run(kinect, holistic, finish_cb):
    """
    Run 4 sit-and-reach repetitions (2 per leg).

    Parameters
    ----------
    kinect       : PyKinectRuntime instance
    holistic     : MediaPipe Holistic instance
    participant  : dict with keys age, height, weight, gender
    finish_cb    : callable — called to terminate the program cleanly
    """
    print("[SAR run] início")
    try:
        print("[SAR run] a chamar show_register_screen_styled...")
        age, height, weight, gender_raw = show_register_screen_styled(
            translate("Sit and Reach"), translate("right_side_label"), 1, 2, finish_cb
        )
        participant = {
            "age":    age,
            "height": height,
            "weight": weight,
            "gender": "Feminine" if gender_raw.strip().upper() == "F" else "Male",
        }

        print(f"[SAR run] cadastro completo: age={age}, height={height}")

        distances_right, distances_left = [], []
        reals_right, reals_left = [], []

        prev_dist = None
        prev_real = None
        prev_side = None

        for rep in range(4):
            print(f"[SAR run] rep={rep} a iniciar")

            show_exercise_intro("Sit and Reach", rep, finish_cb)
            print(f"[SAR run] rep={rep} intro concluído — a chamar run_repetition")

            dist = run_repetition(rep, kinect, holistic, finish_cb)
            print(f"[SAR run] rep={rep} dist={dist}")

            if dist is None:
                print(translate("Exercise not performed correctly."))
                finish_cb()
                return

            side  = "right" if rep in (0, 1) else "left"
            side_label = translate("right_side_label") if rep in (0, 1) else translate("left_side_label")
            real = show_real_distance_screen_styled(
                translate("Sit and Reach"), side_label, (rep % 2) + 1, 2, finish_cb
            )
            if side == "right":
                reals_right.append(real)
            else:
                reals_left.append(real)
            error = abs(abs(float(real)) - abs(dist))

            if rep % 2 == 1:
                for r, d, s in [(prev_real, prev_dist, prev_side), (real, dist, side)]:
                    append_to_excel(_EXCEL_PATH, {
                        "Age": participant["age"], "Height": participant["height"],
                        "Weight": participant["weight"], "Gender": participant["gender"],
                        "Side": s,
                        "Real distance": r, "Calculated distance": d, "Erro": abs(abs(float(r)) - abs(d)),
                    })
                    append_to_log(_LOG_PATH, participant["age"], participant["height"],
                                  participant["weight"], participant["gender"], r, d, s)
            else:
                prev_dist = dist
                prev_real = real
                prev_side = side

            if side == "right":
                distances_right.append(dist)
            else:
                distances_left.append(dist)

            show_repetition_result(
                translate("Sit and Reach"), side_label, (rep % 2) + 1, 2,
                f"{dist:.2f}", real, error, finish_cb
            )

        best_right = max(distances_right)
        best_left  = max(distances_left)
        errors_right = [abs(abs(float(reals_right[i])) - abs(float(distances_right[i]))) for i in range(2)]
        errors_left  = [abs(abs(float(reals_left[i])) - abs(float(distances_left[i]))) for i in range(2)]
        show_exercise_final(
            translate("Sit and Reach"),
            f"{distances_right[0]:.2f}", f"{distances_right[1]:.2f}",
            f"{distances_left[0]:.2f}",  f"{distances_left[1]:.2f}",
            reals_right[0], reals_right[1],
            reals_left[0],  reals_left[1],
            errors_right[0], errors_right[1],
            errors_left[0],  errors_left[1],
            finish_cb
        )
    except ReturnToMenu:
        cv2.destroyAllWindows()
        return
    except Exception as e:
        import traceback, sys
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise