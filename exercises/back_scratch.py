# =============================================================================
# exercises/back_scratch.py
# =============================================================================
# Two repetitions per side (repeats 0-1 → right side, 2-3 → left side).
# Exercise logic (distance threshold, timing) is unchanged.
# Visual components and UI flow mirrored exactly from sit_and_reach.py.
# =============================================================================

import time
import cv2
import mediapipe as mp

from locale_setup import translate
from utils import (
    ReturnToMenu,
    WindowManager,
    calculate_distance_2d,
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
from ui.draw import put_text, measure_text
from ui.theme import W, H, HEADER_H

from config import (
    BACK_SCRATCH_PIXEL_TO_CM,
    BS_DISTANCE_THRESHOLD,
    BS_POSE_HELD_DURATION,
    BS_POSE_NO_HELD_DURATION,
    BS_ERROR,
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

def _side(repeats):
    return "right" if repeats in (0, 1) else "left"


def _draw_landmarks(image, results):
    for hand_lm in (results.left_hand_landmarks, results.right_hand_landmarks):
        mp_drawing.draw_landmarks(
            image, hand_lm, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())


def _draw_middle_finger_both(image, results):
    middle_indices = [9, 10, 11, 12]
    h, w, _ = image.shape
    color = (0, 255, 255)

    def draw_one(hand_lm):
        pts = []
        for i in middle_indices:
            x = int(hand_lm.landmark[i].x * w)
            y = int(hand_lm.landmark[i].y * h)
            pts.append((x, y))
            cv2.circle(image, (x, y), 6, color, -1)
            cv2.circle(image, (x, y), 6, (255, 255, 255), 1)
        for a, b in zip(pts, pts[1:]):
            cv2.line(image, a, b, color, 2)
        return pts[-1]  # tip (landmark 12)

    tip_left  = draw_one(results.left_hand_landmarks)  if results.left_hand_landmarks  else None
    tip_right = draw_one(results.right_hand_landmarks) if results.right_hand_landmarks else None

    if tip_left and tip_right:
        cv2.line(image, tip_left, tip_right, (0, 0, 255), 3)


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
# Core exercise loop
# =============================================================================

def run_repetition(repeats, kinect, holistic, finish_cb):
    print(f"[run_rep] início repeats={repeats}")

    start_time    = None
    last_detected = time.time()
    final_dist    = None
    elapsed       = 0.0

    side_label     = translate("right_side_label") if repeats in (0, 1) else translate("left_side_label")
    rep_num        = (repeats % 2) + 1

    wm = WindowManager("Back Scratch", finish_cb, delay=1, size=(W, H))

    while not wm.should_close:
        if not kinect.has_new_color_frame():
            continue

        image, results, _ = read_kinect_frame(kinect, holistic)
        pose_correct = translate("Incorrect")

        if results.left_hand_landmarks and results.right_hand_landmarks:
            last_detected = time.time()
            _draw_landmarks(image, results)
            _draw_middle_finger_both(image, results)
            pose_correct = translate("Correct")

            ih, iw = image.shape[:2]
            lm_left  = results.left_hand_landmarks.landmark[12]
            lm_right = results.right_hand_landmarks.landmark[12]

            left_hand  = (int(lm_left.x  * iw), int(lm_left.y  * ih))
            right_hand = (int(lm_right.x * iw), int(lm_right.y * ih))

            dist_px  = calculate_distance_2d(left_hand, right_hand)
            distance = (dist_px * BACK_SCRATCH_PIXEL_TO_CM) - BS_ERROR

            elapsed, start_time = _check_distance_timer(distance, start_time)

            if elapsed >= BS_POSE_HELD_DURATION:
                final_dist = round(-distance, 2)
                break
        else:
            if time.time() - last_detected >= BS_POSE_NO_HELD_DURATION:
                start_time = None

        # --- Infraestrutura Visual Espelhada do Sit and Reach ---
        canvas = _make_base(translate("Back Scratch exercise name"), side_label, rep_num, 2)

        # Dimensionamento Proporcional da Imagem da Câmera
        feed_h  = H - HEADER_H - 10
        feed_w  = int(image.shape[1] * feed_h / image.shape[0])
        feed_x  = (W - feed_w) // 2
        resized = cv2.resize(image, (feed_w, feed_h))
        canvas[HEADER_H + 5 : HEADER_H + 5 + feed_h, feed_x : feed_x + feed_w] = resized

        # Overlay Translúcido Superior para Caixa de Status (centrado)
        overlay = canvas.copy()
        bs_box_x = (W - 415) // 2
        cv2.rectangle(overlay, (bs_box_x, HEADER_H + 10), (bs_box_x + 415, HEADER_H + 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

        if pose_correct == translate("Correct"):
            _l1 = f"{translate('Pose')}: {pose_correct}"
            _l2 = f"{translate('Hold')}: {elapsed:.1f}s / {BS_POSE_HELD_DURATION}s" if elapsed > 0 else f"{translate('Hold')}: ---"
            _tw1, _th = measure_text(_l1, 24)
            canvas = put_text(canvas, _l1, (bs_box_x + (415 - _tw1) // 2, HEADER_H + 15), font_size=24, color=(255, 255, 255))
            _tw2, _th = measure_text(_l2, 24)
            canvas = put_text(canvas, _l2, (bs_box_x + (415 - _tw2) // 2, HEADER_H + 43), font_size=24, color=(255, 255, 255))
        else:
            _cal = f"{translate('Calibration')}: ---"
            _tw, _th = measure_text(_cal, 24)
            canvas = put_text(canvas, _cal, (bs_box_x + (415 - _tw) // 2, HEADER_H + 10 + (70 - _th) // 2), font_size=24, color=(255, 255, 255))

        wm.show(canvas)

        key = wm.poll()
        if key == "close":
            finish_cb()
            return None

    return final_dist


# =============================================================================
# Public entry point
# =============================================================================
_EXCEL_PATH = "./arquivos/tabelas_testes/back_scratch_test_orbbec.xlsx"
#_EXCEL_PATH = "./arquivos/tabelas_utentes/back_scratch_utentes.xlsx"
_LOG_PATH   = "./arquivos/logs_utentes/logs_back_scratch_orbbec"


def run(kinect, holistic, finish_cb):
    print("[BS run] início")
    try:
        print("[BS run] a chamar show_register_screen_styled...")
        age, height, weight, gender_raw = show_register_screen_styled(
            translate("Back Scratch exercise name"), translate("right_side_label"), 1, 2, finish_cb
        )
        participant = {
            "age":    age,
            "height": height,
            "weight": weight,
            "gender": "Feminine" if gender_raw.strip().upper() == "F" else "Male",
        }

        print(f"[BS run] cadastro completo: age={age}, height={height}")

        distances_right, distances_left = [], []
        reals_right, reals_left = [], []

        for rep in range(4):
            print(f"[BS run] rep={rep} a iniciar")

            show_exercise_intro("Back Scratch", rep, finish_cb)
            print(f"[BS run] rep={rep} intro concluído — a chamar run_repetition")

            dist = run_repetition(rep, kinect, holistic, finish_cb)
            print(f"[BS run] rep={rep} dist={dist}")

            if dist is None:
                print(translate("Exercise not performed correctly."))
                finish_cb()
                return

            side       = "right" if rep in (0, 1) else "left"
            side_label = translate("right_side_label") if rep in (0, 1) else translate("left_side_label")
            
            real = show_real_distance_screen_styled(
                translate("Back Scratch exercise name"), side_label, (rep % 2) + 1, 2, finish_cb
            )
            if side == "right":
                reals_right.append(real)
            else:
                reals_left.append(real)
            error = abs(abs(float(real)) - abs(dist))

            append_to_excel(_EXCEL_PATH, {
                "Age": participant["age"], "Height": participant["height"],
                "Weight": participant["weight"], "Gender": participant["gender"],
                "Side": side,
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

            show_repetition_result(
                 translate("Back Scratch exercise name"), side_label, (rep % 2) + 1, 2,
                f"{dist:.2f}", real, error, finish_cb
            )

        while len(distances_right) < 2: distances_right.append(0.0)
        while len(distances_left) < 2:  distances_left.append(0.0)
        while len(reals_right) < 2:     reals_right.append(0.0)
        while len(reals_left) < 2:      reals_left.append(0.0)

        errors_right = [abs(abs(float(reals_right[i])) - abs(float(distances_right[i]))) for i in range(2)]
        errors_left  = [abs(abs(float(reals_left[i])) - abs(float(distances_left[i]))) for i in range(2)]
        show_exercise_final(
            translate("Back Scratch exercise name"),
            f"{distances_right[0]:.2f}", f"{distances_right[1]:.2f}",
            f"{distances_left[0]:.2f}",  f"{distances_left[1]:.2f}",
            reals_right[0], reals_right[1],
            reals_left[0],  reals_left[1],
            errors_right[0], errors_right[1],
            errors_left[0],  errors_left[1],
            finish_cb
        )            
        cv2.destroyAllWindows()
        
    except ReturnToMenu:
        cv2.destroyAllWindows()
        return
    except Exception as e:
        import traceback, sys
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise