# =============================================================================
# ui/exercise_intro.py — Exercise introduction / progress screen
# =============================================================================
# Shows before each repetition:
#   - Exercise name as title
#   - Two rows of circles (one per side, 2 reps each)
#   - Completed reps → filled circle with X
#   - Current rep   → hollow circle (○)
#   - Pending reps  → filled solid circle (●)
#
# Usage:
#   from ui.exercise_intro import show_exercise_intro
#   show_exercise_intro("Sit and Reach", rep=0)  # rep 0-3
# =============================================================================

import cv2
import numpy as np

from utils import ReturnToMenu, WindowManager
from locale_setup import translate
from ui.draw import (draw_button, draw_rep_circles, blank_canvas,
                     _get_text_size, _put_text_utf8)
from ui.theme import (
    W, H, BG, BORDER_INSET, BTN_BLUE, BTN_TEXT, DARK_BLUE, INSTITUTION,
    FONT_BTN, FONT_TITLE, FONT_SMALL,
)


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------

def _draw_row(img, label, cx, cy, total, done, current=-1, radius=48):
    """Draw label button + circles for one side row."""
    btn_w, btn_h = 460, 64
    bx = cx - btn_w // 2
    by = cy - btn_h // 2
    cv2.rectangle(img, (bx, by), (bx + btn_w, by + btn_h), BTN_BLUE, -1)
    
    label_scale = 1.5
    tw, th = _get_text_size(label, label_scale)
    _put_text_utf8(img, label, (bx + (btn_w - tw) // 2, by + (btn_h + th) // 2), label_scale, BTN_TEXT)

    circle_y = cy + btn_h // 2 + radius + 20
    draw_rep_circles(img, cx, circle_y, total, done, radius, current)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def show_exercise_intro(exercise_name, rep, finish_cb):
    print("[intro] início")
    
    right_done    = min(rep, 2)
    left_done     = max(rep - 2, 0)
    right_current = rep if rep < 2 else -1
    left_current  = (rep - 2) if rep >= 2 else -1
    is_next = rep > 0
    title   = translate("Next") if is_next else exercise_name
    print(f"[intro] title={title}")

    row_gap  = 180
    start_y  = (H - row_gap * 2) // 2 - 20
    cx       = W // 2

    right_label = f"{translate('right_side_label')} x2"
    left_label  = f"{translate('left_side_label')} x2"
        
    print("[intro] importado")
    print("[intro] canvas criado")

    wm = WindowManager(exercise_name, finish_cb, delay=16)
    while not wm.should_close:
        img = blank_canvas()

        # Outer border with top edge at Y=80 (como o menu)
        cv2.rectangle(img,
                      (BORDER_INSET, 80),
                      (W - BORDER_INSET, H - BORDER_INSET),
                      DARK_BLUE, 2)

        # Institution label — right-aligned with right border
        inst_tw, _ = _get_text_size(INSTITUTION, FONT_SMALL)
        _put_text_utf8(img, INSTITUTION, (W - BORDER_INSET - inst_tw, 70),
                       FONT_SMALL, DARK_BLUE)

        # Title centered on the top border line (Y=80)
        tw, th = _get_text_size(title, FONT_TITLE)
        tx = (W - tw) // 2
        ty = 80 + th // 2
        pad = 20
        cv2.rectangle(img,
                      (tx - pad, ty - th - pad),
                      (tx + tw + pad, ty + pad),
                      BG, -1)
        _put_text_utf8(img, title, (tx, ty), FONT_TITLE, DARK_BLUE)

        # Right side row
        _draw_row(img, right_label, cx, start_y + 50,
                  total=2, done=right_done, current=right_current)

        # Left side row
        _draw_row(img, left_label, cx, start_y + 80 + row_gap,
                  total=2, done=left_done, current=left_current)

        # Footer hint
        hint = translate("Press ENTER to begin") if not is_next else translate("Press ENTER to continue")
        tw, th = _get_text_size(hint, 1.3)
        _put_text_utf8(img, hint, ((W - tw) // 2, H - 75), 1.3, DARK_BLUE)

        # Bottom bar — exercise name / side / rep counter
        side_label = translate("right_side_label") if rep < 2 else translate("left_side_label")
        rep_label  = f"{translate('repetition_label')} {(rep % 2) + 1}/2"
        
        _put_text_utf8(img, exercise_name.upper(), (50, H - 20), 0.6, DARK_BLUE)
        
        tw, th = _get_text_size(side_label, 0.6)
        _put_text_utf8(img, side_label, ((W - tw) // 2, H - 20), 0.6, DARK_BLUE)
        
        tw, th = _get_text_size(rep_label, 0.6)
        _put_text_utf8(img, rep_label, (W - tw - 50, H - 20), 0.6, DARK_BLUE)

        wm.show(img)

        key = wm.poll()
        if key == "close":
            finish_cb()
            return
        elif key in (13, 10):
            wm.close()
            return
        elif key == 27:
            wm.close()
            raise ReturnToMenu()