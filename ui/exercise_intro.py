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

from locale_setup import _
from ui.draw import draw_title_page, draw_button, blank_canvas
from ui.theme import (  
    W, H, BTN_BLUE, BTN_TEXT, DARK_BLUE,
    FONT, FONT_BTN, FONT_LABEL, THICKNESS_BTN, THICKNESS_SMALL,
)

# ---------------------------------------------------------------------------
# Circle helpers
# ---------------------------------------------------------------------------

def _draw_circle_done(img, cx, cy, r):
    """Filled circle with X — completed repetition."""
    cv2.circle(img, (cx, cy), r, BTN_BLUE, -1)
    d = int(r * 0.45)
    cv2.line(img, (cx - d, cy - d), (cx + d, cy + d), (255, 255, 255), 3)
    cv2.line(img, (cx + d, cy - d), (cx - d, cy + d), (255, 255, 255), 3)


def _draw_circle_current(img, cx, cy, r):
    """Hollow ring — current repetition."""
    cv2.circle(img, (cx, cy), r, BTN_BLUE, 4)


def _draw_circle_pending(img, cx, cy, r):
    """Filled solid circle — pending repetition."""
    cv2.circle(img, (cx, cy), r, BTN_BLUE, -1)


def _draw_row(img, label, cx, cy, total, done, current, radius=42):
    """
    Draw label button + circles for one side row.
    *done*    = number of completed reps (show X)
    *current* = index of current rep (show ○), -1 if none active yet
    """
    # Label button
    btn_w, btn_h = 380, 52
    bx = cx - btn_w // 2
    by = cy - btn_h // 2
    cv2.rectangle(img, (bx, by), (bx + btn_w, by + btn_h), BTN_BLUE, -1)
    (tw, th), _bl = cv2.getTextSize(label, FONT, FONT_BTN, THICKNESS_BTN)
    cv2.putText(img, label, (bx + (btn_w - tw) // 2, by + (btn_h + th) // 2),
                FONT, FONT_BTN, BTN_TEXT, THICKNESS_BTN, cv2.LINE_AA)

    # Circles below the label
    gap     = radius * 2 + 28
    total_w = total * radius * 2 + (total - 1) * 28
    start_x = cx - total_w // 2 + radius
    circle_y = cy + btn_h // 2 + radius + 18

    for i in range(total):
        cx_i = start_x + i * gap
        if i < done:
            _draw_circle_done(img, cx_i, circle_y, radius)
        elif i == current:
            _draw_circle_current(img, cx_i, circle_y, radius)
        else:
            _draw_circle_pending(img, cx_i, circle_y, radius)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def show_exercise_intro(exercise_name, rep, finish_cb, is_back_scratch=False):
    print("[intro] início")
    WIN = exercise_name
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    print("[intro] janela criada")
    
    right_done    = min(rep, 2)
    left_done     = max(rep - 2, 0)
    right_current = rep if rep < 2 else -1
    left_current  = (rep - 2) if rep >= 2 else -1
    is_next = rep > 0
    title   = _("Next") if is_next else exercise_name
    print(f"[intro] title={title}")

    row_gap  = 180
    start_y  = (H - row_gap * 2) // 2 - 20
    cx       = W // 2

    right_label = f"{_('right_side_label')} x2"
    left_label  = f"{_('left_side_label')} x2"
        
    print("[intro] importado")
    print("[intro] canvas criado")

    while True:
        img = blank_canvas()

        # Title + decorative lines + institution label
        draw_title_page(img, title)

        # Right side row
        _draw_row(img, right_label, cx, start_y + 80,
                  total=2, done=right_done, current=right_current)

        # Left side row
        _draw_row(img, left_label, cx, start_y + 80 + row_gap,
                  total=2, done=left_done, current=left_current)

        # Footer hint
        hint = _("Press SPACE to begin") if not is_next else _("Press SPACE to continue")
        (tw, th), _bl = cv2.getTextSize(hint, FONT, 0.65, 1)
        cv2.putText(img, hint, ((W - tw) // 2, H - 55),
                    FONT, 0.65, DARK_BLUE, 1, cv2.LINE_AA)

        # Bottom bar — exercise name / side / rep counter
        side_label = _("right_side_label") if rep < 2 else _("left_side_label")
        rep_label  = f"{_('repetition_label')} {(rep % 2) + 1}/2"
        cv2.putText(img, exercise_name.upper(), (50, H - 20),
                    FONT, 0.6, DARK_BLUE, 1, cv2.LINE_AA)
        (tw, th), _bl = cv2.getTextSize(side_label, FONT, 0.6, 1)
        cv2.putText(img, side_label, ((W - tw) // 2, H - 20),
                    FONT, 0.6, DARK_BLUE, 1, cv2.LINE_AA)
        (tw, th), _bl = cv2.getTextSize(rep_label, FONT, 0.6, 1)
        cv2.putText(img, rep_label, (W - tw - 50, H - 20),
                    FONT, 0.6, DARK_BLUE, 1, cv2.LINE_AA)

        cv2.imshow(WIN, img)

        key = cv2.waitKey(16) & 0xFF
        if key == ord(" "):
            cv2.destroyWindow(WIN)
            return
        elif key == 27:
            finish_cb()