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
import unicodedata
from PIL import Image, ImageDraw, ImageFont

from locale_setup import _ as trans
from ui.draw import draw_title_page, draw_button, blank_canvas
from ui.theme import (  
    W, H, BTN_BLUE, BTN_TEXT, DARK_BLUE,
    FONT, FONT_BTN, FONT_LABEL, THICKNESS_BTN, THICKNESS_SMALL,
)

# ---------------------------------------------------------------------------
# Unicode / UTF-8 Text Rendering Helpers (Pillow Wrapper)
# ---------------------------------------------------------------------------

def _get_font(scale):
    """Maps OpenCV scale to an approximate PIL TrueType font size."""
    font_size = max(12, int(scale * 24))
    # Tries common system fonts to ensure cross-platform compatibility
    for font_name in ["arial.ttf", "DejaVuSans.ttf", "calibri.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(font_name, font_size)
        except IOError:
            continue
    return ImageFont.load_default()


def _get_text_size(text, scale):
    """Alternative to cv2.getTextSize supporting Unicode."""
    font = _get_font(scale)
    canvas = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _put_text_utf8(img, text, org, scale, color):
    """Alternative to cv2.putText that correctly renders accents and UTF-8."""
    font = _get_font(scale)
    tw, th = _get_text_size(text, scale)
    x, y = org
    # Adjusts OpenCV's bottom-left baseline to PIL's top-left coordinate
    y_pil = y - th
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    
    # Converts BGR (OpenCV) to RGB (PIL)
    rgb_color = (color[2], color[1], color[0])
    draw.text((x, y_pil), text, font=font, fill=rgb_color)
    
    # Mutates the original OpenCV image buffer in-place
    img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def _remove_accents(text):
    """Fallback utility to strip accents if external functions fail."""
    return "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


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
    *done* = number of completed reps (show X)
    *current* = index of current rep (show ○), -1 if none active yet
    """
    # Label button
    btn_w, btn_h = 380, 52
    bx = cx - btn_w // 2
    by = cy - btn_h // 2
    cv2.rectangle(img, (bx, by), (bx + btn_w, by + btn_h), BTN_BLUE, -1)
    
    tw, th = _get_text_size(label, FONT_BTN)
    _put_text_utf8(img, label, (bx + (btn_w - tw) // 2, by + (btn_h + th) // 2), FONT_BTN, BTN_TEXT)

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
    title   = trans("Next") if is_next else exercise_name
    print(f"[intro] title={title}")

    row_gap  = 180
    start_y  = (H - row_gap * 2) // 2 - 20
    cx       = W // 2

    right_label = f"{trans('right_side_label')} x2"
    left_label  = f"{trans('left_side_label')} x2"
        
    print("[intro] importado")
    print("[intro] canvas criado")

    while True:
        img = blank_canvas()

        # Title + decorative lines + institution label
        draw_title_page(img, _remove_accents(title))

        # Right side row
        _draw_row(img, right_label, cx, start_y + 80,
                  total=2, done=right_done, current=right_current)

        # Left side row
        _draw_row(img, left_label, cx, start_y + 80 + row_gap,
                  total=2, done=left_done, current=left_current)

        # Footer hint — Tamanho alterado para o dobro (1.3) e posição ajustada para H - 75
        hint = trans("Press SPACE to begin") if not is_next else trans("Press SPACE to continue")
        tw, th = _get_text_size(hint, 1.3)
        _put_text_utf8(img, hint, ((W - tw) // 2, H - 75), 1.3, DARK_BLUE)

        # Bottom bar — exercise name / side / rep counter
        side_label = trans("right_side_label") if rep < 2 else trans("left_side_label")
        rep_label  = f"{trans('repetition_label')} {(rep % 2) + 1}/2"
        
        _put_text_utf8(img, exercise_name.upper(), (50, H - 20), 0.6, DARK_BLUE)
        
        tw, th = _get_text_size(side_label, 0.6)
        _put_text_utf8(img, side_label, ((W - tw) // 2, H - 20), 0.6, DARK_BLUE)
        
        tw, th = _get_text_size(rep_label, 0.6)
        _put_text_utf8(img, rep_label, (W - tw - 50, H - 20), 0.6, DARK_BLUE)

        cv2.imshow(WIN, img)

        key = cv2.waitKey(16) & 0xFF
        if key == ord(" "):
            cv2.destroyWindow(WIN)
            return
        elif key == 27:
            finish_cb()