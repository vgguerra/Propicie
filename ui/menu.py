# =============================================================================
# ui/menu.py — Main menu screen
# =============================================================================
# Returns one of: "auto" | "sit_and_reach" | "back_scratch"
#                 "view_data" | "quit"
# based on which button the user clicks.
# =============================================================================

import cv2
import numpy as np

from locale_setup import _
from ui.draw import blank_canvas, draw_title_page, draw_button
from ui.theme import W, H, BORDER_INSET, FONT, FONT_SMALL, THICKNESS_SMALL, DARK_BLUE


# ---------------------------------------------------------------------------
# Button definitions — (msgid, action_key)
# ---------------------------------------------------------------------------
_BUTTONS = [
    ("Automatic",           "auto"),
    ("Sit and Reach",       "sit_and_reach"),
    ("Back Scratch exercise name", "back_scratch"),
    ("Visualize Data",      "view_data"),
    ("End Session",         "quit"),
]

_BTN_W   = 480
_BTN_H   = 58
_BTN_GAP = 22


def _button_rects():
    """Return (x, y, w, h) for each button, vertically centred in the card."""
    total_h = len(_BUTTONS) * _BTN_H + (len(_BUTTONS) - 1) * _BTN_GAP
    start_y = (H - total_h) // 2 + 20   # slight offset to clear title
    x       = (W - _BTN_W) // 2
    rects   = []
    for i in range(len(_BUTTONS)):
        y = start_y + i * (_BTN_H + _BTN_GAP)
        rects.append((x, y, _BTN_W, _BTN_H))
    return rects


def show_main_menu() -> str:
    """
    Display the main menu and return the selected action key.
    Blocks until the user clicks a button or presses ESC (→ 'quit').
    """
    WIN = _("Main Menu")
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)

    rects    = _button_rects()
    selected = None
    hover    = None   # index of hovered button

    def _mouse(event, x, y, flags, param):
        nonlocal selected, hover
        hover = None
        for i, (bx, by, bw, bh) in enumerate(rects):
            if bx <= x <= bx + bw and by <= y <= by + bh:
                hover = i
                if event == cv2.EVENT_LBUTTONDOWN:
                    selected = _BUTTONS[i][1]
                break

    cv2.setMouseCallback(WIN, _mouse)

    while selected is None:
        img = blank_canvas()
        draw_title_page(img, _("Main Menu"))

        for i, (msgid, _action) in enumerate(_BUTTONS):
            bx, by, bw, bh = rects[i]
            draw_button(img, _(msgid), bx, by, bw, bh, hovered=(hover == i))

        cv2.imshow(WIN, img)
        key = cv2.waitKey(16) & 0xFF
        if key == 27:
            selected = "quit"

    cv2.destroyWindow(WIN)
    return selected