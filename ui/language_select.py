# =============================================================================
# ui/language_select.py — Language selection screen
# =============================================================================
# Draws two clickable flags (Portugal / UK) on an OpenCV window.
# Returns "pt_PT" or "en_US" based on the user's click.
# No external image files required — flags are drawn with OpenCV primitives.
# =============================================================================

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Design tokens (match the PDF identity)
# ---------------------------------------------------------------------------
BG_COLOR        = (240, 245, 248)   # light blue-grey background
BORDER_COLOR    = (142,  46,  46)   # dark blue border  (BGR: #2E2E8E)
TEXT_COLOR      = (142,  46,  46)   # dark blue text
HOVER_COLOR     = (200, 210, 230)   # flag hover highlight
INSTITUTION     = "IPBeja"

W, H = 1280, 720   # window dimensions


# ---------------------------------------------------------------------------
# Flag drawing helpers
# ---------------------------------------------------------------------------

def _draw_pt_flag(img, x, y, w, h):
    """Draw the Portuguese flag (simplified: green|red + yellow circle)."""
    green_w = int(w * 0.4)

    # Green stripe
    cv2.rectangle(img, (x, y), (x + green_w, y + h), (34, 139, 34), -1)
    # Red stripe
    cv2.rectangle(img, (x + green_w, y), (x + w, y + h), (0, 0, 205), -1)

    # Yellow circle (armillary sphere simplified)
    cx = x + green_w
    cy = y + h // 2
    r  = int(h * 0.28)
    cv2.circle(img, (cx, cy), r, (0, 215, 255), 3)
    cv2.circle(img, (cx, cy), int(r * 0.65), (255, 255, 255), -1)
    cv2.circle(img, (cx, cy), int(r * 0.65), (0, 0, 139), 2)
    # cross on shield
    cv2.line(img, (cx - int(r*0.3), cy), (cx + int(r*0.3), cy), (0, 0, 139), 2)
    cv2.line(img, (cx, cy - int(r*0.3)), (cx, cy + int(r*0.3)), (0, 0, 139), 2)


def _draw_uk_flag(img, x, y, w, h):
    """Draw the Union Jack (simplified geometric version)."""
    # Blue background
    cv2.rectangle(img, (x, y), (x + w, y + h), (128, 0, 0), -1)  # dark blue

    # White diagonals (St Andrew + St Patrick combined)
    thickness_diag = max(int(h * 0.18), 6)
    cv2.line(img, (x, y),         (x + w, y + h), (255, 255, 255), thickness_diag)
    cv2.line(img, (x + w, y),     (x, y + h),     (255, 255, 255), thickness_diag)

    # Red diagonals (thin)
    thickness_red_diag = max(int(h * 0.07), 3)
    cv2.line(img, (x, y),         (x + w, y + h), (0, 0, 220), thickness_red_diag)
    cv2.line(img, (x + w, y),     (x, y + h),     (0, 0, 220), thickness_red_diag)

    # White cross
    thickness_cross = max(int(h * 0.28), 10)
    cv2.line(img, (x + w//2, y), (x + w//2, y + h), (255, 255, 255), thickness_cross)
    cv2.line(img, (x, y + h//2), (x + w, y + h//2), (255, 255, 255), thickness_cross)

    # Red cross
    thickness_red = max(int(h * 0.16), 6)
    cv2.line(img, (x + w//2, y), (x + w//2, y + h), (0, 0, 220), thickness_red)
    cv2.line(img, (x, y + h//2), (x + w, y + h//2), (0, 0, 220), thickness_red)


def _draw_frame(img):
    """Draw outer border and institution label."""
    # Background
    img[:] = BG_COLOR

    # Outer border (inset 40px)
    cv2.rectangle(img, (40, 40), (W - 40, H - 40), BORDER_COLOR, 2)

    # Institution label — top right
    cv2.putText(img, INSTITUTION, (W - 130, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, TEXT_COLOR, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def show_language_select() -> str:
    """
    Display a language-selection window with two clickable flags.
    Blocks until the user clicks one flag or presses ESC (defaults to pt_PT).

    Returns
    -------
    "pt_PT" or "en_US"
    """
    WIN = "CAPACITA"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)

    # Flag geometry — smaller, pushed to lower third
    flag_w, flag_h = 220, 146
    gap            = 120
    total_w        = flag_w * 2 + gap
    start_x        = (W - total_w) // 2
    flag_y         = H // 2 + 60   # below centre

    pt_rect = (start_x,               flag_y, start_x + flag_w,         flag_y + flag_h)
    uk_rect = (start_x + flag_w + gap, flag_y, start_x + flag_w*2 + gap, flag_y + flag_h)

    selected = None
    hover    = None   # "pt" | "uk" | None

    def _mouse(event, x, y, flags, param):
        nonlocal selected, hover

        # Hover detection
        if pt_rect[0] <= x <= pt_rect[2] and pt_rect[1] <= y <= pt_rect[3]:
            hover = "pt"
        elif uk_rect[0] <= x <= uk_rect[2] and uk_rect[1] <= y <= uk_rect[3]:
            hover = "uk"
        else:
            hover = None

        # Click detection
        if event == cv2.EVENT_LBUTTONDOWN:
            if hover == "pt":
                selected = "pt_PT"
            elif hover == "uk":
                selected = "en_US"

    cv2.setMouseCallback(WIN, _mouse)

    while selected is None:
        img = np.zeros((H, W, 3), dtype=np.uint8)
        _draw_frame(img)

        # CAPACITA title — large, centred, upper half
        title  = "CAPACITA"
        font   = cv2.FONT_HERSHEY_SIMPLEX
        scale  = 4.5
        thick  = 6
        (tw, th), _bl = cv2.getTextSize(title, font, scale, thick)
        tx = (W - tw) // 2
        ty = H // 2 - 30
        cv2.putText(img, title, (tx, ty), font, scale, TEXT_COLOR, thick, cv2.LINE_AA)

        # Subtle subtitle
        sub   = "Select language / Selecione o idioma"
        (sw, sh), _bl = cv2.getTextSize(sub, font, 0.6, 1)
        cv2.putText(img, sub, ((W - sw) // 2, flag_y - 20),
                    font, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)

        # Hover highlights
        if hover == "pt":
            cv2.rectangle(img,
                          (pt_rect[0] - 10, pt_rect[1] - 10),
                          (pt_rect[2] + 10, pt_rect[3] + 10),
                          HOVER_COLOR, -1)
        if hover == "uk":
            cv2.rectangle(img,
                          (uk_rect[0] - 10, uk_rect[1] - 10),
                          (uk_rect[2] + 10, uk_rect[3] + 10),
                          HOVER_COLOR, -1)

        # Draw flags
        _draw_pt_flag(img, pt_rect[0], pt_rect[1], flag_w, flag_h)
        _draw_uk_flag(img, uk_rect[0], uk_rect[1], flag_w, flag_h)

        # Flag borders
        for rect, h_key in [(pt_rect, "pt"), (uk_rect, "uk")]:
            color = (180, 100, 60) if hover == h_key else BORDER_COLOR
            cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)

        cv2.imshow(WIN, img)

        key = cv2.waitKey(16) & 0xFF
        if key == 27:          # ESC → default to PT
            selected = "pt_PT"

    cv2.destroyWindow(WIN)
    return selected