# =============================================================================
# ui/theme.py — Design tokens shared across all screens
# =============================================================================
# Colours in BGR (OpenCV convention)
# =============================================================================

# --- Palette ---
BG          = (240, 245, 248)   # #F8F5F0 — light blue-grey background
DARK_BLUE   = (142,  46,  46)   # #2E2E8E — primary dark blue (borders, text)
MID_BLUE    = (160,  80,  80)   # #50508E — slightly lighter (hover states)
BTN_BLUE    = (139,  56,  56)   # #383894 — button fill
BTN_TEXT    = (255, 255, 255)   # white   — button label
HEADER_BG   = (139,  56,  56)   # #383894 — top bar fill
HEADER_TEXT = (255, 255, 255)   # white
TEXT        = (142,  46,  46)   # same as DARK_BLUE

# --- Layout ---
W, H          = 1280, 720   # window size
BORDER_INSET  = 40          # outer border inset (px)
HEADER_H      = 60          # top bar height (px)
INSTITUTION   = "IPBeja"

# --- Typography scale (cv2 font scale) ---
FONT       = 0   # cv2.FONT_HERSHEY_SIMPLEX
FONT_TITLE = 1.8
FONT_BTN   = 0.85
FONT_SMALL = 0.65
FONT_LABEL = 0.70
THICKNESS_TITLE = 3
THICKNESS_BTN   = 2
THICKNESS_SMALL = 1