# =============================================================================
# utils.py — Shared helpers for geometry, UI windows, and Kinect/MediaPipe I/O
# =============================================================================

import math
import datetime as dt
from locale_setup import _
import cv2
import numpy as np
import pandas as pd
from PIL import ImageFont, ImageDraw, Image

# GEOMETRY

def win_title(name):
    """Return window title with CAPACITA prefix."""
    return f"CAPACITA - {name}"

def calculate_distance_2d(point1, point2):
    """Euclidean distance between two (x, y) points."""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_angle(a, b, c):
    """
    Angle (degrees) at vertex *b* formed by segments b→a and b→c.
    Each argument is a 2-element sequence (x, y).
    """
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])

    dot   = v1[0] * v2[0] + v1[1] * v2[1]
    norm1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    norm2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    cos_a = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    return math.degrees(math.acos(cos_a))


def rolling_average(values):
    """Mean of all elements in *values*."""
    return sum(values) / len(values)


# DRAWING HELPERS

def draw_angle_arc(image, p1, p2, p3, angle):
    """
    Draw two lines (p1→p2, p2→p3) and a filled arc at vertex p2
    that spans *angle* degrees.
    """
    cv2.line(image, p1, p2, (255, 255, 0), 2)
    cv2.line(image, p2, p3, (255, 255, 0), 2)

    p1_arr = np.array(p1)
    p2_arr = np.array(p2)
    radius = int(np.linalg.norm(p1_arr - p2_arr) / 2)

    cv2.ellipse(image, tuple(p2_arr.astype(int)), (radius, radius), 0, 0, angle, (0, 255, 0), -1)
    cv2.ellipse(image, tuple(p2_arr.astype(int)), (radius, radius), 0, 0, angle, (0, 0, 255),  2)


def landmark_to_px(landmark, frame_shape):
    """Convert a normalised MediaPipe landmark to pixel coordinates (x, y)."""
    return (
        int(landmark.x * frame_shape[1]),
        int(landmark.y * frame_shape[0]),
    )

# KINECT / MEDIAPIPE FRAME PROCESSING

def read_kinect_frame(kinect, holistic):
    """
    GRAB THE LATEST KINECT COLOUR FRAME, RUN MEDIAPIPE HOLISTIC,
    AND RETURN (BGR_IMAGE, HOLISTIC_RESULTS, RAW_FRAME).
    """
    raw = kinect.get_last_color_frame()
    raw = raw.reshape((1080, 1920, 4))          # BGRA
    bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = holistic.process(rgb)
    rgb.flags.writeable = True
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), results, raw

# DATA PERSISTENCE

def append_to_excel(filepath, row_dict):
    """Append *row_dict* as a new row to an existing Excel workbook.
    Creates the file with headers if it does not exist."""
    try:
        df = pd.read_excel(filepath, engine="openpyxl")
    except FileNotFoundError:
        df = pd.DataFrame([row_dict])
        df.to_excel(filepath, index=False, engine="openpyxl")
        return
    df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    df.to_excel(filepath, index=False, engine="openpyxl")


def append_to_log(filepath, *fields):
    """Append a comma-separated log line with a leading timestamp."""
    line = ", ".join(str(f) for f in fields)
    with open(filepath, "a") as fh:
        fh.write(f"{dt.datetime.now()}, {line}\n")

# SHARED UI WINDOWS

_FONT_CACHE = {}

def _get_font_cached(font_size):
    if font_size in _FONT_CACHE:
        return _FONT_CACHE[font_size]
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    _FONT_CACHE[font_size] = font
    return font

def put_text_utf8(img, text, pos, font_size=28, color=(0, 0, 0)):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    font = _get_font_cached(font_size)
    color_rgb = (color[2], color[1], color[0])
    draw = ImageDraw.Draw(pil_img)
    draw.text(pos, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

def show_register_screen():
    """
    OpenCV registration form.
    Returns (age, height, weight, gender) as strings.
    """
    fields = [_("Age"), _("Height (cm)"), _("Weight (kg)"), _("Gender (M/F)")]
    values      = ["", "", "", ""]
    active_field = -1

    positions = [(50, 50 + i * 80, 550, 100 + i * 80) for i in range(len(fields))]

    def _mouse_cb(event, x, y, flags, param):
        nonlocal active_field
        if event == cv2.EVENT_LBUTTONDOWN:
            active_field = -1
            for i, (x1, y1, x2, y2) in enumerate(positions):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    active_field = i
                    break

    win = win_title(_("Register"))
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _mouse_cb)

    while True:
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255

        for i, (x1, y1, x2, y2) in enumerate(positions):
            cv2.rectangle(img, (x1, y1), (x2, y2), (230, 230, 230), -1)
            border = (0, 255, 0) if i == active_field else (0, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), border, 2)
            img = put_text_utf8(img, f"{fields[i]}:", (x1 + 10, y1 - 26),
                font_size=20, color=(0, 0, 0))
            img = put_text_utf8(img, values[i], (x1 + 10, y2 - 40),
              font_size=26, color=(0, 0, 0))

        img = put_text_utf8(img, _("Press Enter to confirm"), (50, 360),
            font_size=18, color=(100, 100, 100))
        cv2.imshow(win, img)

        key = cv2.waitKey(10) & 0xFF
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            _quit()
        elif key == 27:
            _quit()
        elif key in (13, 10):
            cv2.destroyAllWindows()
            return tuple(values)
        elif key == 9:
            active_field = (active_field + 1) % len(fields)
        elif active_field != -1:
            if key == 8:
                values[active_field] = values[active_field][:-1]
            elif 32 <= key <= 126:
                values[active_field] += chr(key)


def show_real_distance_screen():
    """
    OpenCV numeric-input window.
    Returns the manually measured distance as a float (cm).
    """
    entered = ""

    win = win_title(_("Real Distance"))
    cv2.namedWindow(win)

    while True:
        img = np.ones((200, 600, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (50, 60), (550, 120), (230, 230, 230), -1)
        cv2.rectangle(img, (50, 60), (550, 120), (0, 0, 0), 2)
        img = put_text_utf8(img, _("real_distance_label") + " (cm):", (50, 40),
              font_size=24, color=(0, 0, 0))
        img = put_text_utf8(img, _("Press Enter to confirm"), (50, 170),
              font_size=18, color=(100, 100, 100))
              
        cv2.imshow(win, img)

        key = cv2.waitKey(10) & 0xFF
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            _quit()
        elif key == 27:
            cv2.destroyAllWindows()
            _quit()
        elif key in (13, 10) and entered:
            cv2.destroyAllWindows()
            return float(entered.replace(",", "."))
        elif key == 8:
            entered = entered[:-1]
        elif (48 <= key <= 57) or key in (44, 46, 43, 45):
            entered += chr(key)

# INTERNAL HELPERS

def _quit():
    """Centralised exit — called by UI helpers that don't own Kinect/cv2."""
    cv2.destroyAllWindows()
    raise SystemExit(0)
