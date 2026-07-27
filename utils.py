# =============================================================================
# utils.py — Shared helpers for geometry, UI windows, and Kinect/MediaPipe I/O
# =============================================================================

import math
import datetime as dt
import cv2
import numpy as np
import pandas as pd


class ReturnToMenu(Exception):
    """Raised when user presses ESC to return to main menu."""
    pass

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


# KINECT / MEDIAPIPE FRAME PROCESSING

def read_kinect_frame(kinect, holistic):
    """
    GRAB THE LATEST COLOUR FRAME, RUN MEDIAPIPE HOLISTIC,
    AND RETURN (BGR_IMAGE, HOLISTIC_RESULTS, RAW_FRAME).
    """
    bgra = kinect.get_last_color_frame()             # (H, W, 4) BGRA
    bgra = cv2.flip(bgra, 1)                         # espelho horizontal (Kinect V2 behaviour)
    bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = holistic.process(rgb)
    rgb.flags.writeable = True
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), results, bgra

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

# INTERNAL HELPERS

class WindowManager:
    """Handles OpenCV window lifecycle: create, show, poll key, close on ESC/X."""

    def __init__(self, title, finish_cb=None, delay=1, size=None, on_mouse=None):
        self.winname = title if title.startswith("CAPACITA") else win_title(title)
        self.finish_cb = finish_cb
        self.delay = delay
        self._closed = False

        cv2.namedWindow(self.winname, cv2.WINDOW_NORMAL)
        if size:
            cv2.resizeWindow(self.winname, size[0], size[1])
        if on_mouse:
            cv2.setMouseCallback(self.winname, on_mouse)

    @property
    def should_close(self):
        return self._closed

    def show(self, canvas):
        cv2.imshow(self.winname, canvas)

    def poll(self):
        key = cv2.waitKey(self.delay) & 0xFF
        if key == 27 or cv2.getWindowProperty(self.winname, cv2.WND_PROP_VISIBLE) < 1:
            self.close()
            return "close"
        return key

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                cv2.destroyWindow(self.winname)
            except:
                pass

def set_app_icon(window_title=None):
    """Set the CAPACITA logo as the icon for a specific or all OpenCV windows."""
    import ctypes, os
    _base = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(_base, "arquivos", "icone-Photoroom.png")
    ico_path = os.path.join(_base, "arquivos", "icone-Photoroom.ico")
    if not os.path.exists(ico_path):
        from PIL import Image as PilImage
        PilImage.open(png_path).save(ico_path, format="ICO", sizes=[(32, 32), (48, 48), (64, 64)])
    hicon = ctypes.windll.user32.LoadImageW(0, ico_path, 1, 0, 0, 0x00000010)
    if hicon:
        if window_title:
            hwnd = ctypes.windll.user32.FindWindowW(None, window_title)
        else:
            hwnd = ctypes.windll.user32.FindWindowW("Main HighGUI class", None)
        if hwnd:
            ctypes.windll.user32.SetClassLongW(hwnd, -14, hicon)
            ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 0, hicon)
            ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 1, hicon)



