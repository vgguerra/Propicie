# =============================================================================
# ui/draw.py — Reusable drawing primitives
# =============================================================================

import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional

from ui.theme import (
    BG, DARK_BLUE, HEADER_BG, HEADER_TEXT, BTN_BLUE, BTN_TEXT,
    TEXT, W, H, BORDER_INSET, HEADER_H,
    FONT, FONT_TITLE, FONT_BTN, FONT_SMALL, FONT_LABEL,
    THICKNESS_TITLE, THICKNESS_BTN, THICKNESS_SMALL, INSTITUTION,
)

# ---------------------------------------------------------------------------
# Unicode / UTF-8 Text Rendering Helpers (Pillow Wrapper)
# ---------------------------------------------------------------------------

_FONT_CACHE = {}

def _get_font(scale):
    """Maps OpenCV scale to an approximate PIL TrueType font size with caching."""
    font_size = max(12, int(scale * 24))
    if font_size in _FONT_CACHE:
        return _FONT_CACHE[font_size]
    for font_name in ["arial.ttf", "DejaVuSans.ttf", "calibri.ttf", "LiberationSans-Regular.ttf"]:
        try:
            font = ImageFont.truetype(font_name, font_size)
            _FONT_CACHE[font_size] = font
            return font
        except IOError:
            continue
    font = ImageFont.load_default()
    _FONT_CACHE[font_size] = font
    return font


def _get_font_size(font_size, is_bold=False):
    """Return a PIL font for a given point size, with optional bold."""
    key = (font_size, is_bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    font_path = r"C:\Windows\Fonts\arialbd.ttf" if is_bold else r"C:\Windows\Fonts\arial.ttf"
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


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
    y_pil = y - th
    rgb_color = (color[2], color[1], color[0])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y_pil), text, font=font, fill=rgb_color)
    img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def put_text(img, text, pos, font_size=22, color=(0, 0, 0), is_bold=False):
    """Draw UTF-8 text onto *img* safely using NFC normalization and return the modified image."""
    text = unicodedata.normalize('NFC', str(text))
    font = _get_font_size(font_size, is_bold)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    color_rgb = (color[2], color[1], color[0])
    ImageDraw.Draw(pil_img).text(pos, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def measure_text(text, font_size=22, is_bold=False):
    """Helper to dynamically calculate text width and height for PIL rendering."""
    text = unicodedata.normalize('NFC', str(text))
    font = _get_font_size(font_size, is_bold)
    try:
        bbox = font.getmask(text).getbbox()
        if bbox:
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        pass
    return len(text) * (font_size // 2), font_size


def put_text_multi(img, items):
    """Draw multiple texts in a single PIL session for efficiency.

    *items* is a list of (text, pos, font_size, color, is_bold) tuples.
    Returns the modified image.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    for text, pos, font_size, color, is_bold in items:
        text = unicodedata.normalize('NFC', str(text))
        font = _get_font_size(font_size, is_bold)
        color_rgb = (color[2], color[1], color[0])
        draw.text(pos, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Canvas helpers
# ---------------------------------------------------------------------------

def blank_canvas() -> np.ndarray:
    """Return a fresh W×H canvas filled with the background colour."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = BG
    return img


def draw_outer_border(img: np.ndarray):
    """Draw the thin dark-blue outer border."""
    cv2.rectangle(img,
                  (BORDER_INSET, BORDER_INSET),
                  (W - BORDER_INSET, H - BORDER_INSET),
                  DARK_BLUE, 2)


def draw_institution(img: np.ndarray, extra: str = ""):
    """
    Draw the institution label top-right.
    *extra* is appended after a ' / ' separator (e.g. operator name).
    """
    label = f"{INSTITUTION} / {extra}" if extra else INSTITUTION
    tw, th = _get_text_size(label, FONT_SMALL)
    _put_text_utf8(img, label, (W - tw - 16, 30), FONT_SMALL, DARK_BLUE)


# ---------------------------------------------------------------------------
# Header bar  (pages with camera feed)
# ---------------------------------------------------------------------------

def draw_header(img: np.ndarray,
                exercise: str,
                side: str,
                rep_current: int,
                rep_total: int):
    """Draw the full-width coloured top bar seen in exercise pages."""
    from locale_setup import translate
    cv2.rectangle(img, (0, 0), (W, HEADER_H), HEADER_BG, -1)

    _trash, text_h = measure_text("A", 24, is_bold=True)
    text_y = max(0, (HEADER_H - text_h) // 2)

    img[:] = put_text(img, exercise.upper(), (20, text_y),
                      font_size=24, color=tuple(HEADER_TEXT), is_bold=True)

    tw, _trash = measure_text(side, 24, is_bold=True)
    img[:] = put_text(img, side, ((W - tw) // 2, text_y),
                      font_size=24, color=tuple(HEADER_TEXT), is_bold=True)

    rep_label = f"{translate('repetition_label')} {rep_current}/{rep_total}"
    tw, _trash = measure_text(rep_label, 24, is_bold=True)
    img[:] = put_text(img, rep_label, (W - tw - 20, text_y),
                      font_size=24, color=tuple(HEADER_TEXT), is_bold=True)


# ---------------------------------------------------------------------------
# Title page layout  (no camera — white card with border)
# ---------------------------------------------------------------------------

def draw_title_page(img: np.ndarray, title: str, institution_extra: str = ""):
    """
    Draw outer border + decorative lines flanking the title + institution label.
    Matches the 'Menu Principal' / 'Sentar e Alcançar' / 'Próximo' pages.
    """
    draw_outer_border(img)
    draw_institution(img, institution_extra)

    tw, th = _get_text_size(title, FONT_TITLE)
    tx = (W - tw) // 2
    ty = 80

    line_y   = ty - th // 2 + 5
    margin   = 30
    line_x1  = BORDER_INSET + margin
    line_x2r = tx - 30
    line_x1r = tx + tw + 30
    line_x2  = W - BORDER_INSET - margin

    cv2.line(img, (line_x1, line_y), (line_x2r, line_y), DARK_BLUE, 2)
    cv2.line(img, (line_x1r, line_y), (line_x2, line_y), DARK_BLUE, 2)

    _put_text_utf8(img, title, (tx, ty), FONT_TITLE, DARK_BLUE)


# ---------------------------------------------------------------------------
# Button
# ---------------------------------------------------------------------------

def draw_button(img: np.ndarray,
                label: str,
                x: int, y: int, w: int, h: int,
                hovered: bool = False):
    """Draw a filled rectangle button with centred label."""
    color = (110, 40, 40) if hovered else BTN_BLUE
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)

    tw, th = _get_text_size(label, FONT_BTN)
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2
    _put_text_utf8(img, label, (tx, ty), FONT_BTN, BTN_TEXT)


# ---------------------------------------------------------------------------
# Card overlay  (white panel for forms)
# ---------------------------------------------------------------------------

def draw_card(img: np.ndarray,
              x: int, y: int, w: int, h: int):
    """Draw a white card with dark-blue border."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), DARK_BLUE, 2)


def draw_card_title(img: np.ndarray, title: str,
                    x: int, y: int, w: int, h: int = 52):
    """Draw filled header inside a card (e.g. 'CADASTRO')."""
    cv2.rectangle(img, (x, y), (x + w, y + h), BTN_BLUE, -1)
    tw, _ = measure_text(title, 28, is_bold=True)
    tx = x + (w - tw) // 2
    img[:] = put_text(img, title, (tx, y + 12), font_size=28,
                      color=(255, 255, 255), is_bold=True)
    return img


# ---------------------------------------------------------------------------
# Progress circles  (exercise intro / next screens)
# ---------------------------------------------------------------------------

def draw_rep_circles(img: np.ndarray,
                     cx: int, cy: int,
                     total: int,
                     done: int,
                     radius: int = 38,
                     current: Optional[int] = None):
    if current is None:
        current = done

    gap     = radius * 2 + 24
    total_w = total * (radius * 2) + (total - 1) * 24
    start_x = cx - total_w // 2 + radius

    for i in range(total):
        cx_i = start_x + i * gap
        if i < done:
            cv2.circle(img, (cx_i, cy), radius, BTN_BLUE, -1)
            d = int(radius * 0.45)
            cv2.line(img, (cx_i - d, cy - d), (cx_i + d, cy + d), (0, 255, 0), 3)
            cv2.line(img, (cx_i + d, cy - d), (cx_i - d, cy + d), (0, 255, 0), 3)
        elif i == current:
            cv2.circle(img, (cx_i, cy), radius, (0, 255, 255), -1)
            cv2.circle(img, (cx_i, cy), radius, BTN_BLUE, 4)
        else:
            cv2.circle(img, (cx_i, cy), radius, BTN_BLUE, -1)
