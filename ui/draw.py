# =============================================================================
# ui/draw.py — Reusable drawing primitives
# =============================================================================

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

# Dicionário global para cachear as fontes já carregadas em memória
_FONT_CACHE = {}

def _get_font(scale):
    """Maps OpenCV scale to an approximate PIL TrueType font size with caching."""
    font_size = max(12, int(scale * 24))
    
    # Se a fonte com esse tamanho já foi carregada, retorna ela direto da memória
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


def _get_text_size(text, scale):
    """Alternative to cv2.getTextSize supporting Unicode."""
    font = _get_font(scale)
    # Usamos uma instância estática leve ou criamos apenas a draw necessária
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
    
    # Conversão eficiente de cores
    rgb_color = (color[2], color[1], color[0])
    
    # Converte o frame atual para PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    
    draw.text((x, y_pil), text, font=font, fill=rgb_color)
    
    # Atualiza o buffer original do OpenCV de forma direta
    img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


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

def draw_header_bar(img: np.ndarray,
                    exercise: str,
                    side: str,
                    rep_current: int,
                    rep_total: int):
    """
    Draw the full-width coloured top bar seen in pages 3-7.
    Layout:  [EXERCISE NAME]   [Side]   [Repetição N/T]
    """
    from locale_setup import _ as trans
    cv2.rectangle(img, (0, 0), (W, HEADER_H), HEADER_BG, -1)

    # Left — exercise name (small caps style)
    _put_text_utf8(img, exercise.upper(), (20, 40), FONT_LABEL, HEADER_TEXT)

    # Centre — side label
    side_label = side
    tw, th = _get_text_size(side_label, FONT_LABEL)
    _put_text_utf8(img, side_label, ((W - tw) // 2, 40), FONT_LABEL, HEADER_TEXT)

    # Right — repetition counter
    rep_label = f"{trans('repetition_label')} {rep_current}/{rep_total}"
    tw, th = _get_text_size(rep_label, FONT_LABEL)
    _put_text_utf8(img, rep_label, (W - tw - 20, 40), FONT_LABEL, HEADER_TEXT)


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

    # Title text
    tw, th = _get_text_size(title, FONT_TITLE)
    tx = (W - tw) // 2
    ty = 80

    # Decorative horizontal lines flanking the title
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
    color = (110, 40, 40) if hovered else BTN_BLUE   # slightly lighter on hover
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)

    tw, th = _get_text_size(label, FONT_BTN)
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2
    _put_text_utf8(img, label, (tx, ty), FONT_BTN, BTN_TEXT)


# ---------------------------------------------------------------------------
# Card overlay  (white semi-transparent panel for forms)
# ---------------------------------------------------------------------------

def draw_card(img: np.ndarray,
              x: int, y: int, w: int, h: int,
              alpha: float = 0.92):
    """Draw a white card with dark-blue border."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), DARK_BLUE, 2)


# ---------------------------------------------------------------------------
# Progress circles  (exercise intro / next screens)
# ---------------------------------------------------------------------------

def draw_rep_circles(img: np.ndarray,
                     cx: int, cy: int,
                     total: int,
                     done: int,
                     radius: int = 38,
                     current: Optional[int] = None):
    """
    Draw *total* circles horizontally centred on (cx, cy).
    Circles with index < done are filled (✗ completed),
    the circle at *current* is hollow (○ pending), rest filled (●).
    If *current* is None, defaults to *done* (first uncompleted rep).
    If *current* < 0, no hollow ring is drawn (all done or all pending).
    """
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