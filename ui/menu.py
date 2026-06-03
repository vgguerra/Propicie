# =============================================================================
# ui/menu.py — Main menu screen
# =============================================================================
# Returns one of: "auto" | "sit_and_reach" | "back_scratch"
#                 "view_data" | "quit"
# based on which button the user clicks.
# =============================================================================

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

from locale_setup import _
from ui.draw import blank_canvas, draw_button
from ui.theme import W, H, BORDER_INSET, DARK_BLUE, BTN_BLUE
from ui.forms import _put_text  # Importa a função para renderizar acentos
from utils import win_title

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
    start_y = (H - total_h) // 2 + 30   # Ligeiro ajuste para dar espaço ao título limpo
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
    WIN = win_title(_("Main Menu"))
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)

    rects    = _button_rects()
    selected = None
    hover    = None   # index of hovered button

    # Carrega a fonte TrueType para calcular comprimentos exatos dos textos com acentos
    try:
        font_title = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 36)
        font_btn = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 22)
    except Exception:
        font_title = ImageFont.load_default()
        font_btn = ImageFont.load_default()

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
        
        # Desenha apenas a moldura exterior retangular azul
        card_x, card_y = 40, 40
        card_w, card_h = W - 80, H - 80
        cv2.rectangle(img, (card_x, card_y), (card_x + card_w, card_y + card_h), DARK_BLUE, 2)

        # Marca de água IPBeja no canto superior direito
        cv2.putText(img, "IPBeja", (W - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, DARK_BLUE, 1, cv2.LINE_AA)

        # 1. TÍTULO PRINCIPAL CENTRALIZADO SEM A LINHA QUEBRADA
        title_text = _("Main Menu").upper()
        # Calcula largura do título via Pillow para centralizar ao pixel
        title_w = font_title.getmask(title_text).getbbox()[2] if title_text else 0
        title_x = (W - title_w) // 2
        img = _put_text(img, title_text, (title_x, 75), font_size=36, color=tuple(DARK_BLUE))

        # 2. LOOP DE DESENHO DOS BOTÕES E TEXTOS PERFEITAMENTE CENTRALIZADOS
        for i, (msgid, _action) in enumerate(_BUTTONS):
            bx, by, bw, bh = rects[i]
            
            # Desenha a estrutura original do botão gráfica (fundo e hover)
            draw_button(img, "", bx, by, bw, bh, hovered=(hover == i))
            
            texto_traduzido = _(msgid)
            
            # MEDIÇÃO DO TEXTO DO BOTÃO:
            # Obtém a largura correta da palavra (mesmo com acentos como "Automático" ou "Coçar as Costas")
            try:
                bbox = font_btn.getmask(texto_traduzido).getbbox()
                text_w = bbox[2] if bbox else 0
                text_h = bbox[3] if bbox else 0
            except Exception:
                text_w = len(texto_traduzido) * 11  # fallback simples de segurança
                text_h = 20

            # Centralização matemática absoluta:
            # O início do X será o centro da tela (W // 2) menos metade do tamanho da palavra
            text_x = (W - text_w) // 2
            # O Y calcula a folga que sobra do botão para centrar verticalmente
            text_y = by + (bh - text_h) // 2 - 2

            # Escreve o texto com a certeza de que o meio dele está alinhado com o meio da tela
            img = _put_text(img, texto_traduzido, (text_x, text_y), font_size=22, color=(255, 255, 255))

        cv2.imshow(WIN, img)
        key = cv2.waitKey(16) & 0xFF
        if key == 27:
            selected = "quit"

    cv2.destroyWindow(WIN)
    return selected