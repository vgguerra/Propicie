# =============================================================================
# ui/menu.py — Main menu screen (Centered Logo & Language Toggle)
# =============================================================================

import cv2
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image

import locale_setup
from locale_setup import _
from ui.draw import blank_canvas, draw_button
from ui.theme import W, H, BORDER_INSET, DARK_BLUE, BTN_BLUE
from ui.forms import _put_text  
from utils import win_title

# Mapeamento absoluto do caminho do logótipo na pasta de arquivos
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOGO_PATH = os.path.join(_BASE_DIR, "arquivos", "logo_final_green.png")

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
    # Ajustado de + 120 para + 50 para aproximar os botões do cabeçalho (Logo/Título)
    start_y = (H - total_h) // 2 + 50   
    x       = (W - _BTN_W) // 2
    rects   = []
    for i in range(len(_BUTTONS)):
        y = start_y + i * (_BTN_H + _BTN_GAP)
        rects.append((x, y, _BTN_W, _BTN_H))
    return rects


# ---------------------------------------------------------------------------
# Mini-Flag drawing helpers
# ---------------------------------------------------------------------------

def _draw_mini_pt_flag(img, x, y, w, h):
    """Draws a mini Portuguese flag using OpenCV primitives."""
    green_w = int(w * 0.4)
    cv2.rectangle(img, (x, y), (x + green_w, y + h), (34, 139, 34), -1)
    cv2.rectangle(img, (x + green_w, y), (x + w, y + h), (0, 0, 205), -1)
    cx, cy = x + green_w, y + h // 2
    r = int(h * 0.28)
    cv2.circle(img, (cx, cy), r, (0, 215, 255), 1)
    cv2.circle(img, (cx, cy), int(r * 0.65), (255, 255, 255), -1)
    cv2.circle(img, (cx, cy), int(r * 0.65), (0, 0, 139),  1)


def _draw_mini_uk_flag(img, x, y, w, h):
    """Draws a mini Union Jack flag using OpenCV primitives."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (128, 0, 0), -1)
    cv2.line(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.line(img, (x + w, y), (x, y + h), (255, 255, 255), 2)
    cv2.line(img, (x, y), (x + w, y + h), (0, 0, 220), 1)
    cv2.line(img, (x + w, y), (x, y + h), (0, 0, 220), 1)
    cv2.line(img, (x + w//2, y), (x + w//2, y + h), (255, 255, 255), 4)
    cv2.line(img, (x, y + h//2), (x + w, y + h//2), (255, 255, 255), 4)
    cv2.line(img, (x + w//2, y), (x + w//2, y + h), (0, 0, 220), 2)
    cv2.line(img, (x, y + h//2), (x + w, y + h//2), (0, 0, 220), 2)


# ---------------------------------------------------------------------------
# Public Main Menu Function
# ---------------------------------------------------------------------------

def show_main_menu() -> str:
    """
    Display the main menu and return the selected action key.
    """
    WIN = win_title(_("Main Menu"))
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)

    rects    = _button_rects()
    selected = None
    hover    = None       
    
    current_lang = "pt_PT"
    
    # Geometria da Mini-Bandeira no Canto Inferior Esquerdo
    flag_x, flag_y = 65, H - 95
    flag_w, flag_h = 60, 40
    hover_flag     = False

    # Carrega fontes TrueType (Alterado para 'arialbd.ttf' para Negrito e tamanho 90)
    try:
        font_title = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 90)
        font_btn = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 22)
    except Exception:
        font_title = ImageFont.load_default()
        font_btn = ImageFont.load_default()

    # Carrega o logótipo em modo UNCHANGED
    logo_img = cv2.imread(_LOGO_PATH, cv2.IMREAD_UNCHANGED)

    def _mouse(event, x, y, flags, param):
        nonlocal selected, hover, current_lang, hover_flag
        hover = None
        hover_flag = False
        
        for i, (bx, by, bw, bh) in enumerate(rects):
            if bx <= x <= bx + bw and by <= y <= by + bh:
                hover = i
                if event == cv2.EVENT_LBUTTONDOWN:
                    selected = _BUTTONS[i][1]
                return

        if flag_x <= x <= flag_x + flag_w and flag_y <= y <= flag_y + flag_h:
            hover_flag = True
            if event == cv2.EVENT_LBUTTONDOWN:
                current_lang = "en_US" if current_lang == "pt_PT" else "pt_PT"
                
                if hasattr(locale_setup, 'set_locale'):
                    locale_setup.set_locale(current_lang)
                elif hasattr(locale_setup, 'set_language'):
                    locale_setup.set_language(current_lang)
                elif hasattr(locale_setup, 'current_lang'):
                    locale_setup.current_lang = current_lang

    cv2.setMouseCallback(WIN, _mouse)

    while selected is None:
        img = blank_canvas()
        
        # Moldura exterior azul
        cv2.rectangle(img, (40, 40), (W - 40, H - 40), DARK_BLUE, 2)
        cv2.putText(img, "IPBeja", (W - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, DARK_BLUE, 1, cv2.LINE_AA)

        # 1. PROCESSAMENTO DE MEDIDAS E CENTRALIZAÇÃO DO BLOCO [LOGO + TEXTO]
        title_text = "CAPACITA"
        try:
            bbox = font_title.getmask(title_text).getbbox()
            text_w = bbox[2] - bbox[0] if bbox else 0
            text_h = bbox[3] - bbox[1] if bbox else 70
        except Exception:
            text_w = len(title_text) * 50
            text_h = 70

        # Alinhamento vertical do topo ajustado ligeiramente para enquadrar o tamanho novo
        title_y = 95 
        gap_logo_text = 20  # Reduzido de 40 para 20 para aproximar o logo do título
        
        if logo_img is not None:
            # Mantém o tamanho grande e imponente para o logótipo (altura de 160px)
            logo_height_target = 160 

            h_logo, w_logo = logo_img.shape[:2]
            new_w_logo = int(logo_height_target * (w_logo / h_logo))
            logo_resized = cv2.resize(logo_img, (new_w_logo, logo_height_target))
            
            # Alinhamento vertical pelos centros exatos do texto e do logo
            text_mid_y = title_y + text_h // 2
            logo_draw_y = text_mid_y - logo_height_target // 2
            logo_draw_y = max(0, logo_draw_y)

            # Largura total combinada do bloco centralizado (Horizontal)
            total_block_w = new_w_logo + gap_logo_text + text_w
            start_x = (W - total_block_w) // 2
            
            # Renderização do Logótipo com tratamento de transparência
            try:
                roi = img[logo_draw_y:logo_draw_y+logo_height_target, start_x:start_x+new_w_logo]
                
                if logo_resized.shape[2] == 4: # PNG Alpha Transparente
                    alpha = logo_resized[:, :, 3] / 255.0
                    alpha = np.expand_dims(alpha, axis=2)
                    blended = logo_resized[:, :, :3] * alpha + roi * (1.0 - alpha)
                    img[logo_draw_y:logo_draw_y+logo_height_target, start_x:start_x+new_w_logo] = blended.astype(np.uint8)
                else: # Remoção de Fundo Preto
                    logo_gray = cv2.cvtColor(logo_resized, cv2.COLOR_BGR2GRAY)
                    thresh_val, mask = cv2.threshold(logo_gray, 15, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)
                    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                    logo_fg = cv2.bitwise_and(logo_resized, logo_resized, mask=mask)
                    img[logo_draw_y:logo_draw_y+logo_height_target, start_x:start_x+new_w_logo] = cv2.add(img_bg, logo_fg)
            except Exception as e:
                print(f"Erro ao renderizar logo: {e}")
                
            # Define o início horizontal do texto colado ao logótipo
            text_final_x = start_x + new_w_logo + gap_logo_text
        else:
            text_final_x = (W - text_w) // 2

        # Desenha o texto "CAPACITA" em Negrito com tamanho 90
        img = _put_text(img, title_text, (text_final_x, title_y - 4), font_size=90, color=tuple(DARK_BLUE))

        # 2. RENDERIZAÇÃO DOS BOTÕES DO MENU
        for i, (msgid, _action) in enumerate(_BUTTONS):
            bx, by, bw, bh = rects[i]
            draw_button(img, "", bx, by, bw, bh, hovered=(hover == i))
            
            texto_traduzido = _(msgid)
            try:
                bbox_btn = font_btn.getmask(texto_traduzido).getbbox()
                btn_w = bbox_btn[2] if bbox_btn else 0
                btn_h = bbox_btn[3] if bbox_btn else 20
            except Exception:
                btn_w = len(texto_traduzido) * 11  
                btn_h = 20

            text_x = (W - btn_w) // 2
            text_y = by + (bh - btn_h) // 2 - 2
            img = _put_text(img, texto_traduzido, (text_x, text_y), font_size=22, color=(255, 255, 255))

        # 3. RENDERIZAÇÃO DA BANDEIRA INVERSA (Alternância de Idioma)
        if current_lang == "pt_PT":
            _draw_mini_uk_flag(img, flag_x, flag_y, flag_w, flag_h)
        else:
            _draw_mini_pt_flag(img, flag_x, flag_y, flag_w, flag_h)
            
        if hover_flag:
            cv2.rectangle(img, (flag_x - 3, flag_y - 3), (flag_x + flag_w + 3, flag_y + flag_h + 3), (200, 210, 230), 2)
        else:
            cv2.rectangle(img, (flag_x, flag_y), (flag_x + flag_w, flag_y + flag_h), DARK_BLUE, 1)

        cv2.imshow(WIN, img)
        key = cv2.waitKey(16) & 0xFF
        if key == 27:
            selected = "quit"

    cv2.destroyWindow(WIN)
    return selected