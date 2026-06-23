# =============================================================================
# ui/menu.py — Main menu screen (Centered Logo & Language Toggle)
# =============================================================================

import cv2
import numpy as np
import os

import locale_setup
from locale_setup import translate
from ui.draw import blank_canvas, draw_button, put_text, measure_text
from ui.theme import W, H, BG, BORDER_INSET, DARK_BLUE, BTN_BLUE
from utils import set_app_icon, WindowManager

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
    """Draws a minimalist Portuguese flag combining basic geometric primitives."""
    # 1. Proporção oficial de fundo: 40% Verde, 60% Vermelho
    green_w = int(w * 0.4)
    cv2.rectangle(img, (x, y), (x + green_w, y + h), (0, 102, 0), -1)       # Verde Clássico
    cv2.rectangle(img, (x + green_w, y), (x + w, y + h), (0, 0, 204), -1)   # Vermelho Clássico
    
    # 2. Centro da Esfera Armilar
    cx, cy = x + green_w, y + h // 2
    r_esfera = int(h * 0.3)
    
    # Esfera Armilar (Círculo Dourado/Oliva -> formato BGR no OpenCV)
    cv2.circle(img, (cx, cy), r_esfera, (20, 200, 220), -1) 
    
    # 3. Brasão em Branco (União de um Quadrado em cima com um Círculo em baixo)
    w_escudo = int(r_esfera * 1.1)
    r_escudo = w_escudo // 2
    
    y_topo = cy - int(r_esfera * 0.5)
    y_meio = cy + int(r_esfera * 0.1) # Ponto exato onde o quadrado termina e o círculo começa
    
    # Parte superior do brasão (Retângulo/Quadrado Branco)
    cv2.rectangle(img, (cx - r_escudo, y_topo), (cx + r_escudo, y_meio), (255, 255, 255), -1)
    
    # Parte inferior do brasão (Círculo Branco para arredondar a base)
    cv2.circle(img, (cx, y_meio), r_escudo, (255, 255, 255), -1)


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
    rects    = _button_rects()
    selected = None
    hover    = None       
    
    current_lang = "pt_PT"
    
    # Geometria da Mini-Bandeira no Canto Inferior Esquerdo
    flag_x, flag_y = 65, H - 95
    flag_w, flag_h = 60, 40
    hover_flag     = False

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
                
                locale_setup.set_language(current_lang)

    wm = WindowManager("Main Menu", size=(W, H), delay=16, on_mouse=_mouse)
    set_app_icon(wm.winname)

    while selected is None:
        img = blank_canvas()
        
        top_border_y = 80

        # Simple outer rectangle border (top edge lowered)
        cv2.rectangle(img, (BORDER_INSET, top_border_y),
                      (W - BORDER_INSET, H - BORDER_INSET), DARK_BLUE, 2)

        # Institution label — above the top border, right-aligned
        inst_tw, inst_th = measure_text("IPBeja", 18)
        img = put_text(img, "IPBeja", (W - BORDER_INSET - inst_tw, top_border_y - inst_th - 6),
                        font_size=18, color=tuple(DARK_BLUE))

        # 1. PROCESSAMENTO DE MEDIDAS E CENTRALIZAÇÃO DO BLOCO [LOGO + TEXTO]
        title_text = "CAPACITA"
        font_size = 60
        text_w, text_h = measure_text(title_text, font_size, is_bold=True)

        gap_logo_text = 5
        title_y = top_border_y - text_h // 2

        if logo_img is not None:
            logo_height_target = 100

            h_logo, w_logo = logo_img.shape[:2]
            new_w_logo = int(logo_height_target * (w_logo / h_logo))
            logo_resized = cv2.resize(logo_img, (new_w_logo, logo_height_target))
            
            logo_draw_y = top_border_y - logo_height_target // 2

            start_x = (W - 480) // 2

            pad_box = 14
            block_top = min(logo_draw_y, title_y)
            block_h = max(logo_height_target, text_h)
            cv2.rectangle(img,
                          (start_x - pad_box, block_top - pad_box),
                          (start_x + 480 + pad_box, block_top + block_h + pad_box),
                          BG, -1)
            
            try:
                roi = img[logo_draw_y:logo_draw_y+logo_height_target, start_x:start_x+new_w_logo]
                
                if logo_resized.shape[2] == 4:
                    alpha = logo_resized[:, :, 3] / 255.0
                    alpha = np.expand_dims(alpha, axis=2)
                    blended = logo_resized[:, :, :3] * alpha + roi * (1.0 - alpha)
                    img[logo_draw_y:logo_draw_y+logo_height_target, start_x:start_x+new_w_logo] = blended.astype(np.uint8)
                else:
                    logo_gray = cv2.cvtColor(logo_resized, cv2.COLOR_BGR2GRAY)
                    thresh_val, mask = cv2.threshold(logo_gray, 15, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)
                    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                    logo_fg = cv2.bitwise_and(logo_resized, logo_resized, mask=mask)
                    img[logo_draw_y:logo_draw_y+logo_height_target, start_x:start_x+new_w_logo] = cv2.add(img_bg, logo_fg)
            except Exception as e:
                print(f"Erro ao renderizar logo: {e}")
                
            text_final_x = start_x + new_w_logo + gap_logo_text
        else:
            text_final_x = (W - 480) // 2

        # Desenha o texto "CAPACITA" em Negrito
        img = put_text(img, title_text, (text_final_x, title_y), font_size=font_size, color=tuple(DARK_BLUE), is_bold=True)

        # 2. RENDERIZAÇÃO DOS BOTÕES DO MENU
        for i, (msgid, _action) in enumerate(_BUTTONS):
            bx, by, bw, bh = rects[i]
            draw_button(img, "", bx, by, bw, bh, hovered=(hover == i))
            
            texto_traduzido = translate(msgid)
            tw, th = measure_text(texto_traduzido, 22)
            text_x = (W - tw) // 2
            text_y = by + (bh - th) // 2
            img = put_text(img, texto_traduzido, (text_x, text_y), font_size=22, color=(255, 255, 255))

        # 3. RENDERIZAÇÃO DA BANDEIRA INVERSA (Alternância de Idioma)
        if current_lang == "pt_PT":
            _draw_mini_uk_flag(img, flag_x, flag_y, flag_w, flag_h)
        else:
            _draw_mini_pt_flag(img, flag_x, flag_y, flag_w, flag_h)
            
        if hover_flag:
            cv2.rectangle(img, (flag_x - 3, flag_y - 3), (flag_x + flag_w + 3, flag_y + flag_h + 3), (200, 210, 230), 2)
        else:
            cv2.rectangle(img, (flag_x, flag_y), (flag_x + flag_w, flag_y + flag_h), DARK_BLUE, 1)

        wm.show(img)
        key = wm.poll()
        if key == "close":
            selected = "quit"

    wm.close()
    return selected