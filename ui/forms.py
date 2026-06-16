import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

from locale_setup import _
from ui.theme import (
    BG, DARK_BLUE, HEADER_BG, HEADER_TEXT, BTN_BLUE, BTN_TEXT,
    W, H, HEADER_H, FONT, FONT_LABEL, THICKNESS_SMALL,
)
from utils import win_title, ReturnToMenu

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _put_text(img, text, pos, font_size=22, color=(0, 0, 0)):
    """Draw UTF-8 text onto *img* and return the modified image."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    color_rgb = (color[2], color[1], color[0])
    ImageDraw.Draw(pil_img).text(pos, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def _measure_text(text, font_size=22, is_bold=False):
    """Helper to dynamically calculate text width and height for PIL rendering."""
    font_path = r"C:\Windows\Fonts\arialbd.ttf" if is_bold else r"C:\Windows\Fonts\arial.ttf"
    try:
        font = ImageFont.truetype(font_path, font_size)
        bbox = font.getmask(text).getbbox()
        if bbox:
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        pass
    return len(text) * (font_size // 2), font_size


def _draw_header(img, exercise, side, rep_current, rep_total):
    """Draw the coloured top bar using safe UTF-8 rendering."""
    cv2.rectangle(img, (0, 0), (W, HEADER_H), HEADER_BG, -1)

    font_size = 24
    _trash, text_h = _measure_text("A", font_size, is_bold=True)
    text_y = max(0, (HEADER_H - text_h) // 2)

    # Left — exercise name
    img[:] = _put_text(img, exercise.upper(), (20, text_y), font_size=font_size, color=tuple(HEADER_TEXT))

    # Centre — side
    tw, _trash = _measure_text(side, font_size, is_bold=True)
    img[:] = _put_text(img, side, ((W - tw) // 2, text_y), font_size=font_size, color=tuple(HEADER_TEXT))

    # Right — repetition
    rep_label = f"{_('repetition_label')} {rep_current}/{rep_total}"
    tw, _trash = _measure_text(rep_label, font_size, is_bold=True)
    img[:] = _put_text(img, rep_label, (W - tw - 20, text_y), font_size=font_size, color=tuple(HEADER_TEXT))


def _draw_card(img, x, y, w, h):
    """Draw a white card with dark-blue border."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), DARK_BLUE, 2)


def _draw_card_title(img, title, x, y, w, h=52):
    """Draw filled header inside card (e.g. 'CADASTRO')."""
    cv2.rectangle(img, (x, y), (x + w, y + h), BTN_BLUE, -1)
    img = _put_text(img, title, (x + 20, y + 10), font_size=28, color=(255, 255, 255))
    return img


def _make_base(exercise, side, rep_current, rep_total):
    """Return a fresh canvas with background and header."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = BG
    _draw_header(img, exercise, side, rep_current, rep_total)
    return img


# ---------------------------------------------------------------------------
# Register screen
# ---------------------------------------------------------------------------

def show_register_screen_styled(exercise, side, rep_current, rep_total):
    """
    Styled registration form with header bar and white card.
    Returns (age, height, weight, gender) as strings.
    """
    print("[register] início")
    try:
        fields = [_("Age"), _("Height (cm)"), _("Weight (kg)"), _("Gender (M/F)")]
        print("[register] fields criados")
        values = ["", "", "", ""]
        active_field = -1

        # Geometria da Janela (Redimensionada para acolher confortavelmente as fontes grandes)
        card_w, card_h = 600, 640
        win_w, win_h = card_w, card_h

        field_x = 40
        field_w = card_w - 80
        field_h = 48
        title_h = 58
        
        # Alinhamento vertical dos blocos [Label + Caixa de Input]
        fields_start = title_h + 35
        block_gap = 120  # Margem confortável entre um conjunto e outro

        positions = []
        label_y_positions = []
        
        for i in range(len(fields)):
            y_start = fields_start + i * block_gap
            label_y_positions.append(y_start)
            
            x1 = field_x
            y1 = y_start + 32  # Afastamento da caixa para não sobrepor o texto da label
            x2 = field_x + field_w
            y2 = y1 + field_h
            positions.append((x1, y1, x2, y2))

        WIN = win_title(_("Register"))
        print(f"[register] WIN={WIN}")

        def _mouse_cb(event, x, y, flags, param):
            nonlocal active_field
            if event == cv2.EVENT_LBUTTONDOWN:
                active_field = -1
                for i, (x1, y1, x2, y2) in enumerate(positions):
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        active_field = i
                        break

        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        print("[register] janela criada")
        cv2.setMouseCallback(WIN, _mouse_cb)

        while True:
            # Cria a lona de fundo interna (Branca)
            img = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            img[:] = (255, 255, 255)

            # Desenha a moldura exterior (Borda Azul nas extremidades exatas da janela)
            cv2.rectangle(img, (0, 0), (win_w - 1, win_h - 1), DARK_BLUE, 2)

            # Barra de título interior (CADASTRO)
            img = _draw_card_title(img, _("Register").upper(), 0, 0, card_w, title_h)

            # Desenho das Labels em Negrito e Caixas de Texto
            for i, (x1, y1, x2, y2) in enumerate(positions):
                
                # Renderização manual da Label com Arial Bold (Tamanho 25)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                draw = ImageDraw.Draw(pil_img)
                try:
                    f_lbl = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 25)
                except Exception:
                    try:
                        f_lbl = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 25)
                    except Exception:
                        f_lbl = ImageFont.load_default()
                
                color_rgb = (DARK_BLUE[2], DARK_BLUE[1], DARK_BLUE[0])
                draw.text((x1, label_y_positions[i]), f"{fields[i]}:", font=f_lbl, fill=color_rgb)
                img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

                # Desenha o retângulo interior da caixa de texto
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
                border = BTN_BLUE if i == active_field else DARK_BLUE
                cv2.rectangle(img, (x1, y1), (x2, y2), border, 2)
                
                # Texto digitado pelo utilizador
                img = _put_text(img, values[i], (x1 + 12, y1 + 10), font_size=22, color=(0, 0, 0))

            # Centralização Horizontal Dinâmica da instrução inferior
            hint_text = _("Press Enter to confirm")
            hint_font_size = 22
            
            text_w, _trash = _measure_text(hint_text, hint_font_size)
            
            # Cálculo do X central e posicionamento no rodapé interno
            hint_x = (win_w - text_w) // 2
            hint_y = win_h - 45
            
            img = _put_text(img, hint_text, (hint_x, hint_y), font_size=hint_font_size, color=tuple(DARK_BLUE))

            cv2.imshow(WIN, img)

            key = cv2.waitKey(10) & 0xFF
            if key == 27:
                cv2.destroyWindow(WIN)
                raise ReturnToMenu()
            elif key in (13, 10):
                cv2.destroyWindow(WIN)
                return tuple(values)
            elif key == 9:
                active_field = (active_field + 1) % len(fields)
            elif active_field != -1:
                if key == 8:
                    values[active_field] = values[active_field][:-1]
                elif 32 <= key <= 126:
                    values[active_field] += chr(key)
    except Exception as e:
        import traceback
        print(f"[register] ERRO: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Real distance screen
# ---------------------------------------------------------------------------

def show_real_distance_screen_styled(exercise, side, rep_current, rep_total):
    """
    Styled real-distance input with header bar and white card.
    Returns the manually measured distance as a float (cm).
    """
    entered = ""

    # Card geometry
    card_w, card_h = 580, 260
    title_h   = 58

    win_h = card_h + 80
    win_w = card_w + 80

    card_x_local = 40
    card_y_local = 10
    card_x = (W - card_w) // 2
    card_y = HEADER_H + (H - HEADER_H - card_h) // 2

    field_x_local = card_x_local + 30
    field_w_local = card_w - 60
    field_y_local = card_y_local + title_h + 50

    title_h   = 58
    field_x   = card_x + 30
    field_w   = card_w - 60
    field_y   = card_y + title_h + 50
    field_h   = 60

    WIN = win_title(_("Real Measurement"))
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    while True:
        img = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        img[:] = BG

        # Card
        _draw_card(img, card_x_local, card_y_local, card_w, card_h)

        img = _draw_card_title(img, _("Real Measurement").upper(),
                               card_x_local, card_y_local, card_w, title_h)
        img = _put_text(img, f"{_('real_distance_label')} (cm):",
                    (field_x_local, card_y_local + title_h + 16),
                    font_size=20, color=tuple(DARK_BLUE))

        # Input field
        cv2.rectangle(img, (field_x_local, field_y_local),
                      (field_x_local + field_w_local, field_y_local + field_h),
                      (255, 255, 255), -1)
        cv2.rectangle(img, (field_x_local, field_y_local),
                      (field_x_local + field_w_local, field_y_local + field_h),
                      DARK_BLUE, 2)
        if not entered:
            img = _put_text(img, "+3.33 / -3.33",
                            (field_x_local + 12, field_y_local + 12),
                            font_size=24, color=(180, 190, 200))
        else:
            img = _put_text(img, entered,
                            (field_x_local + 12, field_y_local + 12),
                            font_size=28, color=(0, 0, 200))

        # Hint
        img = _put_text(img, _("Press Enter to confirm"),
                        (field_x_local, field_y_local + field_h + 14),
                        font_size=16, color=(100, 100, 100))

        cv2.imshow(WIN, img)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            cv2.destroyWindow(WIN)
            raise ReturnToMenu()
        elif key in (13, 10) and entered:
            cv2.destroyWindow(WIN)
            return float(entered.replace(",", "."))
        elif key == 8:
            entered = entered[:-1]
        elif (48 <= key <= 57) or key in (44, 46, 43, 45):
            entered += chr(key)


# ---------------------------------------------------------------------------
# Repetition result screen
# ---------------------------------------------------------------------------

def show_repetition_result(exercise, side, rep_current, rep_total,
                           system_dist, real_dist, finish_cb):
    # Janela compacta
    win_w = 760
    win_h = 580

    card_w = win_w - 60
    card_h = win_h - 100
    card_x = 30
    card_y = 10

    title_h   = 58
    block_h   = 48
    value_h   = 70
    block_gap = 24
    content_x = card_x + 40
    content_w = card_w - 80

    # Y positions for each block
    sys_title_y  = card_y + title_h + 20
    sys_value_y  = sys_title_y + block_h + 8
    real_title_y = sys_value_y + value_h + block_gap
    real_value_y = real_title_y + block_h + 8

    WIN = win_title(_("Repetition Completed"))
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    while True:
        img = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        img[:] = BG

        _draw_card(img, card_x, card_y, card_w, card_h)
        img = _draw_card_title(img, _("Repetition Completed").upper(),
                               card_x, card_y, card_w, title_h)

        # System distance
        cv2.rectangle(img, (content_x, sys_title_y),
                      (content_x + content_w, sys_title_y + block_h), BTN_BLUE, -1)
        img = _put_text(img, _("System Measurement"),
                        (content_x + 16, sys_title_y + 10), font_size=22, color=(255, 255, 255))

        cv2.rectangle(img, (content_x, sys_value_y),
                      (content_x + content_w, sys_value_y + value_h), (255, 255, 255), -1)
        cv2.rectangle(img, (content_x, sys_value_y),
                      (content_x + content_w, sys_value_y + value_h), DARK_BLUE, 2)
        img = _put_text(img, f"{system_dist} cm",
                        (content_x + 16, sys_value_y + 16), font_size=28, color=tuple(DARK_BLUE))

        # Real distance
        cv2.rectangle(img, (content_x, real_title_y),
                      (content_x + content_w, real_title_y + block_h), BTN_BLUE, -1)
        img = _put_text(img, _("Real Measurement Result"),
                        (content_x + 16, real_title_y + 10), font_size=22, color=(255, 255, 255))

        cv2.rectangle(img, (content_x, real_value_y),
                      (content_x + content_w, real_value_y + value_h), (255, 255, 255), -1)
        cv2.rectangle(img, (content_x, real_value_y),
                      (content_x + content_w, real_value_y + value_h), DARK_BLUE, 2)
        img = _put_text(img, f"{real_dist:.2f} cm",
                        (content_x + 16, real_value_y + 16), font_size=28, color=tuple(DARK_BLUE))

        # Footer hint
        hint_y = card_y + card_h + 12
        img = _put_text(img, _("space_esc"),
                        ((win_w - 420) // 2, hint_y), font_size=16, color=tuple(DARK_BLUE))

        # Bottom bar — Corrigido para suportar acentos e caracteres especiais de tradução
        font_size_bottom = 16
        _, text_h_bottom = _measure_text("A", font_size_bottom)
        bottom_y = win_h - text_h_bottom - 12
        
        rep_label = f"{_('repetition_label')} {rep_current}/{rep_total}"
        
        img = _put_text(img, exercise.upper(), (20, bottom_y), font_size=font_size_bottom, color=tuple(DARK_BLUE))
        
        tw_side, _trash = _measure_text(side, font_size_bottom)
        img = _put_text(img, side, ((win_w - tw_side) // 2, bottom_y), font_size=font_size_bottom, color=tuple(DARK_BLUE))
        
        tw_rep, _trash = _measure_text(rep_label, font_size_bottom)
        img = _put_text(img, rep_label, (win_w - tw_rep - 20, bottom_y), font_size=font_size_bottom, color=tuple(DARK_BLUE))

        cv2.imshow(WIN, img)

        key = cv2.waitKey(16) & 0xFF
        if key == ord(" "):
            cv2.destroyWindow(WIN)
            return
        elif key == 27:
            cv2.destroyWindow(WIN)
            raise ReturnToMenu()


# ---------------------------------------------------------------------------
# Exercise final result screen
# ---------------------------------------------------------------------------

def show_exercise_final(exercise,
                        system_right_1, system_right_2,
                        system_left_1,  system_left_2,
                        real_right_1,   real_right_2,
                        real_left_1,    real_left_2,
                        finish_cb):

    WIN = win_title(_("Exercise Completed"))
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    # Layout - Otimizado para fontes maiores e melhor preenchimento
    card_x, card_y = 40, 40
    card_w, card_h = W - 80, H - 120
    title_h   = 58
    sec_h     = 48   # secção lado
    rep_h     = 28   # subtítulo repetição
    box_h     = 140  # AUMENTADO: Caixa de valores maior (antes era 110)
    gap       = 12   # Ajustado para equilibrar o espaço vertical reconstruído
    col_w     = (card_w - 80) // 2   # largura de cada coluna
    col1_x    = card_x + 40
    col2_x    = col1_x + col_w + 20

    # Y de cada secção calculados dinamicamente com base no novo tamanho
    right_sec_y  = card_y + title_h + gap
    right_rep_y  = right_sec_y + sec_h + 4
    right_box_y  = right_rep_y + rep_h + 2
    left_sec_y   = right_box_y + box_h + gap + 4
    left_rep_y   = left_sec_y + sec_h + 4
    left_box_y   = left_rep_y + rep_h + 2

    while True:
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:] = BG

        # Borda exterior
        cv2.rectangle(img, (card_x, card_y), (card_x + card_w, card_y + card_h), DARK_BLUE, 2)

        # IPBeja — Renderizado de forma segura com _put_text
        img = _put_text(img, "IPBeja", (W - 110, 12), font_size=18, color=tuple(DARK_BLUE))

        # Título principal seguro contra caracteres UTF-8
        title_text = _("Exercise Completed")
        title_font_size = 36
        tw, th = _measure_text(title_text, title_font_size, is_bold=True)
        tx = (W - tw) // 2
        ty = card_y + 12
        line_y = ty + th // 2 + 5
        cv2.line(img, (card_x + 20, line_y), (tx - 20, line_y), DARK_BLUE, 2)
        cv2.line(img, (tx + tw + 20, line_y), (card_x + card_w - 20, line_y), DARK_BLUE, 2)
        img = _put_text(img, title_text, (tx, ty), font_size=title_font_size, color=tuple(DARK_BLUE))

        # ── Secção Lado Direito ──
        cv2.rectangle(img, (col1_x, right_sec_y), (col1_x + col_w * 2 + 20, right_sec_y + sec_h), BTN_BLUE, -1)
        lbl_right = _("right_side_label")
        tw, th = _measure_text(lbl_right, 24, is_bold=True)
        img = _put_text(img, lbl_right, (col1_x + (col_w * 2 + 20 - tw) // 2, right_sec_y + (sec_h - th) // 2),
                        font_size=24, color=(255, 255, 255))

        # Subtítulos repetições direito
        for ci, label in enumerate([_("rep_1_label"), _("rep_2_label")]):
            cx = col1_x if ci == 0 else col2_x
            tw, th = _measure_text(label, 18, is_bold=True)
            img = _put_text(img, label, (cx + (col_w - tw) // 2, right_rep_y + (rep_h - th) // 2),
                            font_size=18, color=tuple(DARK_BLUE))

        # Caixas valores direito
        for ci, (sys_v, real_v) in enumerate([(system_right_1, real_right_1), (system_right_2, real_right_2)]):
            cx = col1_x if ci == 0 else col2_x
            box_y = right_box_y
            cv2.rectangle(img, (cx, box_y), (cx + col_w, box_y + box_h), (255, 255, 255), -1)
            cv2.rectangle(img, (cx, box_y), (cx + col_w, box_y + box_h), DARK_BLUE, 2)
            
            img = _put_text(img, f"{_('system_distance_label')}: {sys_v} cm",
                            (cx + 16, box_y + 20), font_size=26, color=tuple(DARK_BLUE))
            img = _put_text(img, f"{_('real_distance_label')}: {real_v:.2f} cm",
                            (cx + 16, box_y + 80), font_size=26, color=tuple(DARK_BLUE))
                                
        # ── Secção Lado Esquerdo ──
        cv2.rectangle(img, (col1_x, left_sec_y), (col1_x + col_w * 2 + 20, left_sec_y + sec_h), BTN_BLUE, -1)
        lbl_left = _("left_side_label")
        tw, th = _measure_text(lbl_left, 24, is_bold=True)
        img = _put_text(img, lbl_left, (col1_x + (col_w * 2 + 20 - tw) // 2, left_sec_y + (sec_h - th) // 2),
                        font_size=24, color=(255, 255, 255))

        # Subtítulos repetições esquerdo
        for ci, label in enumerate([_("rep_1_label"), _("rep_2_label")]):
            cx = col1_x if ci == 0 else col2_x
            tw, th = _measure_text(label, 18, is_bold=True)
            img = _put_text(img, label, (cx + (col_w - tw) // 2, left_rep_y + (rep_h - th) // 2),
                            font_size=18, color=tuple(DARK_BLUE))

        # Caixas valores esquerdo
        for ci, (sys_v, real_v) in enumerate([(system_left_1, real_left_1), (system_left_2, real_left_2)]):
            cx = col1_x if ci == 0 else col2_x
            box_y = left_box_y
            cv2.rectangle(img, (cx, box_y), (cx + col_w, box_y + box_h), (255, 255, 255), -1)
            cv2.rectangle(img, (cx, box_y), (cx + col_w, box_y + box_h), DARK_BLUE, 2)
            
            img = _put_text(img, f"{_('system_distance_label')}: {sys_v} cm",
                            (cx + 16, box_y + 20), font_size=26, color=tuple(DARK_BLUE))
            img = _put_text(img, f"{_('real_distance_label')}: {real_v:.2f} cm",
                            (cx + 16, box_y + 80), font_size=26, color=tuple(DARK_BLUE))

        # Exercise label base — Seguro contra acentos
        ex_upper = exercise.upper()
        tw, th = _measure_text(ex_upper, 20, is_bold=True)
        img = _put_text(img, ex_upper, ((W - tw) // 2, card_y + card_h + 15),
                        font_size=20, color=tuple(DARK_BLUE))

        # Hint — Seguro contra acentos
        hint_exit = _("Press ESC to exit")
        tw, th = _measure_text(hint_exit, 16)
        img = _put_text(img, hint_exit, ((W - tw) // 2, card_y + card_h + 45),
                        font_size=16, color=tuple(DARK_BLUE))

        cv2.imshow(WIN, img)
        key = cv2.waitKey(16) & 0xFF
        if key == 27: 
            cv2.destroyWindow(WIN)
            break