# =============================================================================
# ui/forms.py — Register and Real Distance screens with new design
# =============================================================================
# Both screens follow the PDF design:
#   - Full-width coloured header bar (exercise | side | repetition)
#   - White card centred on a light-blue background
#   - Input fields inside the card
# =============================================================================

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

from locale_setup import _
from ui.theme import (
    BG, DARK_BLUE, HEADER_BG, HEADER_TEXT, BTN_BLUE, BTN_TEXT,
    W, H, HEADER_H, FONT, FONT_LABEL, THICKNESS_SMALL,
)
from utils import win_title

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


def _draw_header(img, exercise, side, rep_current, rep_total):
    """Draw the coloured top bar."""
    cv2.rectangle(img, (0, 0), (W, HEADER_H), HEADER_BG, -1)

    # Left — exercise name
    cv2.putText(img, exercise.upper(), (20, 40),
                FONT, FONT_LABEL, HEADER_TEXT, THICKNESS_SMALL, cv2.LINE_AA)

    # Centre — side
    (tw, th), _th = cv2.getTextSize(side, FONT, FONT_LABEL, THICKNESS_SMALL)
    cv2.putText(img, side, ((W - tw) // 2, 40),
                FONT, FONT_LABEL, HEADER_TEXT, THICKNESS_SMALL, cv2.LINE_AA)

    # Right — repetition
    rep_label = f"{_('repetition_label')} {rep_current}/{rep_total}"
    (tw, th), _th = cv2.getTextSize(rep_label, FONT, FONT_LABEL, THICKNESS_SMALL)
    cv2.putText(img, rep_label, (W - tw - 20, 40),
                FONT, FONT_LABEL, HEADER_TEXT, THICKNESS_SMALL, cv2.LINE_AA)


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

    Parameters
    ----------
    exercise    : translated exercise name (e.g. "Sentar e Alcançar")
    side        : translated side label   (e.g. "Lado Direito")
    rep_current : current rep number (1-based)
    rep_total   : total reps (2)
    """
    print("[register] início")
    try:
        fields = [_("Age"), _("Height (cm)"), _("Weight (kg)"), _("Gender (M/F)")]
        print("[register] fields criados")
        values = ["", "", "", ""]
        active_field = -1

        # Card geometry
        card_w, card_h = 580, 480
        card_x = 40
        card_y = 10
        print(f"[register] card_x={card_x} card_y={card_y}")

        win_w = card_w + 80   # 660
        win_h = card_h + 60   # 540

        # Field geometry inside card
        field_x      = card_x + 30
        field_w      = card_w - 60
        field_h      = 52
        field_gap    = 22
        title_h      = 58
        fields_start = card_y + title_h + 24

        positions = [
            (field_x,
            fields_start + i * (field_h + field_gap),
            field_x + field_w,
            fields_start + i * (field_h + field_gap) + field_h)
            for i in range(len(fields))
        ]

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
            img = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            img[:] = BG

            _draw_card(img, card_x, card_y, card_w, card_h)
            img = _draw_card_title(img, _("Register").upper(),
                                card_x, card_y, card_w, title_h)

            # Card background
            _draw_card(img, card_x, card_y, card_w, card_h)

            # Card title bar
            img = _draw_card_title(img, _("Register").upper(),
                                card_x, card_y, card_w, title_h)

            # Fields
            for i, (x1, y1, x2, y2) in enumerate(positions):
                img = _put_text(img, f"{fields[i]}:", (x1, y1 - 22),
                                font_size=18, color=tuple(DARK_BLUE))
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
                border = BTN_BLUE if i == active_field else DARK_BLUE
                cv2.rectangle(img, (x1, y1), (x2, y2), border, 2)
                img = _put_text(img, values[i], (x1 + 10, y1 + 12),
                                font_size=22, color=(0, 0, 0))

            # Footer hint
            hint_y = card_y + card_h + 12
            img = _put_text(img, _("Press Enter to confirm"),
                            ((win_w - 300) // 2, hint_y),
                            font_size=18, color=tuple(DARK_BLUE))
                            
            cv2.imshow(WIN, img)

            key = cv2.waitKey(10) & 0xFF
            if key == 27:
                cv2.destroyWindow(WIN)
                raise SystemExit(0)
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
            raise SystemExit(0)
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

        # Bottom bar
        rep_label = f"{_('repetition_label')} {rep_current}/{rep_total}"
        cv2.putText(img, exercise.upper(), (20, win_h - 14),
                    FONT, 0.55, DARK_BLUE, 1, cv2.LINE_AA)
        (tw, th), _bl = cv2.getTextSize(side, FONT, 0.55, 1)
        cv2.putText(img, side, ((win_w - tw) // 2, win_h - 14),
                    FONT, 0.55, DARK_BLUE, 1, cv2.LINE_AA)
        (tw, th), _bl = cv2.getTextSize(rep_label, FONT, 0.55, 1)
        cv2.putText(img, rep_label, (win_w - tw - 20, win_h - 14),
                    FONT, 0.55, DARK_BLUE, 1, cv2.LINE_AA)

        cv2.imshow(WIN, img)

        key = cv2.waitKey(16) & 0xFF
        if key == ord(" "):
            cv2.destroyWindow(WIN)
            return
        elif key == 27:
            finish_cb()


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
    title_h  = 58
    sec_h    = 48   # secção lado
    rep_h    = 28   # subtítulo repetição
    box_h    = 140  # AUMENTADO: Caixa de valores maior (antes era 110)
    gap      = 12   # Ajustado para equilibrar o espaço vertical reconstruído
    col_w    = (card_w - 80) // 2   # largura de cada coluna
    col1_x   = card_x + 40
    col2_x   = col1_x + col_w + 20

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

        # IPBeja
        cv2.putText(img, "IPBeja", (W - 110, 28), FONT, 0.7, DARK_BLUE, 1, cv2.LINE_AA)

        # Título
        (tw, th), _bl = cv2.getTextSize(_("Exercise Completed"), FONT, 1.6, 3)
        tx = (W - tw) // 2
        ty = card_y + 50
        line_y = ty - th // 2 + 5
        cv2.line(img, (card_x + 20, line_y), (tx - 20, line_y), DARK_BLUE, 2)
        cv2.line(img, (tx + tw + 20, line_y), (card_x + card_w - 20, line_y), DARK_BLUE, 2)
        cv2.putText(img, _("Exercise Completed"), (tx, ty), FONT, 1.6, DARK_BLUE, 3, cv2.LINE_AA)

        # ── Secção Lado Direito ──
        cv2.rectangle(img, (col1_x, right_sec_y), (col1_x + col_w * 2 + 20, right_sec_y + sec_h), BTN_BLUE, -1)
        (tw, th), _bl = cv2.getTextSize(_("right_side_label"), FONT, 1.0, 2)
        cv2.putText(img, _("right_side_label"), (col1_x + (col_w * 2 + 20 - tw) // 2, right_sec_y + sec_h - 12),
                    FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # Subtítulos repetições direito
        for ci, label in enumerate([_("rep_1_label"), _("rep_2_label")]):
            cx = col1_x if ci == 0 else col2_x
            (tw, th), _bl = cv2.getTextSize(label, FONT, 0.75, 1)
            cv2.putText(img, label, (cx + (col_w - tw) // 2, right_rep_y + rep_h - 6),
                        FONT, 0.75, DARK_BLUE, 1, cv2.LINE_AA)

        # Caixas valores direito
        for ci, (sys_v, real_v) in enumerate([(system_right_1, real_right_1), (system_right_2, real_right_2)]):
            cx = col1_x if ci == 0 else col2_x
            box_y = right_box_y
            cv2.rectangle(img, (cx, box_y), (cx + col_w, box_y + box_h), (255, 255, 255), -1)
            cv2.rectangle(img, (cx, box_y), (cx + col_w, box_y + box_h), DARK_BLUE, 2)
            
            # AUMENTADO: font_size=26 e posições Y ajustadas para centralizar o texto maior
            img = _put_text(img, f"{_('system_distance_label')}: {sys_v} cm",
                            (cx + 16, box_y + 20), font_size=26, color=tuple(DARK_BLUE))
            img = _put_text(img, f"{_('real_distance_label')}: {real_v:.2f} cm",
                            (cx + 16, box_y + 80), font_size=26, color=tuple(DARK_BLUE))
                                
        # ── Secção Lado Esquerdo ──
        cv2.rectangle(img, (col1_x, left_sec_y), (col1_x + col_w * 2 + 20, left_sec_y + sec_h), BTN_BLUE, -1)
        (tw, th), _bl = cv2.getTextSize(_("left_side_label"), FONT, 1.0, 2)
        cv2.putText(img, _("left_side_label"), (col1_x + (col_w * 2 + 20 - tw) // 2, left_sec_y + sec_h - 12),
                    FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # Subtítulos repetições esquerdo
        for ci, label in enumerate([_("rep_1_label"), _("rep_2_label")]):
            cx = col1_x if ci == 0 else col2_x
            (tw, th), _bl = cv2.getTextSize(label, FONT, 0.75, 1)
            cv2.putText(img, label, (cx + (col_w - tw) // 2, left_rep_y + rep_h - 6),
                        FONT, 0.75, DARK_BLUE, 1, cv2.LINE_AA)

        # Caixas valores esquerdo
        for ci, (sys_v, real_v) in enumerate([(system_left_1, real_left_1), (system_left_2, real_left_2)]):
            cx = col1_x if ci == 0 else col2_x
            box_y = left_box_y
            cv2.rectangle(img, (cx, box_y), (cx + col_w, box_y + box_h), (255, 255, 255), -1)
            cv2.rectangle(img, (cx, box_y), (cx + col_w, box_y + box_h), DARK_BLUE, 2)
            
            # AUMENTADO: font_size=26 e posições Y ajustadas para centralizar o texto maior
            img = _put_text(img, f"{_('system_distance_label')}: {sys_v} cm",
                            (cx + 16, box_y + 20), font_size=26, color=tuple(DARK_BLUE))
            img = _put_text(img, f"{_('real_distance_label')}: {real_v:.2f} cm",
                            (cx + 16, box_y + 80), font_size=26, color=tuple(DARK_BLUE))

        # Exercise label base
        (tw, th), _bl = cv2.getTextSize(exercise.upper(), FONT, 0.8, 1)
        cv2.putText(img, exercise.upper(), ((W - tw) // 2, card_y + card_h + 30),
                    FONT, 0.8, DARK_BLUE, 1, cv2.LINE_AA)

        # Hint
        img = _put_text(img, _("Press ESC to exit"),
                        ((W - 200) // 2, card_y + card_h + 55),
                        font_size=16, color=tuple(DARK_BLUE))

        cv2.imshow(WIN, img)
        key = cv2.waitKey(16) & 0xFF
        if key == 27: 
            cv2.destroyWindow(WIN)
            finish_cb()
            break