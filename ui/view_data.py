# =============================================================================
# ui/view_data.py — Checklist Data Visualization Screen (Age Filter Included)
# =============================================================================

import cv2
import numpy as np
import pandas as pd
import os

# Configuração absoluta do backend em memória do Matplotlib
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

from locale_setup import translate
from ui.draw import blank_canvas, put_text, measure_text
from ui.theme import W, H, BORDER_INSET, DARK_BLUE, BTN_BLUE, BG
from utils import WindowManager

# Normalização de cabeçalhos comuns do Excel
_COL_NORMALIZE = [
    ('Weigth', 'Weight'), ('Género', 'Gender'), ('Genero', 'Gender'),
    ('Sexo', 'Gender'), ('lado', 'Side'), ('Lado', 'Side'),
    ('idade', 'Age'), ('Idade', 'Age'),
]

# Mapeamento absoluto dos caminhos das tabelas utentes
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH_SR = os.path.join(_BASE_DIR, "arquivos", "tabelas_utentes", "sit_and_reach_2_utentes.xlsx")
_PATH_BS = os.path.join(_BASE_DIR, "arquivos", "tabelas_utentes", "back_scratch_utentes.xlsx")

# Tolerância adaptativa para extensões alternativas .slsx
if not os.path.exists(_PATH_SR):
    _PATH_SR_alt = os.path.join(_BASE_DIR, "arquivos", "tabelas_utentes", "sit_and_reach_2_utentes.slsx")
    if os.path.exists(_PATH_SR_alt): _PATH_SR = _PATH_SR_alt

if not os.path.exists(_PATH_BS):
    _PATH_BS_alt = os.path.join(_BASE_DIR, "arquivos", "tabelas_utentes", "back_scratch_utentes.slsx")
    if os.path.exists(_PATH_BS_alt): _PATH_BS = _PATH_BS_alt


def _generate_processed_chart(df):
    """Gera o Line Chart comparativo das últimas repetições executadas."""
    plt.rcdefaults()
    try:
        if df.empty: return None
        
        df_chart = df.reset_index(drop=True)
        
        fig = plt.figure(figsize=(9.2, 3.4), dpi=100)
        ax = fig.add_subplot(111)
        
        real_distance_color = "#FF0000"  
        calculated_distance_color  = '#142E8B'  
        
        if 'Real distance' in df_chart.columns and 'Calculated distance' in df_chart.columns:
            ax.plot(df_chart.index + 1, df_chart['Real distance'], marker='o', linewidth=2, color=real_distance_color, label=translate('real_distance_label'))
            ax.plot(df_chart.index + 1, df_chart['Calculated distance'], marker='s', linewidth=2, color=calculated_distance_color, label=translate('system_distance_label'))
            
        ax.set_title(translate("chart_title"), fontsize=11, fontweight='bold', color=calculated_distance_color, pad=6)
        ax.set_xlabel(translate("chart_xlabel"), fontsize=9, color=calculated_distance_color)
        ax.set_ylabel("cm", fontsize=9, color=calculated_distance_color)
        
        ax.grid(True, linestyle='--', alpha=0.5, color='#CBD5E1')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', frameon=True, facecolor='#F8FAFC', edgecolor=real_distance_color)
        
        plt.tight_layout()
        
        fig.canvas.draw()
        img_rgba = np.array(fig.canvas.buffer_rgba())
        img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
        
        plt.clf()
        plt.close(fig)
        return img_bgr
    except Exception as e:
        print(f"Erro ao gerar gráfico: {e}")
        return None


def _draw_chk(canvas, x, y, checked, label):
    cv2.rectangle(canvas, (x, y), (x + 24, y + 24), tuple(DARK_BLUE), 2)
    if checked:
        cv2.rectangle(canvas, (x + 5, y + 5), (x + 19, y + 19), tuple(DARK_BLUE), -1)
    return put_text(canvas, label, (x + 36, y + 3), font_size=16, color=tuple(DARK_BLUE))


def show_data_visualization():
    wm = WindowManager("Visualize Data", size=(W, H), delay=20, on_mouse=_mouse_callback)

    # Dicionário de Estados das Checkboxes
    opts = {
        "exe_sr": True, "exe_bs": True,
        "gen_f": True,  "gen_m": True,
        "side_esq": True, "side_dir": True
    }

    # ── Per-exercise data loading ──
    sr_full = pd.DataFrame()
    if os.path.exists(_PATH_SR):
        try:
            sr_full = pd.read_excel(_PATH_SR)
        except Exception as e: print(f"Erro SR: {e}")
    bs_full = pd.DataFrame()
    if os.path.exists(_PATH_BS):
        try:
            bs_full = pd.read_excel(_PATH_BS)
        except Exception as e: print(f"Erro BS: {e}")

    sr_total = len(sr_full)
    bs_total = len(bs_full)
    sr_min = 0
    sr_max = max(0, sr_total - 1)
    bs_min = 0
    bs_max = max(0, bs_total - 1)
    dragging = None  # ("sr","min"), ("sr","max"), ("bs","min"), ("bs","max") or None

    # ── Filtering helpers ──
    def _apply_filters(df):
        if df.empty:
            return df
        for old_col, new_col in _COL_NORMALIZE:
            if old_col in df.columns and new_col not in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        if 'Gender' in df.columns:
            if opts["gen_f"] and not opts["gen_m"]:
                df = df[df['Gender'].astype(str).str.strip().str.upper().str.startswith('F')]
            elif opts["gen_m"] and not opts["gen_f"]:
                df = df[df['Gender'].astype(str).str.strip().str.upper().str.startswith('M')]
        if 'Side' in df.columns:
            if opts["side_dir"] and not opts["side_esq"]:
                df = df[df['Side'].astype(str).str.lower() == 'right']
            elif opts["side_esq"] and not opts["side_dir"]:
                df = df[df['Side'].astype(str).str.lower() == 'left']
        return df

    def _build_filtered_data():
        df_list = []
        if opts["exe_sr"] and not sr_full.empty:
            d = sr_full.iloc[sr_min:sr_max+1].copy()
            d['Origem_Ex'] = "Sit & Reach"
            df_list.append(d)
        if opts["exe_bs"] and not bs_full.empty:
            d = bs_full.iloc[bs_min:bs_max+1].copy()
            d['Origem_Ex'] = "Back Scratch"
            df_list.append(d)
        if not df_list:
            return pd.DataFrame()
        df = pd.concat(df_list, ignore_index=True)
        return _apply_filters(df)

    df_current = _build_filtered_data()
    current_chart = _generate_processed_chart(df_current)

    # ── Slider constants ──
    SLIDER_LEFT  = 370
    SLIDER_RIGHT = 600
    SLIDER_H     = 8
    HANDLE_R     = 8
    SR_SLIDER_Y  = 172
    BS_SLIDER_Y  = 222

    def _draw_slider(canvas, idx_to_px, cur_min, cur_max, total, slider_y):
        if total <= 1:
            return canvas
        min_px = idx_to_px(cur_min)
        max_px = idx_to_px(cur_max)
        cv2.rectangle(canvas, (SLIDER_LEFT, slider_y - SLIDER_H//2), (min_px, slider_y + SLIDER_H//2), tuple(DARK_BLUE), -1)
        cv2.rectangle(canvas, (min_px, slider_y - SLIDER_H//2), (max_px, slider_y + SLIDER_H//2), tuple(BTN_BLUE), -1)
        cv2.rectangle(canvas, (max_px, slider_y - SLIDER_H//2), (SLIDER_RIGHT, slider_y + SLIDER_H//2), tuple(DARK_BLUE), -1)
        cv2.rectangle(canvas, (SLIDER_LEFT, slider_y - SLIDER_H//2), (SLIDER_RIGHT, slider_y + SLIDER_H//2), (100, 100, 100), 1)
        cv2.circle(canvas, (min_px, slider_y), HANDLE_R, tuple(DARK_BLUE), -1)
        cv2.circle(canvas, (max_px, slider_y), HANDLE_R, tuple(DARK_BLUE), -1)
        lbl_min = str(cur_min + 1)
        lbl_max = str(cur_max + 1)
        tw_min, _ = measure_text(lbl_min, 14)
        canvas = put_text(canvas, lbl_min, (min_px - tw_min//2, slider_y - 22), font_size=14, color=tuple(DARK_BLUE))
        tw_max, _ = measure_text(lbl_max, 14)
        canvas = put_text(canvas, lbl_max, (max_px - tw_max//2, slider_y - 22), font_size=14, color=tuple(DARK_BLUE))
        canvas = put_text(canvas, "1", (SLIDER_LEFT - 10, slider_y - 6), font_size=12, color=tuple(DARK_BLUE))
        tw_end, _ = measure_text(str(total), 12)
        canvas = put_text(canvas, str(total), (SLIDER_RIGHT - tw_end + 10, slider_y - 6), font_size=12, color=tuple(DARK_BLUE))
        return canvas

    def _sr_idx_to_px(idx):
        if sr_total <= 1:
            return (SLIDER_LEFT + SLIDER_RIGHT) // 2
        return int(SLIDER_LEFT + (idx / (sr_total - 1)) * (SLIDER_RIGHT - SLIDER_LEFT))

    def _sr_px_to_idx(px):
        if sr_total <= 1:
            return 0
        ratio = (px - SLIDER_LEFT) / (SLIDER_RIGHT - SLIDER_LEFT)
        idx = int(round(ratio * (sr_total - 1)))
        return max(0, min(sr_total - 1, idx))

    def _bs_idx_to_px(idx):
        if bs_total <= 1:
            return (SLIDER_LEFT + SLIDER_RIGHT) // 2
        return int(SLIDER_LEFT + (idx / (bs_total - 1)) * (SLIDER_RIGHT - SLIDER_LEFT))

    def _bs_px_to_idx(px):
        if bs_total <= 1:
            return 0
        ratio = (px - SLIDER_LEFT) / (SLIDER_RIGHT - SLIDER_LEFT)
        idx = int(round(ratio * (bs_total - 1)))
        return max(0, min(bs_total - 1, idx))

    def _on_slider(mx, my, slider_y):
        return SLIDER_LEFT <= mx <= SLIDER_RIGHT and abs(my - slider_y) <= HANDLE_R + 15

    def _near_handle(px, mx, my, slider_y):
        return abs(mx - px) <= HANDLE_R + 5 and abs(my - slider_y) <= HANDLE_R + 15

    # ── Click zones (colunas deslocadas para a direita) ──
    click_zones = [
        (100, 160, 125, 185, "exe_sr"),
        (100, 210, 125, 235, "exe_bs"),
        (680, 160, 705, 185, "gen_f"),
        (680, 210, 705, 235, "gen_m"),
        (840, 160, 865, 185, "side_esq"),
        (840, 210, 865, 235, "side_dir")
    ]

    btn_calc = (1075, 178, 1195, 218)
    hover_calc = False

    # ── Mouse callback ──
    def _mouse_callback(event, x, y, flags, param):
        nonlocal sr_min, sr_max, bs_min, bs_max, dragging, hover_calc, sr_full, bs_full
        nonlocal sr_total, bs_total, df_current, current_chart

        if event == cv2.EVENT_LBUTTONDOWN:
            sr_min_px = _sr_idx_to_px(sr_min)
            sr_max_px = _sr_idx_to_px(sr_max)
            bs_min_px = _bs_idx_to_px(bs_min)
            bs_max_px = _bs_idx_to_px(bs_max)

            if _near_handle(sr_min_px, x, y, SR_SLIDER_Y):
                dragging = ("sr", "min"); return
            if _near_handle(sr_max_px, x, y, SR_SLIDER_Y):
                dragging = ("sr", "max"); return
            if _near_handle(bs_min_px, x, y, BS_SLIDER_Y):
                dragging = ("bs", "min"); return
            if _near_handle(bs_max_px, x, y, BS_SLIDER_Y):
                dragging = ("bs", "max"); return
            if _on_slider(x, y, SR_SLIDER_Y):
                click_idx = _sr_px_to_idx(x)
                if abs(click_idx - sr_min) <= abs(click_idx - sr_max):
                    sr_min = min(click_idx, sr_max)
                else:
                    sr_max = max(click_idx, sr_min)
                return
            if _on_slider(x, y, BS_SLIDER_Y):
                click_idx = _bs_px_to_idx(x)
                if abs(click_idx - bs_min) <= abs(click_idx - bs_max):
                    bs_min = min(click_idx, bs_max)
                else:
                    bs_max = max(click_idx, bs_min)
                return

        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging is not None:
                ex, handle = dragging
                if ex == "sr":
                    new_idx = _sr_px_to_idx(x)
                    if handle == "min":
                        sr_min = min(new_idx, sr_max)
                    else:
                        sr_max = max(new_idx, sr_min)
                else:
                    new_idx = _bs_px_to_idx(x)
                    if handle == "min":
                        bs_min = min(new_idx, bs_max)
                    else:
                        bs_max = max(new_idx, bs_min)
                return
            hover_calc = (btn_calc[0] <= x <= btn_calc[2] and
                          btn_calc[1] <= y <= btn_calc[3])
            return

        elif event == cv2.EVENT_LBUTTONUP:
            if dragging is not None:
                dragging = None
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Botão Calcular — rebuild chart from current slider/filter state
        if btn_calc[0] <= x <= btn_calc[2] and btn_calc[1] <= y <= btn_calc[3]:
            df_current = _build_filtered_data()
            current_chart = _generate_processed_chart(df_current)
            return

        # Checkboxes
        for x1, y1, x2, y2, key in click_zones:
            if x1 <= x <= x2 and y1 <= y <= y2:
                if key == "exe_sr" and opts["exe_sr"] and not opts["exe_bs"]: continue
                if key == "exe_bs" and opts["exe_bs"] and not opts["exe_sr"]: continue
                if key == "gen_f" and opts["gen_f"] and not opts["gen_m"]: continue
                if key == "gen_m" and opts["gen_m"] and not opts["gen_f"]: continue
                if key == "side_esq" and opts["side_esq"] and not opts["side_dir"]: continue
                if key == "side_dir" and opts["side_dir"] and not opts["side_esq"]: continue
                opts[key] = not opts[key]
                break

    while not wm.should_close:
        img = blank_canvas()
        cv2.rectangle(img, (BORDER_INSET, 80), (W - BORDER_INSET, H - BORDER_INSET), DARK_BLUE, 2)

        # Título
        title_text = translate("Visualize Data")
        font_size_title = 43
        tw, th = measure_text(title_text, font_size_title, is_bold=True)
        tx = (W - tw) // 2
        ty = 80 - th // 2 - 10
        pad = 14
        cv2.rectangle(img, (tx - pad, ty - pad), (tx + tw + pad, ty + th + pad), BG, -1)
        img = put_text(img, title_text, (tx, ty), font_size=font_size_title, color=tuple(DARK_BLUE), is_bold=True)

        # Faixa de Instrução Principal
        cv2.rectangle(img, (80, 110), (1070, 145), tuple(DARK_BLUE), -1)
        img = put_text(img, translate("choose"), (95, 118), font_size=15, color=(255, 255, 255))

        # Checkboxes
        img = _draw_chk(img, 100, 160, opts["exe_sr"], translate("Sit and Reach"))
        img = _draw_chk(img, 100, 210, opts["exe_bs"], translate("Back Scratch exercise name"))
        img = _draw_chk(img, 680, 160, opts["gen_f"], translate("Feminine"))
        img = _draw_chk(img, 680, 210, opts["gen_m"], translate("Male"))
        img = _draw_chk(img, 840, 160, opts["side_esq"], translate("Left"))
        img = _draw_chk(img, 840, 210, opts["side_dir"], translate("Right"))

        # Botão Calcular
        bc_color = tuple(BTN_BLUE) if hover_calc else (255, 255, 255)
        txt_color = (255, 255, 255) if hover_calc else tuple(DARK_BLUE)
        cv2.rectangle(img, (btn_calc[0], btn_calc[1]), (btn_calc[2], btn_calc[3]), tuple(DARK_BLUE), 1)
        cv2.rectangle(img, (btn_calc[0]+1, btn_calc[1]+1), (btn_calc[2]-1, btn_calc[3]-1), bc_color, -1)
        calc_text = translate("calculate")
        tw_calc, th_calc = measure_text(calc_text, 28)
        cx = btn_calc[0] + (btn_calc[2] - btn_calc[0] - tw_calc) // 2
        cy = btn_calc[1] + (btn_calc[3] - btn_calc[1] - th_calc) // 2
        img = put_text(img, calc_text, (cx, cy), font_size=28, color=txt_color)

        img = _draw_slider(img, _sr_idx_to_px, sr_min, sr_max, sr_total, SR_SLIDER_Y)
        img = _draw_slider(img, _bs_idx_to_px, bs_min, bs_max, bs_total, BS_SLIDER_Y)

        # ── Chart area ──
        gy = 300
        if current_chart is not None:
            gh, gw, _ = current_chart.shape
            gx = (W - gw) // 2
            if not df_current.empty:
                erro_col = next((c for c in ['Erro', 'erro', 'Error'] if c in df_current.columns), None)
                if erro_col:
                    avg_err = df_current[erro_col].mean()
                    err_txt = f"{translate('avarage_error')} {avg_err:.2f} cm"
                    tw_e, _ = measure_text(err_txt, 14)
                    img = put_text(img, err_txt, ((W - tw_e) // 2, gy - 20), font_size=14, color=tuple(DARK_BLUE))
            img[gy:gy+gh, gx:gx+gw] = current_chart
            cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), DARK_BLUE, 1)
            _content_bottom = gy + gh
        else:
            gx = BORDER_INSET
            cv2.rectangle(img, (gx, gy), (W - gx, H - 120), (255, 255, 255), -1)
            cv2.rectangle(img, (gx, gy), (W - gx, H - 120), DARK_BLUE, 1)
            tw_e, _ = measure_text(translate("no_data"), 16)
            img = put_text(img, translate("no_data"), ((W - tw_e) // 2, gy + 100), font_size=16, color=tuple(DARK_BLUE))
            _content_bottom = H - 120

        # Rodapé — centrado entre o fim do conteúdo e o fim do retângulo exterior
        _outer_bottom = H - BORDER_INSET
        _gap = _outer_bottom - _content_bottom
        _esc_text = translate("esc_return")
        _tw_esc, _th_esc = measure_text(_esc_text, 26)
        _esc_y = _content_bottom + _gap // 2 - _th_esc // 2
        img = put_text(img, _esc_text, ((W - _tw_esc) // 2, _esc_y), font_size=26, color=tuple(DARK_BLUE))

        wm.show(img)
        key = wm.poll()
        if key == "close":
            break

    wm.close()