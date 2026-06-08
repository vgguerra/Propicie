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

from locale_setup import _ as _tr
from ui.draw import blank_canvas
from ui.theme import W, H, BORDER_INSET, DARK_BLUE, BTN_BLUE, BG
from ui.forms import _put_text, _measure_text
from utils import win_title

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


def _load_and_filter_data(opts):
    """Lê as planilhas selecionadas e aplica a filtragem combinada das Checkboxes."""
    df_list = []
    
    # 1. Carregamento Seletivo por Exercício
    if opts["exe_sr"] and os.path.exists(_PATH_SR):
        try:
            df_sr = pd.read_excel(_PATH_SR)
            df_sr['Origem_Ex'] = "Sit & Reach"
            df_list.append(df_sr)
        except Exception as e: print(f"Erro SR: {e}")
            
    if opts["exe_bs"] and os.path.exists(_PATH_BS):
        try:
            df_bs = pd.read_excel(_PATH_BS)
            df_bs['Origem_Ex'] = "Back Scratch"
            df_list.append(df_bs)
        except Exception as e: print(f"Erro BS: {e}")
            
    if not df_list:
        return pd.DataFrame()
        
    df = pd.concat(df_list, ignore_index=True)
    if df.empty:
        return df

    # Normalização de cabeçalhos comuns do Excel
    for old_col, new_col in [('Weigth', 'Weight'), ('Género', 'Gender'), ('Genero', 'Gender'), ('Sexo', 'Gender'), ('lado', 'Side'), ('Lado', 'Side'), ('idade', 'Age'), ('Idade', 'Age')]:
        if old_col in df.columns and new_col not in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    # 2. Filtro de Género cruzado
    if 'Gender' in df.columns:
        if opts["gen_f"] and not opts["gen_m"]:
            df = df[df['Gender'].astype(str).str.strip().str.upper().str.startswith('F')]
        elif opts["gen_m"] and not opts["gen_f"]:
            df = df[df['Gender'].astype(str).str.strip().str.upper().str.startswith('M')]

    # 3. Filtro de Lado Corporal
    if 'Side' in df.columns:
        if opts["side_dir"] and not opts["side_esq"]:
            df = df[df['Side'].astype(str).str.lower() == 'right']
        elif opts["side_esq"] and not opts["side_dir"]:
            df = df[df['Side'].astype(str).str.lower() == 'left']

    # 4. Filtro de Faixa Etária (Idade)
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        if opts["age_g60"] and not opts["age_l60"]:
            df = df[df['Age'] >= 60]
        elif opts["age_l60"] and not opts["age_g60"]:
            df = df[df['Age'] < 60]

    return df


def _generate_processed_chart(df):
    """Gera o Line Chart comparativo das últimas repetições executadas."""
    plt.rcdefaults()
    try:
        if df.empty: return None
        
        # Isola as últimas 25 ocorrências ordenadas para estabilidade visual
        df_chart = df.tail(100).reset_index()
        
        fig = plt.figure(figsize=(9.2, 3.4), dpi=100)
        ax = fig.add_subplot(111)
        
        c_dark_blue = '#142E8B'  
        c_btn_blue  = '#4A72E4'  
        
        if 'Real distance' in df_chart.columns and 'Calculated distance' in df_chart.columns:
            ax.plot(df_chart.index + 1, df_chart['Real distance'], marker='o', linewidth=2, color=c_dark_blue, label=_tr('Distância Real'))
            ax.plot(df_chart.index + 1, df_chart['Calculated distance'], marker='s', linewidth=2, color=c_btn_blue, label=_tr('Distância Calculada'))
            
        ax.set_title(_tr("Medições Filtradas (cm)"), fontsize=11, fontweight='bold', color=c_dark_blue, pad=6)
        ax.set_xlabel(_tr("Sequência Cronológica de Repetições"), fontsize=9, color=c_dark_blue)
        ax.set_ylabel("cm", fontsize=9, color=c_dark_blue)
        
        ax.grid(True, linestyle='--', alpha=0.5, color='#CBD5E1')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', frameon=True, facecolor='#F8FAFC', edgecolor=c_dark_blue)
        
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


def show_data_visualization():
    WIN = win_title(_tr("Visualize Data"))
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)
    
    # Dicionário de Estados das Checkboxes — Inicializados todos como True
    opts = {
        "exe_sr": True, "exe_bs": True,      # Exercícios
        "gen_f": True,  "gen_m": True,       # Géneros
        "side_esq": True, "side_dir": True,  # Lados Corporais
        "age_g60": True, "age_l60": True     # Faixas Etárias
    }
    
    # Renderização Inicial Automática
    df_current = _load_and_filter_data(opts)
    current_chart = _generate_processed_chart(df_current)
    
    # Coordenadas Simétricas das 4 Colunas (X1, Y1, X2, Y2, Chave)
    click_zones = [
        # Coluna 1: Exercícios (X = 100)
        (100, 170, 125, 195, "exe_sr"),
        (100, 220, 125, 245, "exe_bs"),
        # Coluna 2: Géneros (X = 360)
        (360, 170, 385, 195, "gen_f"),
        (360, 220, 385, 245, "gen_m"),
        # Coluna 3: Lados Corporais (X = 580)
        (580, 170, 605, 195, "side_esq"),
        (580, 220, 605, 245, "side_dir"),
        # Coluna 4: Idades (X = 780)
        (780, 170, 805, 195, "age_g60"),
        (780, 220, 805, 245, "age_l60")
    ]
    
    # Botão Calcular posicionado à extrema direita da faixa reguladora
    btn_calc = (1000, 120, 1110, 155)
    hover_calc = False

    def _mouse_callback(event, x, y, flags, param):
        nonlocal current_chart, df_current, hover_calc
        
        # Gestão do Hover do Botão Calcular
        if btn_calc[0] <= x <= btn_calc[2] and btn_calc[1] <= y <= btn_calc[3]:
            hover_calc = True
            if event == cv2.EVENT_LBUTTONDOWN:
                df_current = _load_and_filter_data(opts)
                current_chart = _generate_processed_chart(df_current)
        else:
            hover_calc = False
            
        # Gestão de Cliques nas Checkboxes com Regra de Mínimo 1 Ativo por Grupo
        if event == cv2.EVENT_LBUTTONDOWN:
            for x1, y1, x2, y2, key in click_zones:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    
                    # 1. Bloqueio para Grupo Exercícios
                    if key == "exe_sr" and opts["exe_sr"] and not opts["exe_bs"]: continue
                    if key == "exe_bs" and opts["exe_bs"] and not opts["exe_sr"]: continue
                    
                    # 2. Bloqueio para Grupo Género
                    if key == "gen_f" and opts["gen_f"] and not opts["gen_m"]: continue
                    if key == "gen_m" and opts["gen_m"] and not opts["gen_f"]: continue
                    
                    # 3. Bloqueio para Grupo Lado Corporal
                    if key == "side_esq" and opts["side_esq"] and not opts["side_dir"]: continue
                    if key == "side_dir" and opts["side_dir"] and not opts["side_esq"]: continue
                    
                    # 4. Bloqueio para Grupo Faixa Etária
                    if key == "age_g60" and opts["age_g60"] and not opts["age_l60"]: continue
                    if key == "age_l60" and opts["age_l60"] and not opts["age_g60"]: continue
                    
                    # Inverte o estado da checkbox validada
                    opts[key] = not opts[key]
                    break

    cv2.setMouseCallback(WIN, _mouse_callback)

    while True:
        img = blank_canvas()
        cv2.rectangle(img, (BORDER_INSET, 80), (W - BORDER_INSET, H - BORDER_INSET), DARK_BLUE, 2)

        # Título centrado na linha superior da borda (como o menu)
        title_text = _tr("Visualize Data")
        font_size_title = 43
        tw, th = _measure_text(title_text, font_size_title, is_bold=True)
        tx = (W - tw) // 2
        ty = 80 - th // 2
        pad = 14
        cv2.rectangle(img, (tx - pad, ty - pad), (tx + tw + pad, ty + th + pad), BG, -1)
        img = _put_text(img, title_text, (tx, ty), font_size=font_size_title, color=tuple(DARK_BLUE), is_bold=True)
        
        # Faixa de Instrução Principal (Alargada para cobrir o novo layout)
        cv2.rectangle(img, (80, 120), (980, 155), tuple(DARK_BLUE), -1)
        img = _put_text(img, _tr("choose"), (95, 128), font_size=15, color=(255, 255, 255))
        
        # Desenho modular das Checkboxes gráficas
        def _draw_chk(canvas, x, y, checked, label):
            cv2.rectangle(canvas, (x, y), (x + 24, y + 24), tuple(DARK_BLUE), 2)
            if checked:
                cv2.rectangle(canvas, (x + 5, y + 5), (x + 19, y + 19), tuple(DARK_BLUE), -1)
            return _put_text(canvas, label, (x + 36, y + 3), font_size=16, color=tuple(DARK_BLUE))

        # Renderização das 4 Colunas Sétricas no Painel OpenCV
        img = _draw_chk(img, 100, 170, opts["exe_sr"], _tr("Sentar e Alcançar"))
        img = _draw_chk(img, 100, 220, opts["exe_bs"], _tr("Alcançar atrás das Costas"))
        
        img = _draw_chk(img, 360, 170, opts["gen_f"], _tr("Feminino"))
        img = _draw_chk(img, 360, 220, opts["gen_m"], _tr("Masculino"))
        
        img = _draw_chk(img, 580, 170, opts["side_esq"], _tr("Esquerdo"))
        img = _draw_chk(img, 580, 220, opts["side_dir"], _tr("Direito"))
        
        img = _draw_chk(img, 780, 170, opts["age_g60"], _tr("+60"))
        img = _draw_chk(img, 780, 220, opts["age_l60"], _tr("-60"))
        
        # Renderização Estilizada do Botão Calcular
        bc_color = tuple(BTN_BLUE) if hover_calc else (255, 255, 255)
        txt_color = (255, 255, 255) if hover_calc else tuple(DARK_BLUE)
        cv2.rectangle(img, (btn_calc[0], btn_calc[1]), (btn_calc[2], btn_calc[3]), tuple(DARK_BLUE), 1)
        cv2.rectangle(img, (btn_calc[0]+1, btn_calc[1]+1), (btn_calc[2]-1, btn_calc[3]-1), bc_color, -1)
        img = _put_text(img, _tr("calculate"), (btn_calc[0] + 18, btn_calc[1] + 8), font_size=14, color=txt_color)

        # Zona de Exibição do Gráfico Dinâmico (centralizado em x)
        gy = 310
        if current_chart is not None and not df_current.empty:
            gh, gw, _ = current_chart.shape
            gx = (W - gw) // 2
            
            # Erro Médio — centrado acima da tabela
            erro_col = next((c for c in ['Erro', 'erro', 'Error'] if c in df_current.columns), None)
            if erro_col:
                avg_err = df_current[erro_col].mean()
                err_txt = f"{_tr('avarage_error')}: {avg_err:.2f} cm"
                tw_e, _ = _measure_text(err_txt, 14)
                img = _put_text(img, err_txt, ((W - tw_e) // 2, gy - 20), font_size=14, color=tuple(DARK_BLUE))
            
            img[gy:gy+gh, gx:gx+gw] = current_chart
            cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), DARK_BLUE, 1)
        else:
            gx = BORDER_INSET
            cv2.rectangle(img, (gx, gy), (W - gx, H - 120), (255, 255, 255), -1)
            cv2.rectangle(img, (gx, gy), (W - gx, H - 120), DARK_BLUE, 1)
            tw_e, _ = _measure_text(_tr("no_data"), 16)
            img = _put_text(img, _tr("no_data"), ((W - tw_e) // 2, gy + 100), font_size=16, color=tuple(DARK_BLUE))

        # Rodapé de Saída Uniforme
        img = _put_text(img, _tr("esc_return"), (W // 2 - 180, H - 65), font_size=14, color=tuple(DARK_BLUE))

        cv2.imshow(WIN, img)
        key = cv2.waitKey(20) & 0xFF
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == 27:  
            break
            
    cv2.destroyWindow(WIN)