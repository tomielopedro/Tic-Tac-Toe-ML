import streamlit as st
import numpy as np
import pandas as pd
from random import choice
import joblib
from pathlib import Path
from datetime import datetime
import os
import json

st.set_page_config(layout="wide", page_title="TIC TAC TOE — AI Lab")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@400;600;700&display=swap');
section[data-testid="stSidebar"] { background: #0d1117; }
div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] div[data-testid="column"] { padding: 0 0px !important; }
div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] { gap: 2px !important; }
div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] div.stButton > button {
    width: 100px; height: 100px; background-color: #161b22; border-radius: 12px;
    font-size: 42px; font-family: 'JetBrains Mono', monospace; border: 1px solid #30363d;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4); color: #e6edf3; cursor: pointer;
    transition: all 0.15s ease; padding: 0;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] div.stButton > button:hover {
    background-color: #1f2937; border-color: #58a6ff; box-shadow: 0 0 12px rgba(88,166,255,0.15);
}
</style>
""", unsafe_allow_html=True)

st.title("🎮 TIC TAC TOE — AI Analyst Lab")

CAMINHO_BASE_MODELOS = Path("../models")
CAMINHO_HISTORICO = Path("../data/historico/historico_partidas.json")
CAMINHO_CORRECOES = Path("../data/correcoes/dataset_correcoes.csv")
COLUNAS_TREINO = [f'pos_{i}' for i in range(1, 10)]
COLUNAS_CSV = COLUNAS_TREINO + ['classe', 'modelo_origem', 'previsao_modelo', 'timestamp']


def listar_modelos_disponiveis():
    if not CAMINHO_BASE_MODELOS.exists():
        return ["Nenhum"]
    pastas = [p.name for p in CAMINHO_BASE_MODELOS.iterdir() if p.is_dir()]
    return ["Nenhum"] + sorted(pastas)


@st.cache_resource
def carregar_modelo(nome_pasta_modelo):
    if nome_pasta_modelo == "Nenhum":
        return None, None
    pasta_alvo = CAMINHO_BASE_MODELOS / nome_pasta_modelo
    try:
        arquivos_pkl = list(pasta_alvo.glob("*.pkl"))
        caminho_encoder = next((f for f in arquivos_pkl if "encoder" in f.name.lower()), None)
        caminho_modelo = next((f for f in arquivos_pkl if "encoder" not in f.name.lower()), None)
        if caminho_modelo and caminho_encoder:
            return joblib.load(caminho_modelo), joblib.load(caminho_encoder)
        else:
            st.sidebar.error(f"⚠️ Faltam arquivos na pasta '{nome_pasta_modelo}'.")
            return None, None
    except Exception as e:
        st.sidebar.error(f"⚠️ Erro ao carregar: {e}")
        return None, None


def carregar_historico():
    if CAMINHO_HISTORICO.exists():
        with open(CAMINHO_HISTORICO, "r") as f:
            return json.load(f)
    return []


def salvar_historico(historico):
    os.makedirs(os.path.dirname(CAMINHO_HISTORICO), exist_ok=True)
    with open(CAMINHO_HISTORICO, "w") as f:
        json.dump(historico, f, indent=2, ensure_ascii=False)


def registrar_partida(resultado, modelo_nome, acuracia, acertos, total, log_jogadas, origem="manual"):
    historico = carregar_historico()
    partida = {
        "id": len(historico) + 1,
        "timestamp": datetime.now().isoformat(),
        "modelo": modelo_nome,
        "resultado": resultado,
        "acuracia_modelo": round(acuracia, 1),
        "acertos": acertos,
        "total_jogadas": total,
        "log_jogadas": log_jogadas,
        "origem": origem,
    }
    historico.append(partida)
    salvar_historico(historico)


def salvar_correcao(tabuleiro_flat, classe_real, modelo_nome, previsao_modelo):
    os.makedirs(os.path.dirname(CAMINHO_CORRECOES), exist_ok=True)
    nova_linha = tabuleiro_flat.tolist() + [classe_real, modelo_nome, previsao_modelo, datetime.now().isoformat()]
    df_nova = pd.DataFrame([nova_linha], columns=COLUNAS_CSV)
    if os.path.exists(CAMINHO_CORRECOES):
        df_nova.to_csv(CAMINHO_CORRECOES, mode='a', header=False, index=False)
    else:
        df_nova.to_csv(CAMINHO_CORRECOES, mode='w', header=True, index=False)


def check_winner(board):
    for i in range(3):
        if abs(board[i].sum()) == 3:
            return board[i][0]
        if abs(board[:, i].sum()) == 3:
            return board[0][i]
    if abs(np.trace(board)) == 3:
        return board[0][0]
    if abs(np.trace(np.fliplr(board))) == 3:
        return board[0][2]
    return None


def obter_status_real(board):
    winner = check_winner(board)
    if winner:
        return "X venceu" if winner == 1 else "O venceu"
    elif (board != 0).all():
        return "Empate"
    return "Tem jogo"


def render_cell(value):
    if value == 1:
        return "X"
    elif value == -1:
        return "O"
    return " "


def get_empty_cells_from(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]


def simular_partida_completa(modelo, encoder, modelo_nome):
    board = np.zeros((3, 3), dtype=int)
    turno = 1
    log_jogadas = []
    acertos = 0
    total = 0
    correcoes_salvas = 0

    while True:
        empty = get_empty_cells_from(board)
        if not empty:
            break
        r, c = choice(empty)
        board[r][c] = turno

        tab_flat = board.flatten()
        df_tab = pd.DataFrame([tab_flat], columns=COLUNAS_TREINO)
        
        # Lógica especial para MLP_Extra_Feature
        if modelo_nome == "MLP_Extra_Feature":
            tabuleiro_completo = ((df_tab[COLUNAS_TREINO] != 0).sum(axis=1) == 9).astype(int)
            df_tab['tabuleiro_completo'] = tabuleiro_completo
            prev_num = modelo.predict(df_tab)
        else:
            prev_num = modelo.predict(df_tab[COLUNAS_TREINO])
            
        prev_text = encoder.inverse_transform(prev_num)[0]
        status_real = obter_status_real(board)

        total += 1
        acertou = prev_text == status_real
        if acertou:
            acertos += 1
        else:
            salvar_correcao(tab_flat, status_real, modelo_nome, prev_text)
            correcoes_salvas += 1

        log_jogadas.append({
            "tabuleiro": tab_flat.tolist(),
            "previsao": prev_text,
            "real": status_real,
            "acertou": acertou,
        })

        if check_winner(board) is not None:
            break

        turno = -turno

    resultado_final = obter_status_real(board)
    acuracia = (acertos / total * 100) if total > 0 else 0.0

    registrar_partida(
        resultado=resultado_final,
        modelo_nome=modelo_nome,
        acuracia=acuracia,
        acertos=acertos,
        total=total,
        log_jogadas=log_jogadas,
        origem="simulacao",
    )

    return {
        "resultado": resultado_final,
        "acuracia": acuracia,
        "acertos": acertos,
        "total": total,
        "log_jogadas": log_jogadas,
        "correcoes_salvas": correcoes_salvas,
        "tabuleiro_final": board.copy(),
    }


defaults = {
    "board": np.zeros((3, 3), dtype=int),
    "turn": 1,
    "ai_acertos": 0,
    "ai_total_jogadas": 0,
    "board_apos_x": None,
    "hashes_avaliados": set(),
    "partida_registrada": False,
    "log_jogadas": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def handle_click(row, col):
    if st.session_state.board[row][col] == 0 and check_winner(st.session_state.board) is None:
        st.session_state.board[row][col] = 1
        st.session_state.board_apos_x = st.session_state.board.copy()
        st.session_state.turn = -1
        if check_winner(st.session_state.board) is None:
            empty = get_empty_cells_from(st.session_state.board)
            if empty:
                bi, bj = choice(empty)
                st.session_state.board[bi][bj] = -1
                st.session_state.turn = 1


with st.sidebar:
    st.header("⚙️ Configurações")
    modelos_encontrados = listar_modelos_disponiveis()
    modelo_selecionado = st.selectbox("Selecione a IA Analista", modelos_encontrados)
    modelo_ia, encoder = carregar_modelo(modelo_selecionado)
    st.markdown("---")
    st.write("Você joga como: **X**")
    st.write("Bot joga como: **O**")
    st.markdown("---")
    historico = carregar_historico()
    if historico:
        total_p = len(historico)
        vitorias = sum(1 for p in historico if p["resultado"] == "X venceu")
        derrotas = sum(1 for p in historico if p["resultado"] == "O venceu")
        empates = sum(1 for p in historico if p["resultado"] == "Empate")
        st.caption("📊 Resumo Geral")
        st.markdown(f"**{total_p}** partidas · **{vitorias}**W / **{derrotas}**L / **{empates}**D")


tab_main, tab_sim, tab_history, tab_models, tab_data = st.tabs([
    "🎮 Jogo", "🤖 Simulação", "📜 Histórico", "📊 Comparação de Modelos", "🗃️ Dataset de Correções"
])

with tab_main:
    col1, espaco, col2 = st.columns([1, 0.3, 1.2])

    with col1:
        for r in range(3):
            cols = st.columns(3)
            for c in range(3):
                with cols[c]:
                    label = render_cell(st.session_state.board[r][c])
                    disabled = st.session_state.board[r][c] != 0 or check_winner(st.session_state.board) is not None
                    st.button(label, key=f"cell_{r}_{c}", disabled=disabled, on_click=handle_click, args=(r, c))

        if st.button("🔄 Reiniciar Jogo", type="primary"):
            for k, v in defaults.items():
                st.session_state[k] = v if not isinstance(v, (set, list)) else type(v)()
            st.session_state.board = np.zeros((3, 3), dtype=int)
            st.rerun()

    with col2:
        st.header("🧠 Análise em Tempo Real")

        if modelo_selecionado == "Nenhum" or modelo_ia is None:
            st.info("👈 Selecione um modelo na barra lateral para ativar o painel da IA.")
        else:
            tabuleiro_flat = st.session_state.board.flatten()
            df_jogada = pd.DataFrame([tabuleiro_flat], columns=COLUNAS_TREINO)
            
            # Lógica especial para MLP_Extra_Feature
            if modelo_selecionado == "MLP_Extra_Feature":
                tabuleiro_completo = ((df_jogada[COLUNAS_TREINO] != 0).sum(axis=1) == 9).astype(int)
                df_jogada['tabuleiro_completo'] = tabuleiro_completo
                previsao_numerica = modelo_ia.predict(df_jogada)
            else:
                previsao_numerica = modelo_ia.predict(df_jogada[COLUNAS_TREINO])
                
            previsao_texto = encoder.inverse_transform(previsao_numerica)[0]

            sym_real = obter_status_real(st.session_state.board)
            resultado_correto = (sym_real == previsao_texto)

            tabuleiros_para_avaliar = []
            if st.session_state.board_apos_x is not None:
                tabuleiros_para_avaliar.append(st.session_state.board_apos_x)
            tabuleiros_para_avaliar.append(st.session_state.board)

            for tab in tabuleiros_para_avaliar:
                tab_hash = tab.tobytes()
                if tab_hash not in st.session_state.hashes_avaliados and tab.any():
                    tab_flat = tab.flatten()
                    df_tab = pd.DataFrame([tab_flat], columns=COLUNAS_TREINO)
                    
                    if modelo_selecionado == "MLP_Extra_Feature":
                        tab_comp = ((df_tab[COLUNAS_TREINO] != 0).sum(axis=1) == 9).astype(int)
                        df_tab['tabuleiro_completo'] = tab_comp
                        prev_num = modelo_ia.predict(df_tab)
                    else:
                        prev_num = modelo_ia.predict(df_tab[COLUNAS_TREINO])
                        
                    prev_text = encoder.inverse_transform(prev_num)[0]
                    status_tab = obter_status_real(tab)

                    st.session_state.ai_total_jogadas += 1
                    acertou = prev_text == status_tab
                    if acertou:
                        st.session_state.ai_acertos += 1

                    st.session_state.log_jogadas.append({
                        "tabuleiro": tab_flat.tolist(),
                        "previsao": prev_text,
                        "real": status_tab,
                        "acertou": acertou,
                    })
                    st.session_state.hashes_avaliados.add(tab_hash)

            st.markdown(f"**Modelo ativo:** `{modelo_selecionado}`")

            with st.container(border=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("🤖 Previsão da IA")
                    color_map = {"Tem jogo": "info", "Empate": "warning"}
                    getattr(st, color_map.get(previsao_texto, "success"))(f"**{previsao_texto}**")
                with c2:
                    st.caption("🎯 Status Real")
                    getattr(st, color_map.get(sym_real, "success"))(f"**{sym_real}**")

            if resultado_correto:
                st.success("**O modelo acertou a avaliação!**", icon="✅")
            else:
                st.error("**O modelo errou a avaliação.**", icon="🚨")

            acertos = st.session_state.ai_acertos
            total = st.session_state.ai_total_jogadas
            acuracia = (acertos / total * 100) if total > 0 else 0.0
            winner = check_winner(st.session_state.board)
            partida_encerrada = winner is not None or (st.session_state.board != 0).all()

            if partida_encerrada:
                resultado_partida = sym_real

                if not st.session_state.partida_registrada:
                    registrar_partida(
                        resultado=resultado_partida,
                        modelo_nome=modelo_selecionado,
                        acuracia=acuracia,
                        acertos=acertos,
                        total=total,
                        log_jogadas=st.session_state.log_jogadas,
                    )
                    st.session_state.partida_registrada = True

                st.markdown("---")
                cor = '#4ade80' if acuracia >= 70 else '#facc15' if acuracia >= 40 else '#f87171'
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #0d1117, #161b22);
                    border: 2px solid {cor}; border-radius: 12px;
                    padding: 18px 24px; text-align: center; margin: 8px 0;
                ">
                    <div style="font-size: 12px; color: #8b949e; letter-spacing: 1.5px; text-transform: uppercase;">
                        Acurácia da Partida
                    </div>
                    <div style="font-size: 42px; font-weight: 900; color: {cor};">
                        {acuracia:.0f}%
                    </div>
                    <div style="font-size: 14px; color: #8b949e; margin-top: 4px;">
                        {acertos} acertos em {total} jogadas
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.metric(
                    label="Acurácia da Partida (parcial)",
                    value=f"{acuracia:.0f}%",
                    delta=f"{acertos}/{total} jogadas" if total > 0 else "Aguardando jogadas",
                    delta_color="off",
                )

            st.divider()

            st.subheader("🛠️ Treinamento Contínuo")
            with st.expander("📝 Salvar correção no Dataset"):
                classe_correta = st.selectbox(
                    "Qual era o status correto desta jogada?",
                    ["Tem jogo", "X venceu", "O venceu", "Empate"],
                    index=["Tem jogo", "X venceu", "O venceu", "Empate"].index(sym_real),
                )
                if st.button("Salvar Correção", type="primary", use_container_width=True):
                    salvar_correcao(tabuleiro_flat, classe_correta, modelo_selecionado, previsao_texto)
                    st.success("✅ Correção salva com modelo de origem!")

            with st.expander("🔢 Ver Matriz Numérica"):
                st.dataframe(df_jogada, hide_index=True)


with tab_sim:
    st.header("🤖 Simulação — Máquina vs Máquina")
    st.caption("Duas máquinas jogam aleatoriamente. O modelo analisa cada estado do tabuleiro. Erros são salvos automaticamente no dataset de correções.")

    modelos_sim = [m for m in listar_modelos_disponiveis() if m != "Nenhum"]

    if not modelos_sim:
        st.warning("Nenhum modelo encontrado na pasta de modelos.")
    else:
        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            modelo_sim_nome = st.selectbox("Modelo para simulação", modelos_sim, key="sim_modelo")
        with col_cfg2:
            n_partidas = st.number_input("Número de partidas", min_value=1, max_value=500, value=10, step=5, key="sim_n")

        if st.button("▶️ Iniciar Simulação", type="primary", use_container_width=True):
            modelo_sim, encoder_sim = carregar_modelo(modelo_sim_nome)
            if modelo_sim is None or encoder_sim is None:
                st.error("Não foi possível carregar o modelo selecionado.")
            else:
                resultados = []
                total_correcoes = 0
                progress = st.progress(0, text="Simulando partidas...")

                for i in range(n_partidas):
                    res = simular_partida_completa(modelo_sim, encoder_sim, modelo_sim_nome)
                    resultados.append(res)
                    total_correcoes += res["correcoes_salvas"]
                    progress.progress((i + 1) / n_partidas, text=f"Partida {i + 1}/{n_partidas}")

                progress.empty()
                st.success(f"✅ Simulação concluída! {n_partidas} partidas jogadas.")

                st.session_state.sim_resultados = resultados
                st.session_state.sim_modelo_nome = modelo_sim_nome
                st.session_state.sim_total_correcoes = total_correcoes

        if "sim_resultados" in st.session_state and st.session_state.sim_resultados:
            resultados = st.session_state.sim_resultados
            modelo_sim_nome = st.session_state.sim_modelo_nome
            total_correcoes = st.session_state.sim_total_correcoes

            st.divider()
            st.subheader(f"Resultados — `{modelo_sim_nome}`")

            acuracias = [r["acuracia"] for r in resultados]
            resultados_finais = [r["resultado"] for r in resultados]
            total_acertos_sim = sum(r["acertos"] for r in resultados)
            total_jogadas_sim = sum(r["total"] for r in resultados)
            acuracia_global = (total_acertos_sim / total_jogadas_sim * 100) if total_jogadas_sim > 0 else 0

            k1, k2, k3, k4, k5 = st.columns(5)
            with k1:
                st.metric("Partidas", len(resultados))
            with k2:
                st.metric("Acurácia Global", f"{acuracia_global:.1f}%")
            with k3:
                st.metric("Acurácia Média", f"{np.mean(acuracias):.1f}%")
            with k4:
                st.metric("Correções Salvas", total_correcoes)
            with k5:
                st.metric("Desvio Padrão", f"{np.std(acuracias):.1f}%")

            r1, r2, r3 = st.columns(3)
            with r1:
                x_wins = resultados_finais.count("X venceu")
                st.metric("X Venceu", f"{x_wins} ({x_wins/len(resultados)*100:.0f}%)")
            with r2:
                o_wins = resultados_finais.count("O venceu")
                st.metric("O Venceu", f"{o_wins} ({o_wins/len(resultados)*100:.0f}%)")
            with r3:
                draws = resultados_finais.count("Empate")
                st.metric("Empates", f"{draws} ({draws/len(resultados)*100:.0f}%)")

            st.subheader("Distribuição de Acurácia por Partida")
            df_acc = pd.DataFrame({
                "Partida": list(range(1, len(acuracias) + 1)),
                "Acurácia (%)": acuracias,
            })
            st.line_chart(df_acc, x="Partida", y="Acurácia (%)", height=300)

            st.subheader("Análise de Erros Agregada")
            todos_erros_sim = []
            for r in resultados:
                for j in r["log_jogadas"]:
                    if not j["acertou"]:
                        todos_erros_sim.append({"previsao": j["previsao"], "real": j["real"]})

            if todos_erros_sim:
                df_err = pd.DataFrame(todos_erros_sim)
                confusion = df_err.groupby(["previsao", "real"]).size().reset_index(name="contagem")
                confusion.columns = ["Modelo Previu", "Status Real", "Ocorrências"]
                st.dataframe(confusion, hide_index=True, use_container_width=True)

                err1, err2 = st.columns(2)
                with err1:
                    st.markdown("**Erros por status real:**")
                    epr = df_err["real"].value_counts().reset_index()
                    epr.columns = ["Status Real", "Erros"]
                    st.bar_chart(epr, x="Status Real", y="Erros", height=250)
                with err2:
                    st.markdown("**Erros por previsão incorreta:**")
                    epp = df_err["previsao"].value_counts().reset_index()
                    epp.columns = ["Previsão", "Erros"]
                    st.bar_chart(epp, x="Previsão", y="Erros", height=250)
            else:
                st.success("O modelo não cometeu nenhum erro em toda a simulação!")

            st.subheader("Detalhes por Partida")
            partida_idx = st.selectbox(
                "Selecione a partida simulada",
                list(range(len(resultados))),
                format_func=lambda x: f"Partida #{x + 1} — {resultados[x]['resultado']} — {resultados[x]['acuracia']:.0f}%",
            )
            partida_det = resultados[partida_idx]

            det1, det2 = st.columns([1, 2])
            with det1:
                st.markdown("**Tabuleiro Final:**")
                board_final = partida_det["tabuleiro_final"]
                board_str = ""
                for row in board_final:
                    row_str = " | ".join(["X" if v == 1 else "O" if v == -1 else "·" for v in row])
                    board_str += f"`{row_str}`\n\n"
                st.markdown(board_str)

            with det2:
                rows_det = []
                for i, j in enumerate(partida_det["log_jogadas"], 1):
                    rows_det.append({
                        "Jogada": i,
                        "Previsão": j["previsao"],
                        "Real": j["real"],
                        "Status": "✅" if j["acertou"] else "❌",
                    })
                st.dataframe(pd.DataFrame(rows_det), hide_index=True, use_container_width=True)


with tab_history:
    historico = carregar_historico()
    if not historico:
        st.info("Nenhuma partida registrada ainda. Jogue uma partida com um modelo ativo!")
    else:
        st.header("📜 Histórico de Partidas")

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            modelos_no_hist = sorted(set(p["modelo"] for p in historico))
            filtro_modelo = st.multiselect("Filtrar por modelo", modelos_no_hist, default=modelos_no_hist)
        with col_f2:
            resultados_possiveis = ["X venceu", "O venceu", "Empate"]
            filtro_resultado = st.multiselect("Filtrar por resultado", resultados_possiveis, default=resultados_possiveis)
        with col_f3:
            origens_possiveis = sorted(set(p.get("origem", "manual") for p in historico))
            filtro_origem = st.multiselect("Filtrar por origem", origens_possiveis, default=origens_possiveis)

        hist_filtrado = [
            p for p in historico
            if p["modelo"] in filtro_modelo
            and p["resultado"] in filtro_resultado
            and p.get("origem", "manual") in filtro_origem
        ]

        if hist_filtrado:
            df_hist = pd.DataFrame(hist_filtrado)
            df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
            df_hist = df_hist.sort_values("timestamp", ascending=False)

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Total Partidas", len(df_hist))
            with k2:
                win_rate = (df_hist["resultado"] == "X venceu").mean() * 100
                st.metric("Win Rate (X)", f"{win_rate:.0f}%")
            with k3:
                st.metric("Acurácia Média dos Modelos", f"{df_hist['acuracia_modelo'].mean():.1f}%")
            with k4:
                melhor = df_hist.loc[df_hist["acuracia_modelo"].idxmax()]
                st.metric("Melhor Acurácia", f"{melhor['acuracia_modelo']:.0f}%", delta=melhor["modelo"])

            st.subheader("Acurácia por Partida ao Longo do Tempo")
            chart_data = df_hist[["timestamp", "acuracia_modelo", "modelo"]].copy()
            chart_data = chart_data.rename(columns={"acuracia_modelo": "Acurácia (%)", "timestamp": "Data"})
            st.scatter_chart(chart_data, x="Data", y="Acurácia (%)", color="modelo", height=300)

            st.subheader("Detalhes das Partidas")
            if "origem" not in df_hist.columns:
                df_hist["origem"] = "manual"
            display_cols = ["id", "timestamp", "modelo", "resultado", "acuracia_modelo", "acertos", "total_jogadas", "origem"]
            st.dataframe(
                df_hist[display_cols].rename(columns={
                    "id": "ID", "timestamp": "Data/Hora", "modelo": "Modelo",
                    "resultado": "Resultado", "acuracia_modelo": "Acurácia (%)",
                    "acertos": "Acertos", "total_jogadas": "Total Jogadas", "origem": "Origem",
                }),
                hide_index=True, use_container_width=True,
            )

            st.subheader("🔍 Detalhe Jogada-a-Jogada")
            ids_disp = df_hist["id"].tolist()
            partida_sel = st.selectbox("Selecione a partida", ids_disp, format_func=lambda x: f"Partida #{x}")
            partida_info = next(p for p in historico if p["id"] == partida_sel)

            if partida_info.get("log_jogadas"):
                rows = []
                for i, j in enumerate(partida_info["log_jogadas"], 1):
                    rows.append({
                        "Jogada": i,
                        "Previsão IA": j["previsao"],
                        "Status Real": j["real"],
                        "Resultado": "✅ Acertou" if j["acertou"] else "❌ Errou",
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

                erros = [j for j in partida_info["log_jogadas"] if not j["acertou"]]
                if erros:
                    st.warning(f"O modelo errou em **{len(erros)}** de **{len(partida_info['log_jogadas'])}** avaliações.")
                    st.caption("Padrão dos erros:")
                    err_df = pd.DataFrame(erros)
                    if len(err_df) > 0:
                        confusion = err_df.groupby(["previsao", "real"]).size().reset_index(name="contagem")
                        confusion.columns = ["Previu", "Era na verdade", "Quantidade"]
                        st.dataframe(confusion, hide_index=True)
                else:
                    st.success("O modelo acertou **todas** as avaliações nesta partida!")
            else:
                st.caption("Log detalhado não disponível para esta partida.")
        else:
            st.warning("Nenhuma partida encontrada com os filtros selecionados.")


with tab_models:
    historico = carregar_historico()
    if not historico:
        st.info("Nenhum dado de partidas ainda. Jogue partidas com diferentes modelos para ver a comparação.")
    else:
        st.header("📊 Comparação de Modelos")

        df_all = pd.DataFrame(historico)
        modelos_unicos = df_all["modelo"].unique().tolist()

        if len(modelos_unicos) < 1:
            st.info("Jogue ao menos uma partida com um modelo ativo.")
        else:
            st.subheader("Visão Geral por Modelo")
            resumo_rows = []
            for m in sorted(modelos_unicos):
                subset = df_all[df_all["modelo"] == m]
                n_partidas = len(subset)
                acuracia_media = subset["acuracia_modelo"].mean()
                acuracia_std = subset["acuracia_modelo"].std() if n_partidas > 1 else 0
                acuracia_min = subset["acuracia_modelo"].min()
                acuracia_max = subset["acuracia_modelo"].max()
                total_acertos = subset["acertos"].sum()
                total_jogadas = subset["total_jogadas"].sum()
                acuracia_global = (total_acertos / total_jogadas * 100) if total_jogadas > 0 else 0

                resumo_rows.append({
                    "Modelo": m, "Partidas": n_partidas,
                    "Acurácia Média": f"{acuracia_media:.1f}%",
                    "Acurácia Global": f"{acuracia_global:.1f}%",
                    "Desvio Padrão": f"{acuracia_std:.1f}%",
                    "Min": f"{acuracia_min:.0f}%", "Max": f"{acuracia_max:.0f}%",
                    "Total Acertos": int(total_acertos), "Total Jogadas": int(total_jogadas),
                })

            st.dataframe(pd.DataFrame(resumo_rows), hide_index=True, use_container_width=True)

            st.subheader("Acurácia Global por Modelo")
            chart_bar = []
            for m in sorted(modelos_unicos):
                subset = df_all[df_all["modelo"] == m]
                total_a = subset["acertos"].sum()
                total_j = subset["total_jogadas"].sum()
                acc = (total_a / total_j * 100) if total_j > 0 else 0
                chart_bar.append({"Modelo": m, "Acurácia Global (%)": round(acc, 1)})
            st.bar_chart(pd.DataFrame(chart_bar), x="Modelo", y="Acurácia Global (%)", horizontal=False, height=350)

            st.subheader("🔬 Análise de Erros — Onde Cada Modelo Erra")
            modelo_analise = st.selectbox("Selecione o modelo para analisar erros", sorted(modelos_unicos))
            partidas_modelo = [p for p in historico if p["modelo"] == modelo_analise]

            todos_erros = []
            todos_acertos = []
            for p in partidas_modelo:
                for j in p.get("log_jogadas", []):
                    entry = {"previsao": j["previsao"], "real": j["real"], "acertou": j["acertou"]}
                    if j["acertou"]:
                        todos_acertos.append(entry)
                    else:
                        todos_erros.append(entry)

            total_avaliacoes = len(todos_erros) + len(todos_acertos)
            if total_avaliacoes == 0:
                st.caption("Sem dados detalhados de jogadas para este modelo.")
            else:
                e1, e2, e3 = st.columns(3)
                with e1:
                    st.metric("Total Avaliações", total_avaliacoes)
                with e2:
                    st.metric("Erros", len(todos_erros))
                with e3:
                    taxa_erro = len(todos_erros) / total_avaliacoes * 100
                    st.metric("Taxa de Erro", f"{taxa_erro:.1f}%")

                if todos_erros:
                    df_erros = pd.DataFrame(todos_erros)

                    st.markdown("**Matriz de Confusão dos Erros:**")
                    confusion = df_erros.groupby(["previsao", "real"]).size().reset_index(name="contagem")
                    confusion.columns = ["Modelo Previu", "Status Real", "Ocorrências"]
                    st.dataframe(confusion, hide_index=True, use_container_width=True)

                    st.markdown("**Distribuição de erros por status real:**")
                    erro_por_real = df_erros["real"].value_counts().reset_index()
                    erro_por_real.columns = ["Status Real", "Quantidade de Erros"]
                    st.bar_chart(erro_por_real, x="Status Real", y="Quantidade de Erros", height=250)

                    st.markdown("**Distribuição de erros por previsão incorreta:**")
                    erro_por_prev = df_erros["previsao"].value_counts().reset_index()
                    erro_por_prev.columns = ["Previsão Incorreta", "Quantidade"]
                    st.bar_chart(erro_por_prev, x="Previsão Incorreta", y="Quantidade", height=250)
                else:
                    st.success(f"🎯 O modelo **{modelo_analise}** não cometeu nenhum erro registrado!")

            if len(modelos_unicos) >= 2:
                st.divider()
                st.subheader("⚔️ Comparação Direta")
                comp1, comp2 = st.columns(2)
                with comp1:
                    m1 = st.selectbox("Modelo A", sorted(modelos_unicos), key="comp_m1")
                with comp2:
                    m2_options = [m for m in sorted(modelos_unicos) if m != m1]
                    m2 = st.selectbox("Modelo B", m2_options, key="comp_m2") if m2_options else None

                if m2:
                    def calc_stats(nome):
                        s = df_all[df_all["modelo"] == nome]
                        ta = s["acertos"].sum()
                        tj = s["total_jogadas"].sum()
                        return (ta/tj*100) if tj > 0 else 0
                    
                    acc1 = calc_stats(m1)
                    acc2 = calc_stats(m2)
                    
                    st.write(f"Comparando **{m1}** vs **{m2}**")
                    st.progress(acc1/100, text=f"{m1}: {acc1:.1f}%")
                    st.progress(acc2/100, text=f"{m2}: {acc2:.1f}%")
