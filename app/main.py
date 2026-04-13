import streamlit as st
import numpy as np
import pandas as pd
from random import choice
import joblib 
from pathlib import Path
import os

st.set_page_config(layout="wide")
st.title("TIC TAC TOE vs Random Bot com Análise de IA")

# --- 1. CONFIGURAÇÃO DE DIRETÓRIOS DINÂMICOS ---
CAMINHO_BASE_MODELOS = Path("../models")

def listar_modelos_disponiveis():
    """Lê a pasta base e retorna o nome de todas as subpastas que existem lá."""
    if not CAMINHO_BASE_MODELOS.exists():
        return ["Nenhum"]
    # Pega apenas os nomes das pastas (ignora arquivos soltos)
    pastas = [pasta.name for pasta in CAMINHO_BASE_MODELOS.iterdir() if pasta.is_dir()]
    return ["Nenhum"] + pastas

# --- 2. CARREGAMENTO DO MODELO SELECIONADO ---
@st.cache_resource
def carregar_modelo(nome_pasta_modelo):
    """Carrega dinamicamente o modelo e o encoder da pasta selecionada."""
    if nome_pasta_modelo == "Nenhum":
        return None, None
        
    pasta_alvo = CAMINHO_BASE_MODELOS / nome_pasta_modelo
    try:
        # Pega todos os arquivos .pkl dentro da pasta escolhida
        arquivos_pkl = list(pasta_alvo.glob("*.pkl"))
        
        # Identifica automaticamente quem é quem pelo nome do arquivo
        caminho_encoder = next((f for f in arquivos_pkl if "encoder" in f.name.lower()), None)
        caminho_modelo = next((f for f in arquivos_pkl if "encoder" not in f.name.lower()), None)
        
        if caminho_modelo and caminho_encoder:
            model = joblib.load(caminho_modelo)
            le = joblib.load(caminho_encoder)
            return model, le
        else:
            st.sidebar.error(f"⚠️ Faltam arquivos na pasta '{nome_pasta_modelo}'. Certifique-se de ter o modelo e o encoder (com 'encoder' no nome) salvos lá.")
            return None, None
            
    except Exception as e:
        st.sidebar.error(f"⚠️ Erro ao carregar o modelo: {e}")
        return None, None

# --- 3. VARIÁVEIS DE SESSÃO E FUNÇÕES DO JOGO ---
if "board" not in st.session_state:
    st.session_state.board = np.zeros((3, 3), dtype=int)
if "turn" not in st.session_state:
    st.session_state.turn = 1

def render_cell(value):
    if value == 1:
        return "X"
    elif value == -1:
        return "O"
    return " "

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

def get_empty_cells():
    return [(i, j) for i in range(3) for j in range(3) if st.session_state.board[i][j] == 0]

def handle_click(row, col):
    if st.session_state.board[row][col] == 0 and check_winner(st.session_state.board) is None:
        # Jogada do Player (X)
        st.session_state.board[row][col] = 1
        st.session_state.turn = -1
        
        # Jogada do Bot (O)
        if check_winner(st.session_state.board) is None:
            empty = get_empty_cells()
            if empty:
                bi, bj = choice(empty)
                st.session_state.board[bi][bj] = -1
                st.session_state.turn = 1

# --- 4. SIDEBAR (Menu Dinâmico) ---
with st.sidebar:
    st.header('Configurações')

    modelos_encontrados = listar_modelos_disponiveis()
    modelo_selecionado = st.selectbox(
        "Selecione a IA Analista", 
        modelos_encontrados
    )

    modelo_ia, encoder = carregar_modelo(modelo_selecionado)
    
    st.markdown("---")
    st.write("Você joga como: **X**")
    st.write("Bot joga como: **O**")

col1, espaco, col2 = st.columns([1, 0.3, 1])

# --- 5. COLUNA 1 (Tabuleiro Visual) ---
with col1:
    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] div[data-testid="column"] {
        padding: 0 0px !important;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] {
        gap: 2px !important;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] div.stButton > button {
        width: 100px;
        height: 100px;
        background-color: #1e1e1e;
        border-radius: 15px;
        font-size: 42px;
        border: none;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        color: white;
        cursor: pointer;
        transition: background-color 0.2s;
        padding: 0;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] div.stButton > button:hover {
        background-color: #2e2e2e;
        border: none;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] div.stButton > button:focus {
        border: none;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

    for r in range(3):
        cols = st.columns(3)
        for c in range(3):
            with cols[c]:
                label = render_cell(st.session_state.board[r][c])
                disabled = st.session_state.board[r][c] != 0 or check_winner(st.session_state.board) is not None
                st.button(label, key=f"cell_{r}_{c}", disabled=disabled, on_click=handle_click, args=(r, c))



    if st.button("Reiniciar Jogo", type="primary"):
        st.session_state.board = np.zeros((3, 3), dtype=int)
        st.session_state.turn = 1

# --- 6. COLUNA 2 (Análise da IA) ---
# --- 6. COLUNA 2 (Análise da IA e Correção de Erros) ---
with col2:
    st.header("🧠 Análise em Tempo Real")
    
    if modelo_selecionado != "Nenhum":
        if modelo_ia is not None and encoder is not None:
            # Pega o tabuleiro atual e formata
            tabuleiro_flat = st.session_state.board.flatten()
            colunas_treino = [f'pos_{i}' for i in range(1, 10)]
            df_jogada = pd.DataFrame([tabuleiro_flat], columns=colunas_treino)
            
            # Faz a previsão da IA
            previsao_numerica = modelo_ia.predict(df_jogada)
            previsao_texto = encoder.inverse_transform(previsao_numerica)[0]
            
            # Calcula o Status Real do jogo
            winner = check_winner(st.session_state.board)
            sym_real = 'Tem jogo'
            if winner:
                sym_real = "X venceu" if winner == 1 else "O venceu"
            elif (st.session_state.board != 0).all():
                sym_real = 'Empate'
            
            # Verifica se o modelo acertou
            resultado_correto = (sym_real == previsao_texto)
            
            # --- ÁREA VISUAL DO DASHBOARD ---
            st.markdown(f"**Modelo ativo:** `{modelo_selecionado}`")
            
            # Card de Comparação
            with st.container(border=True):
                c1, c2 = st.columns(2)
                
                with c1:
                    st.caption("🤖 Previsão da IA")
                    if previsao_texto == 'Tem jogo':
                        st.info(f"🔮 **{previsao_texto}**")
                    elif previsao_texto == 'Empate':
                        st.warning(f"⚖️ **{previsao_texto}**")
                    else:
                        st.success(f"🏆 **{previsao_texto}**")
                        
                with c2:
                    st.caption("🎯 Status Real")
                    if sym_real == 'Tem jogo':
                        st.info(f"🎮 **{sym_real}**")
                    elif sym_real == 'Empate':
                        st.warning(f"⚖️ **{sym_real}**")
                    else:
                        st.success(f"🏆 **{sym_real}**")
            
            # Veredito Grande
            if resultado_correto:
                st.success("**O modelo acertou a avaliação!**", icon="✅")
            else:
                st.error("**O modelo errou a avaliação.**", icon="🚨")
            
            st.divider() # Linha divisória elegante
            
            # --- SISTEMA DE RETROALIMENTAÇÃO (ACTIVE LEARNING) ---
            st.subheader("🛠️ Treinamento Contínuo")
            st.write("A IA errou a previsão? Reporte a classe correta para ensiná-la no futuro.")
            
            with st.expander("📝 Salvar correção no Dataset"):
                # O usuário escolhe o que o modelo DEVERIA ter respondido
                classe_correta = st.selectbox(
                    "Qual era o status correto desta jogada?",
                    ["Tem jogo", "X venceu", "O venceu", "Empate"],
                    index=["Tem jogo", "X venceu", "O venceu", "Empate"].index(sym_real) # Traz o Status Real como padrão!
                )
                
                if st.button("Salvar Correção", type="primary", use_container_width=True):
                    # Prepara a nova linha
                    nova_linha = tabuleiro_flat.tolist()
                    nova_linha.append(classe_correta)
                    
                    # Cria o DataFrame
                    colunas_csv = colunas_treino + ['classe']
                    df_nova_linha = pd.DataFrame([nova_linha], columns=colunas_csv)
                    
                    caminho_csv = '../data/correcoes/dataset_correcoes.csv'
                    
                    # Cria a pasta caso ela não exista (evita erro de diretório não encontrado)
                    os.makedirs(os.path.dirname(caminho_csv), exist_ok=True)
                    
                    # Salva no CSV
                    if os.path.exists(caminho_csv):
                        df_nova_linha.to_csv(caminho_csv, mode='a', header=False, index=False)
                    else:
                        df_nova_linha.to_csv(caminho_csv, mode='w', header=True, index=False)
                        
                    st.success("✅ Jogada salva com sucesso!")
    
            with st.expander("🔢 Ver Matriz Numérica"):
                st.dataframe(df_jogada, hide_index=True)
                
    else:
        # Mensagem mais amigável caso não tenha modelo selecionado
        st.info("👈 Selecione um modelo na barra lateral para ativar o painel da IA.")