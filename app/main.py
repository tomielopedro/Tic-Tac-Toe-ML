import streamlit as st
import numpy as np
from random import choice

st.set_page_config(layout="wide")
st.title("TIC TAC TOE")

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
        st.session_state.board[row][col] = 1
        st.session_state.turn = -1
        if check_winner(st.session_state.board) is None:
            empty = get_empty_cells()
            if empty:
                bi, bj = choice(empty)
                st.session_state.board[bi][bj] = -1
                st.session_state.turn = 1

with st.sidebar:
    st.write('Settings')

col1, espaco, col2 = st.columns([1, 0.7, 1])

with col1:
    st.markdown("""
    <style>
    /* Espaçamento só das linhas do tabuleiro (stHorizontalBlock aninhado) */
    div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] div[data-testid="column"] {
        padding: 0 0px !important;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] {
        gap: 2px !important;
    }

    /* Estilo só dos botões do tabuleiro */
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

    winner = check_winner(st.session_state.board)
    if winner:
        sym = "X" if winner == 1 else "O"
        st.success(f"{sym} Venceu!")
    elif (st.session_state.board != 0).all():
        st.info("Empate!")


    if st.button("Reiniciar"):
        st.session_state.board = np.zeros((3, 3), dtype=int)
        st.session_state.turn = 1

with col2:
    st.write(st.session_state.board)