# Tic-Tac-Toe — AI Analyst Lab

Aplicação interativa em **Streamlit** que combina o clássico Jogo da Velha com um laboratório de análise e comparação de modelos de Machine Learning. O usuário joga contra um bot aleatório enquanto um modelo treinado atua como "analista", classificando o estado do tabuleiro em tempo real. O sistema também permite simulações em massa (máquina vs máquina) para estressar os modelos, com salvamento automático de erros para retreinamento.

---

## Funcionalidades

### Jogo — Humano vs Bot

O jogador (X) enfrenta um bot de jogadas aleatórias (O) em uma interface visual com tabuleiro 3×3. A cada jogada, o modelo selecionado na sidebar avalia o tabuleiro e exibe sua previsão ao lado do status real. Ao final da partida, um card de acurácia mostra a performance do modelo durante aquele jogo. A partida é registrada automaticamente no histórico persistente.

### Simulação — Máquina vs Máquina

Duas máquinas jogam aleatoriamente entre si por N partidas (configurável de 1 a 500). O modelo selecionado analisa cada estado intermediário do tabuleiro. Quando o modelo erra uma previsão, a correção é salva automaticamente no dataset de correções com o nome do modelo, a previsão incorreta, a classe real e o timestamp. Ao final da simulação, o painel exibe acurácia global, média, desvio padrão, distribuição de resultados (X/O/Empate), gráfico de acurácia por partida, matriz de confusão dos erros agregada, e drill-down partida por partida com tabuleiro final e log jogada-a-jogada.

### Histórico de Partidas

Todas as partidas (manuais e simuladas) são salvas em `data/historico/historico_partidas.json`. A aba de histórico oferece filtros por modelo, resultado e origem (manual/simulação), KPIs (total de partidas, win rate, acurácia média, melhor acurácia), gráfico de dispersão da acurácia ao longo do tempo, tabela detalhada com todas as partidas, e drill-down jogada-a-jogada com mini-matriz de confusão dos erros de cada partida individual.

### Comparação de Modelos

Tabela comparativa com métricas por modelo: acurácia média, acurácia global, desvio padrão, mínimo, máximo, total de acertos e jogadas. Inclui gráfico de barras da acurácia global, análise de erros por modelo (matriz de confusão, distribuição por status real e por previsão incorreta), e comparação direta entre dois modelos com gráfico de evolução da acurácia ao longo das partidas.

### Dataset de Correções

Centraliza todas as amostras onde os modelos erraram. Cada registro inclui o estado do tabuleiro (9 posições), a classe correta, o modelo que errou, a previsão que ele deu e o timestamp. Alimentado de duas formas: manualmente pelo painel de retroalimentação durante o jogo, ou automaticamente pela simulação. A aba exibe KPIs (total de correções, modelos com correções, previsão errada mais comum), gráfico de correções por modelo e tabela editável. O sistema é retrocompatível com datasets antigos de 10 colunas.

### Carregamento Dinâmico de Modelos

O sistema rastreia automaticamente o diretório `models/`. Cada subpasta deve conter dois arquivos `.pkl`: o modelo treinado e o encoder (identificado pela presença de "encoder" no nome do arquivo). Novos modelos adicionados nas suas respectivas pastas aparecem instantaneamente no menu da sidebar sem necessidade de alterar o código.

---

## Estrutura do Projeto

```
Tic-Tac-Toe-ML/
├── app/
│   └── main.py                  # Aplicação Streamlit principal
├── models/
│   ├── mlp/                     # Exemplo: pasta com modelo MLP
│   │   ├── mlp_model.pkl
│   │   └── label_encoder.pkl
│   ├── random_forest/           # Exemplo: pasta com Random Forest
│   │   ├── rf_model.pkl
│   │   └── rf_encoder.pkl
│   └── .../                     # Outros modelos
├── data/
│   ├── historico/
│   │   └── historico_partidas.json
│   └── correcoes/
│       └── dataset_correcoes.csv
└── README.md
```

---

## Como Executar

```bash
pip install streamlit numpy pandas joblib scikit-learn
cd app
streamlit run main.py
```

---

## Formato dos Dados

### Tabuleiro

Matriz 3×3 codificada como vetor de 9 posições (`pos_1` a `pos_9`):

| Valor | Significado |
|-------|-------------|
| `1`   | X           |
| `-1`  | O           |
| `0`   | Vazio       |

### Classes de Previsão

O modelo classifica cada estado do tabuleiro em uma das quatro classes: `Tem jogo`, `X venceu`, `O venceu` ou `Empate`.

### Dataset de Correções (CSV)

| Coluna | Descrição |
|--------|-----------|
| `pos_1` a `pos_9` | Estado do tabuleiro |
| `classe` | Status real correto |
| `modelo_origem` | Nome do modelo que errou |
| `previsao_modelo` | O que o modelo previu |
| `timestamp` | Data/hora do registro |

### Histórico de Partidas (JSON)

Cada registro contém: `id`, `timestamp`, `modelo`, `resultado`, `acuracia_modelo`, `acertos`, `total_jogadas`, `log_jogadas` (lista de dicts com tabuleiro, previsão, real e acertou) e `origem` ("manual" ou "simulacao").

---

## Adicionando Novos Modelos

1. Treine o modelo usando as 9 features (`pos_1` a `pos_9`) e as 4 classes de saída.
2. Salve o modelo e o `LabelEncoder` como `.pkl` com `joblib.dump()`.
3. Crie uma pasta dentro de `models/` com o nome do modelo (ex: `models/svm/`).
4. Coloque os dois arquivos `.pkl` na pasta. O arquivo do encoder deve conter "encoder" no nome.
5. Reinicie o app — o modelo aparecerá automaticamente no seletor da sidebar.