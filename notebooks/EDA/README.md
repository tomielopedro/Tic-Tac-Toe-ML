# Análise Exploratória e Data Augmentation: Tic-Tac-Toe Endgame Dataset

## Visão Geral
Este projeto documenta a análise exploratória e a engenharia de dados (Data Augmentation) aplicadas a um dataset de configurações de fim de partida do Jogo da Velha (Tic-Tac-Toe). O objetivo principal foi adequar os dados aos requisitos de classificação em 4 classes distintas, gerando estados intermediários e expandindo casos sub-representados.

## 1. Análise Exploratória e Validação Inicial

### Verificação de Duplicatas
Foi realizada uma verificação de jogos duplicados considerando as 9 posições do tabuleiro (`pos_1` a `pos_9`).
- **Resultado:** Não foram encontradas duplicatas. Todas as 958 configurações originais são únicas.

### Achados da Análise (Dataset Original)
O dataset original continha 958 configurações legais de **fim de partida**. A análise identificou três classes distintas:
1. **X venceu:** 626 amostras (65%)
2. **O venceu:** 316 amostras (33%)
3. **Empate:** 16 amostras (2%)

*Descoberta importante:* A classe original "negative" do dataset era binária, agrupando tanto vitórias de O quanto empates. Essa classificação refletia apenas "X não venceu", falhando em diferenciar uma derrota de um empate.

### Estrutura dos Dados e Padrões
- O dataset contém 9 posições de tabuleiro onde cada célula pode ter: `'x'` (marca de X), `'o'` (marca de O) ou `'b'` (vazia).
- A validação das classes foi feita verificando as 8 combinações vencedoras (3 linhas, 3 colunas, 2 diagonais).
- **Padrões:** X aparece em maior quantidade em todas as posições, reflexo de ser o primeiro a jogar. A posição central (`pos_5`) mostrou-se estrategicamente vital: quando dominada por X, há alta correlação com vitória; quando dominada por O, há correlação negativa com a vitória de X.

---

## 2. O Problema Identificado

O requisito do trabalho exigia a classificação em **4 classes**, mas o dataset continha apenas 3. Faltava a classe **"Tem jogo"**, que representa estados intermediários onde ainda há posições vazias e nenhum jogador venceu. 

Essa classe não existia porque o conjunto de dados original focava estritamente em *endgames* (jogos finalizados). Além disso, havia uma sub-representação matemática da classe de Empate (contendo apenas 16 das 32 possibilidades reais de empate do jogo da velha).

---

## 3. Metodologia de Solução (Data Augmentation)

Para resolver as deficiências do dataset, foi implementado um pipeline em Python (Pandas/NumPy) focado em Geração Aleatória de Estados Intermediários com Validação Estrita. O processo foi dividido nas seguintes etapas:

### Passo 1: Padronização Numérica
Os dados categóricos foram convertidos para valores numéricos para facilitar o processamento matemático:
- `'x'` $\rightarrow$ `1`
- `'o'` $\rightarrow$ `-1`
- `'b'` $\rightarrow$ `0`

### Passo 2: Expansão dos Casos de Empate
Para obter os 32 casos totais de empate possíveis no jogo da velha, realizou-se o espelhamento matemático das 16 configurações existentes (multiplicando o tabuleiro padronizado por `-1`, ou seja, onde era X virou O, e vice-versa). 

### Passo 3: Geração de Estados Intermediários ("Tem jogo")
A partir das configurações de fim de jogo, novos estados foram gerados através de engenharia reversa:
1. Identificação das peças no tabuleiro original.
2. Remoção aleatória de 1 a 3 peças.
3. **Validação Estrita:** Garantia de que o novo estado gerado não contém uma vitória configurada.
4. **Validação de Turno:** Verificação matemática (`0 <= sum(tabuleiro) <= 1`) para garantir que o estado gerado obedece à regra de que "X" sempre joga primeiro.
5. Classificação como "Tem jogo".
6. O processo foi repetido e iterado até atingir uma volumetria satisfatória, seguido por uma etapa de eliminação de duplicatas globais.

---

## 4. Resultados: Distribuição Final do Dataset

Após a execução completa do pipeline de Data Augmentation, o dataset foi expandido com sucesso para atender aos requisitos das 4 classes de forma validada. A nova distribuição das classes ficou da seguinte forma:

| Classe | Quantidade de Amostras | Observação |
| :--- | :--- | :--- |
| **Tem jogo** | 626 | *Estados intermediários gerados e validados* |
| **X venceu** | 626 | *Mantido do dataset original* |
| **O venceu** | 316 | *Mantido do dataset original* |
| **Empate** | 32 | *Expandido (16 originais + 16 espelhados)* |
| **TOTAL** | **1600** | *Dataset finalizado pronto para treinamento* |

## 5. Próximos Passos
- Realizar análise comparativa detalhada (EDA) focada nas 4 classes.
- Extrair visualizações da nova distribuição.
- Iniciar a divisão dos dados (Train/Test Split) e o treinamento dos modelos de Machine Learning utilizando o novo dataset padronizado numericamente.