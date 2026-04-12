# Pipeline de Pré-processamento e Data Augmentation: Tic-Tac-Toe

Este documento detalha o código desenvolvido no notebook **`02_preprocessing.ipynb`** para realizar o pré-processamento e o aumento de dados (Data Augmentation) do dataset *Tic-Tac-Toe Endgame*. 

O objetivo prático deste notebook é transformar um dataset focado apenas em finais de jogo (3 classes desbalanceadas) em um dataset numérico, balanceado e contendo 4 classes, incluindo estados intermediários da partida ("Tem jogo"). O pipeline consome os dados analisados na etapa anterior (`eda_1.csv`) e gera um novo arquivo balanceado e pronto para modelagem (`preprocessed_1.csv`).

---

## Arquitetura do Código: Funções e Objetivos

O pipeline foi construído de forma modular. Abaixo está a explicação de cada função e seu propósito prático dentro do fluxo de dados:

### 1. `classificar_vitoria(board)`
**Objetivo Prático:** Atuar como o "juiz" do jogo. É a principal trava de segurança para garantir que não criemos estados inválidos.
* **Como funciona:** Recebe um tabuleiro numérico (array de 9 posições). Ela soma as linhas, colunas e diagonais mapeadas na constante `WIN_LINES`. 
* **Regra Matemática:** Como X=1 e O=-1, qualquer linha com soma igual a `3` indica vitória de X, e `-3` indica vitória de O. Retorna `0` se não houver vencedor.

### 2. `padronizar_dataset(df)`
**Objetivo Prático:** Converter os dados categóricos para numéricos. Isso é obrigatório para que os algoritmos de Machine Learning consigam processar os dados, além de facilitar incrivelmente a lógica de Data Augmentation.
* **Transformação:** * `'x'` (Marca de X) $\rightarrow$ `1`
  * `'o'` (Marca de O) $\rightarrow$ `-1`
  * `'b'` (Vazio) $\rightarrow$ `0`

### 3. `expandir_empates(df)`
**Objetivo Prático:** Dobrar o tamanho da classe minoritária "Empate" usando as regras lógicas do jogo, passando de 16 para 32 amostras (o total de empates possíveis).
* **Como funciona:** Como o dataset já está numérico, a função simplesmente clona os casos de empate e os multiplica por `-1`. Isso cria um "espelhamento perfeito", onde todas as peças "X" viram "O" e vice-versa, gerando configurações inéditas e válidas de empate.

### 4. `gerar_estados_intermediarios(df, n_geracao_por_linha, seed)`
**Objetivo Prático:** Criar do zero a classe ausente "Tem jogo" (estados onde o tabuleiro não está cheio e ninguém venceu ainda), aplicando engenharia reversa nos jogos finalizados.
* **Como funciona:**
  1. Copia um estado final do tabuleiro.
  2. Identifica onde existem peças e remove aleatoriamente de 1 a 3 peças (transformando em `0`).
  3. **Validação 1:** Checa usando `classificar_vitoria` se a remoção desfez qualquer vitória.
  4. **Validação 2 (Turno Válido):** Checa se a soma total do tabuleiro resulta em `0` ou `1`. Como o "X" (1) joga primeiro, um tabuleiro no meio do jogo só pode estar empatado em peças (soma 0) ou ter um X a mais (soma 1). Isso evita a criação de cenários impossíveis.

### 5. `pipeline_data_augmentation(df)`
**Objetivo Prático:** Orquestrar a execução de todas as funções anteriores, garantir a limpeza final dos dados e realizar o **balanceamento das classes**.
* **Como funciona:**
  1. Executa a padronização e a expansão de empates.
  2. Executa a geração de estados intermediários pedindo uma quantidade folgada de amostras.
  3. **Balanceamento Automático:** Identifica o tamanho da maior classe existente (X venceu, com 626 linhas) e recorta a nova classe "Tem jogo" para ter exatamente esse mesmo limite. Isso impede que a geração aleatória cause um super-desbalanceamento.
  4. Concatena tudo em um DataFrame único e executa um `drop_duplicates` global para garantir que não haja vazamento de dados entre classes.

---