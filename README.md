# Tic-Tac-Toe com Machine Learning

**Disciplina:** PUCRS – Inteligência Artificial  
**Professor:** Silvia Moraes  
**Código de identificação:** t32

---

## Índice

1. [Enunciado](#enunciado)
2. [Objetivo](#objetivo)
3. [Dataset](#dataset)
4. [Divisão do Dataset](#divisão-do-dataset)
5. [Solução de IA](#solução-de-ia)
6. [Front End](#front-end)
7. [Definições e Critérios](#definições-e-critérios)
8. [Pontuação](#pontuação)
9. [Relatório - Conteúdo Esperado](#relatório---conteúdo-esperado-ppt)
10. [Análise Exploratória do Dataset](#análise-exploratória-tic-tac-toe-endgame-dataset)
11. [Pipeline de Pré-processamento](#pipeline-de-pré-processamento-e-data-augmentation-tic-tac-toe)
12. [Aplicação Streamlit](#aplicação-streamlit-tic-tac-toe-vs-bot-com-análise-de-ia-mlp)
13. [Estrutura do Projeto](#estrutura-do-projeto)

---

## Enunciado

Neste primeiro trabalho prático da disciplina, você vai construir um **sistema de IA para o jogo da velha** em um tabuleiro clássico 3x3. O objetivo da IA **não é ser um dos players**, mas sim **verificar o estado de jogo**. A seguir serão descritas as etapas do trabalho.

---

## Objetivo

A IA que você implementará deve receber como entrada o estado atual de um tabuleiro do jogo da velha e classificar esse estado em:

- **Tem jogo**
- **Jogador X venceu**
- **Jogador O venceu**
- **Empate**

---

## Dataset

Acesse o link [https://archive.ics.uci.edu/dataset/101/tic+tac+toe+endgame](https://archive.ics.uci.edu/dataset/101/tic+tac+toe+endgame) e obtenha um dataset que possui instâncias do tabuleiro do jogo da velha.

### Requisitos do Dataset

- Analise o dataset e verifique se ele atende às necessidades do problema apresentado
- No caso de não atender plenamente, realize as adequações que julgar pertinentes (limpeza, transformação, etc.)
- **Registre** os eventuais problemas que você encontrou e **todos os passos** que você executou (justificando esses passos) para construir o dataset que será usado para construir a solução de IA
- Procure gerar um **dataset balanceado**
- De preferência, **não use todas as instâncias**, apenas as mais representativas
- Você pode iniciar com **200 amostras de cada classe**, quando possível

---

## Divisão do Dataset

Divida fisicamente o conjunto de dados em **treino, validação e teste**. Precisam ser os mesmos conjuntos nos experimentos, pois testaremos mais de um algoritmo de IA.

### Procedimento

- O conjunto de **validação** deve ser usado para definir os parâmetros mais adequados de cada algoritmo
- A partir dessas definições, use o conjunto de **teste** para avaliar de fato o desempenho da IA
- **Alternativa:** Divida o conjunto em treino e teste, e aplique **validação cruzada** no conjunto de treino
- O melhor modelo encontrado deve ter seu desempenho avaliado a partir do conjunto de teste

---

## Solução de IA

Construa a sua solução testando **ao menos 5 algoritmos classificadores**. Dentre os 5, deve ter:

- **k-NN** (obrigatório)
- **MLP** (obrigatório)
- **Árvores de decisão** (obrigatório)
- **Dois outros algoritmos** de livre escolha

### Requisitos

- Inclua em seu relatório uma **pequena explicação** de como funcionam os dois algoritmos de livre escolha
- **Todas as decisões referentes a parâmetros** desses algoritmos devem ser apresentadas e justificadas no seu relatório
- No caso da **MLP**, não esqueça de informar a **topologia usada**
- **Meça os seus resultados** utilizando:
  - Acurácia
  - Precision
  - Recall
  - F-measure
- **Busque bons resultados** e **evite overfitting** (procure as melhores configurações e parâmetros para os algoritmos)
- **Compare os resultados** e **escolha o melhor algoritmo** para o problema
- Mostre a sua comparação usando **tabelas e gráficos**
- **Justifique sua escolha** no texto do seu relatório

---

## Front End

Construa um **front end mínimo** (não precisa ser gráfico) para o jogo da velha, onde **dois players possam interagir**. Um player deve ser **humano** e o outro a **máquina jogando de forma aleatória**.

### Funcionalidades

- A cada turno (a cada jogada de usuário/computador), a **solução de IA escolhida** deve indicar:
  - Se um dos jogadores ganhou
  - Se houve empate
  - Se ainda há jogo
- A partir da **saída do seu algoritmo de IA**, seu front end deve:
  - Dar ou não seguimento ao jogo
  - Fornecer os feedbacks necessários (mensagens ao usuário sobre o estado do jogo)
- **Contabilize acertos e erros** da solução durante a interação
- **Meça a acurácia** da solução durante as interações com os usuários
- **Registre** isso no seu relatório
- **Procedimento de teste:**
  - Encerre o jogo quando a IA não detectar o fim de jogo
  - Continue o jogo quando a IA detectar o fim do jogo incorretamente

---

## Definições e Critérios

- Os **grupos podem ser de até 5 alunos**
- **Distribua as atividades** entre os integrantes do grupo de forma que todos trabalhem (1 algoritmo de IA por aluno, ao menos)
- **Se inscreva no moodle**
- Alunos que não formarem grupo terão grupo definido pelo professor
- **Data de entrega e apresentação:** no cronograma disponível no moodle
- Na data da **apresentação**, **todos os integrantes do grupo devem estar presentes**
- A **avaliação não é apenas sobre o que foi entregue**, mas também sobre o **domínio/conhecimento demonstrado** pelos integrantes durante a apresentação
- O **desconhecimento ou falta de domínio** sobre o código e das funcionalidades implementadas pode **zerar ou reduzir consideravelmente a nota**

---

## Pontuação

| Item | Pontos |
|------|--------|
| Dataset (documentado) | 1,0 |
| Soluções de IA e documentação (1,0 por algoritmo, configuração, testes e análise de resultados) | 5,0 |
| Front End (com mensagens da IA e score) | 1,0 |
| Relatório (formato ppt): introdução, dataset e suas modificações, desenvolvimento das soluções, decisões e justificativas, comparações, resultados obtidos e conclusão | 2,0 |
| **TOTAL** | **10,0** |

---

## Observações Importantes

- **Código incorreto**, ausência na apresentação (não justificada), **falta de domínio** durante apresentação e **não cumprimento do enunciado** provocam **decréscimo na nota**
- **Cópia de trabalhos de colegas zeram o trabalho**
- **Indique as ferramentas de IA** que você usou, especificando **onde foram usadas**

---

## Relatório - Conteúdo Esperado (PPT) {#relatório---conteúdo-esperado-ppt}

O relatório em formato PowerPoint deve incluir:

1. **Introdução** - contexto e objetivo do trabalho
2. **Dataset** - número de amostras por classe e modificações realizadas
3. **Desenvolvimento das Soluções** - algoritmos, parametrização e justificativas
4. **Comparações** - tabelas e gráficos comparativos
5. **Resultados** - obtidos durante treinamento/validação/teste e interação via Front End
6. **Conclusão** - dificuldades encontradas e ganhos obtidos

---

## Análise Exploratória: Tic-Tac-Toe Endgame Dataset {#análise-exploratória-tic-tac-toe-endgame-dataset}

## Validação Inicial

Verificação de jogos duplicados realizada através de: `df.duplicated(subset=[f'pos_{i}' for i in range(1,10)]).any()`

**Resultado:** Não há jogos duplicados no dataset. Todas as 958 configurações são únicas.

---

## Achados da Análise

O dataset contém 958 configurações legais de fim de partida do jogo da velha. A análise exploratória identificou três classes distintas de resultado:

1. **X venceu:** 626 amostras (65%)
2. **O venceu:** 316 amostras (33%)
3. **Empate:** 16 amostras (2%)

**Descoberta importante:** a classe original "negative" do dataset era binária, agrupando tanto vitórias de O quanto empates. Essa classificação refletia apenas "X não venceu", não diferenciando entre derrota e empate.

---

## Estrutura dos Dados

O dataset contém 9 posições de tabuleiro (pos_1 a pos_9) onde cada célula pode ter um dos três valores:

- `'x'`: posição com marca de X
- `'o'`: posição com marca de O
- `'b'`: posição vazia

A validação das classes foi feita verificando as 8 combinações vencedoras do jogo (3 linhas, 3 colunas, 2 diagonais).

---

## Padrões Identificados

A análise de frequência mostrou que X aparece em maior quantidade em todas as posições, reflexo de X jogar sempre primeiro em um jogo real. Correlações identificadas revelaram que a posição central (pos_5) é estrategicamente importante: quando X ocupa pos_5, há maior correlação com vitória; quando O ocupa, há correlação negativa com vitória de X.

O dataset apresenta um leve desbalanceamento, com prevalência da classe positiva (65% vs 35%).

---

## Problema Identificado

O requisito do trabalho exige classificação em 4 classes, mas o dataset atual contém apenas 3 classes resultantes. Falta a classe "Tem jogo", que representaria estados intermediários do jogo onde ainda há posições vazias e nenhum jogador venceu ainda.

Essa classe não existe no dataset porque o conjunto contém apenas "endgame" (fim de jogo) - todas as 958 configurações representam estados finais onde o jogo terminou.

---

## Estratégia Definida

Após avaliar 4 abordagens possíveis, foi escolhida a **Opção de Geração Aleatória de Estados Intermediários com Validação**.

Essa estratégia opera da seguinte forma:

- Para cada configuração de fim de jogo
- Remove-se aleatoriamente 1-3 posições preenchidas (deixando como vazio)
- Valida-se que a nova configuração não contém vitória
- Valida-se que tem pelo menos uma posição vazia
- Classifica-se como "Tem jogo"
- O processo é repetido N vezes por amostra original

---

## Pipeline de Pré-processamento e Data Augmentation: Tic-Tac-Toe {#pipeline-de-pré-processamento-e-data-augmentation-tic-tac-toe}

Este documento detalha o código desenvolvido no notebook **`02_preprocessing.ipynb`** para realizar o pré-processamento e o aumento de dados (Data Augmentation) do dataset *Tic-Tac-Toe Endgame*.

O objetivo prático deste notebook é transformar um dataset focado apenas em finais de jogo (3 classes desbalanceadas) em um dataset numérico, balanceado e contendo 4 classes, incluindo estados intermediários da partida ("Tem jogo"). O pipeline consome os dados analisados na etapa anterior (`eda_1.csv`) e gera um novo arquivo balanceado e pronto para modelagem (`preprocessed_1.csv`).

---

## Arquitetura do Código: Funções e Objetivos

O pipeline foi construído de forma modular. Abaixo está a explicação de cada função e seu propósito prático dentro do fluxo de dados:

### 1. `classificar_vitoria(board)`

**Objetivo Prático:** Atuar como o "juiz" do jogo. É a principal trava de segurança para garantir que não criemos estados inválidos.

- **Como funciona:** Recebe um tabuleiro numérico (array de 9 posições). Ela soma as linhas, colunas e diagonais mapeadas na constante `WIN_LINES`.
- **Regra Matemática:** Como X=1 e O=-1, qualquer linha com soma igual a `3` indica vitória de X, e `-3` indica vitória de O. Retorna `0` se não houver vencedor.

### 2. `padronizar_dataset(df)`

**Objetivo Prático:** Converter os dados categóricos para numéricos. Isso é obrigatório para que os algoritmos de Machine Learning consigam processar os dados, além de facilitar incrivelmente a lógica de Data Augmentation.

- **Transformação:**
  - `'x'` (Marca de X) → `1`
  - `'o'` (Marca de O) → `-1`
  - `'b'` (Vazio) → `0`

### 3. `expandir_empates(df)`

**Objetivo Prático:** Dobrar o tamanho da classe minoritária "Empate" usando as regras lógicas do jogo, passando de 16 para 32 amostras (o total de empates possíveis).

- **Como funciona:** Como o dataset já está numérico, a função simplesmente clona os casos de empate e os multiplica por `-1`. Isso cria um "espelhamento perfeito", onde todas as peças "X" viram "O" e vice-versa, gerando configurações inéditas e válidas de empate.

### 4. `gerar_estados_intermediarios(df, n_geracao_por_linha, seed)`

**Objetivo Prático:** Criar do zero a classe ausente "Tem jogo" (estados onde o tabuleiro não está cheio e ninguém venceu ainda), aplicando engenharia reversa nos jogos finalizados.

- **Como funciona:**
  1. Copia um estado final do tabuleiro.
  2. Identifica onde existem peças e remove aleatoriamente de 1 a 3 peças (transformando em `0`).
  3. **Validação 1:** Checa usando `classificar_vitoria` se a remoção desfez qualquer vitória.
  4. **Validação 2 (Turno Válido):** Checa se a soma total do tabuleiro resulta em `0` ou `1`. Como o "X" (1) joga primeiro, um tabuleiro no meio do jogo só pode estar empatado em peças (soma 0) ou ter um X a mais (soma 1). Isso evita a criação de cenários impossíveis.

### 5. `pipeline_data_augmentation(df)`

**Objetivo Prático:** Orquestrar a execução de todas as funções anteriores, garantir a limpeza final dos dados e realizar o **balanceamento das classes**.

- **Como funciona:**
  1. Executa a padronização e a expansão de empates.
  2. Executa a geração de estados intermediários pedindo uma quantidade folgada de amostras.
  3. **Balanceamento Automático:** Identifica o tamanho da maior classe existente (X venceu, com 626 linhas) e recorta a nova classe "Tem jogo" para ter exatamente esse mesmo limite. Isso impede que a geração aleatória cause um super-desbalanceamento.
  4. Concatena tudo em um DataFrame único e executa um `drop_duplicates` global para garantir que não haja vazamento de dados entre classes.

---

## Aplicação Streamlit: Tic-Tac-Toe vs Bot com Análise de IA (MLP) {#aplicação-streamlit-tic-tac-toe-vs-bot-com-análise-de-ia-mlp}

Este projeto é uma aplicação interativa desenvolvida em **Streamlit** onde o usuário joga o clássico Jogo da Velha (Tic-Tac-Toe) contra um bot. O grande diferencial do projeto é a integração com uma **Rede Neural Artificial (MLP - Multi-Layer Perceptron)** treinada para atuar como "analista" da partida, avaliando o tabuleiro em tempo real e prevendo o status do jogo.

---

## Principais Funcionalidades

- **Jogo Interativo:** Interface gráfica fluida para jogar contra um bot de jogadas aleatórias.
- **Análise em Tempo Real:** A cada clique, o tabuleiro é convertido em matriz e enviado para a IA, que classifica o estado atual em: `Tem jogo`, `X venceu`, `O venceu` ou `Empate`.
- **Carregamento Dinâmico de Modelos:** O sistema rastreia automaticamente o diretório de modelos do projeto. Novos modelos treinados (como Random Forest ou SVM) adicionados em suas respectivas pastas aparecerão instantaneamente no menu do aplicativo, sem necessidade de alterar o código.
- **Active Learning (Treinamento Contínuo):** Um painel integrado permite ao usuário atuar como supervisor da IA. Se o modelo errar uma previsão, o usuário pode reportar o status correto. O sistema salva automaticamente essa nova amostra em um CSV (`dataset_correcoes.csv`), criando um banco de dados focado nas fraquezas do modelo para re-treinamentos futuros.

---

## Estrutura do Projeto {#estrutura-do-projeto}

```
.
├── README.md
├── app
│   ├── README.md
│   └── main.py
├── data
│   ├── correcoes
│   │   └── dataset_correcoes.csv
│   ├── processed
│   │   ├── eda_1.csv
│   │   └── preprocessed_1.csv
│   ├── raw
│   │   ├── tic-tac-toe.data
│   │   └── tic-tac-toe.names
│   └── splits
├── models
│   └── MLP
│       ├── mlp_label_encoder.pkl
│       └── mlp_model.pkl
├── notebooks
│   ├── EDA
│   │   ├── 01_eda.ipynb
│   │   └── README.md
│   ├── MODELING
│   │   └── 03_modeling.ipynb
│   └── PREPROCESS
│       ├── 02_preprocessing.ipynb
│       └── README.MD
└── requirements.txt
```

---