# T1 – Tic Tac Toe com ML

**Disciplina:** PUCRS – Inteligência Artificial  
**Professor:** Silvia Moraes  
**Código de identificação:** t32

---

## Enunciado

Neste primeiro trabalho prático da disciplina, você vai construir um **sistema de IA para o jogo da velha** em um tabuleiro clássico 3x3. O objetivo da IA **não é ser um dos players**, mas sim **verificar o estado de jogo**. A seguir serão descritas as etapas do trabalho.

---

## 1. Objetivo

A IA que você implementará deve receber como entrada o estado atual de um tabuleiro do jogo da velha e classificar esse estado em:

- **Tem jogo**
- **Jogador X venceu**
- **Jogador O venceu**
- **Empate**

---

## 2. Dataset

Acesse o link [https://archive.ics.uci.edu/dataset/101/tic+tac+toe+endgame](https://archive.ics.uci.edu/dataset/101/tic+tac+toe+endgame) e obtenha um dataset que possui instâncias do tabuleiro do jogo da velha.

### Requisitos do Dataset

- Analise o dataset e verifique se ele atende às necessidades do problema apresentado
- No caso de não atender plenamente, realize as adequações que julgar pertinentes (limpeza, transformação, etc.)
- **Registre** os eventuais problemas que você encontrou e **todos os passos** que você executou (justificando esses passos) para construir o dataset que será usado para construir a solução de IA
- Procure gerar um **dataset balanceado**
- De preferência, **não use todas as instâncias**, apenas as mais representativas
- Você pode iniciar com **200 amostras de cada classe**, quando possível
- **As variáveis devem ter em seus nomes o código 32**

---

## 3. Divisão do Dataset

Divida fisicamente o conjunto de dados em **treino, validação e teste**. Precisam ser os mesmos conjuntos nos experimentos, pois testaremos mais de um algoritmo de IA.

### Procedimento

- O conjunto de **validação** deve ser usado para definir os parâmetros mais adequados de cada algoritmo
- A partir dessas definições, use o conjunto de **teste** para avaliar de fato o desempenho da IA
- **Alternativa:** Divida o conjunto em treino e teste, e aplique **validação cruzada** no conjunto de treino
- O melhor modelo encontrado deve ter seu desempenho avaliado a partir do conjunto de teste
- **As variáveis devem ter em seus nomes o código 32**

---

## 4. Solução de IA

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
- **As variáveis devem ter em seus nomes o código 32**

---

## 5. Front End

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
- **As variáveis devem ter em seus nomes o código 32**

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
| **Dataset** (documentado) | 1,0 |
| **Soluções de IA e documentação** (1,0 por algoritmo, configuração, testes e análise de resultados) | 5,0 |
| **Front End** (com mensagens da IA e score) | 1,0 |
| **Relatório** (formato ppt): introdução, dataset e suas modificações, desenvolvimento das soluções, decisões e justificativas, comparações, resultados obtidos e conclusão | 2,0 |
| **TOTAL** | **10,0** |

---

## Observações Importantes

- ⚠️ **Código incorreto**, ausência na apresentação (não justificada), **falta de domínio** durante apresentação e **não cumprimento do enunciado** provocam **decréscimo na nota**
- ⚠️ **Cópia de trabalhos de colegas zeram o trabalho**
- ✅ **Indique as ferramentas de IA** que você usou, especificando **onde foram usadas**

---

## Relatório - Conteúdo Esperado (PPT)

O relatório em formato PowerPoint deve incluir:

1. **Introdução** - contexto e objetivo do trabalho
2. **Dataset** - número de amostras por classe e modificações realizadas
3. **Desenvolvimento das Soluções** - algoritmos, parametrização e justificativas
4. **Comparações** - tabelas e gráficos comparativos
5. **Resultados** - obtidos durante treinamento/validação/teste e interação via Front End
6. **Conclusão** - dificuldades encontradas e ganhos obtidos

---