# Análise Exploratória: Tic-Tac-Toe Endgame Dataset

## Achados da Análise

O dataset contém 958 configurações legais de fim de partida do jogo da velha. A análise exploratória identificou três classes distintas de resultado:

1. X venceu: 626 amostras (65%)
2. O venceu: 316 amostras (33%)
3. Empate: 16 amostras (2%)

Descoberta importante: a classe original "negative" do dataset era binária, agrupando tanto vitórias de O quanto empates. Essa classificação refletia apenas "X não venceu", não diferenciando entre derrota e empate.

## Estrutura dos Dados

O dataset contém 9 posições de tabuleiro (pos_1 a pos_9) onde cada célula pode ter um dos três valores:
- 'x': posição com marca de X
- 'o': posição com marca de O
- 'b': posição vazia

A validação das classes foi feita verificando as 8 combinações vencedoras do jogo (3 linhas, 3 colunas, 2 diagonais).

## Padrões Identificados

A análise de frequência mostrou que X aparece em maior quantidade em todas as posições, reflexo de X jogar sempre primeiro em um jogo real. Correlações identificadas revelaram que a posição central (pos_5) é estrategicamente importante: quando X ocupa pos_5, há maior correlação com vitória; quando O ocupa, há correlação negativa com vitória de X.

O dataset apresenta um leve desbalanceamento, com prevalência da classe positiva (65% vs 35%).

## Problema Identificado

O requisito do trabalho exige classificação em 4 classes, mas o dataset atual contém apenas 3 classes resultantes. Falta a classe "Tem jogo", que representaria estados intermediários do jogo onde ainda há posições vazias e nenhum jogador venceu ainda.

Essa classe não existe no dataset porque o conjunto contém apenas "endgame" (fim de jogo) - todas as 958 configurações representam estados finais onde o jogo terminou.

## Estratégia Definida:

Após avaliar 4 abordagens possíveis, foi escolhida a opção de: Geração Aleatória de Estados Intermediários com Validação.

Essa estratégia opera da seguinte forma:
- Para cada configuração de fim de jogo
- Remove-se aleatoriamente 1-3 posições preenchidas (deixando como vazio)
- Valida-se que a nova configuração não contém vitória
- Valida-se que tem pelo menos uma posição vazia
- Classifica-se como "Tem jogo"
- O processo é repetido N vezes por amostra original

Justificativa: simplicidade de implementação, elimina dependencia de datasets externos. 
