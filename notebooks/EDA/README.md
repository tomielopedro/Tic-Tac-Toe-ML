# Análise Exploratória: Tic-Tac-Toe Endgame Dataset

## Validação Inicial

Verificação de jogos duplicados realizada através de: df.duplicated(subset=[f'pos_{i}' for i in range(1,10)]).any()
Resultado: Não há jogos duplicados no dataset. Todas as 958 configurações são únicas.

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

Após avaliar 4 abordagens possíveis, foi escolhida a Opção de: Geração Aleatória de Estados Intermediários com Validação.

Essa estratégia opera da seguinte forma:
- Para cada configuração de fim de jogo
- Remove-se aleatoriamente 1-3 posições preenchidas (deixando como vazio)
- Valida-se que a nova configuração não contém vitória
- Valida-se que tem pelo menos uma posição vazia
- Classifica-se como "Tem jogo"
- O processo é repetido N vezes por amostra original

## Próximos Passos

### 1. Padronização do Dataset

O dataset deve ser padronizado através de codificação numérica:
- 'x' transformar em 1
- 'o' transformar em -1
- 'b' transformar em 0

Esta transformação é necessária para que os algoritmos de classificação possam processar os dados adequadamente.

### 2. Expansão dos Casos de Empate

Atualmente o dataset contém apenas 16 casos de empate das 32 possibilidades de empate que existem no jogo da velha. Para obter os 32 casos, deve-se realizar espelhamento das configurações de empate existentes, transformando:
- x em o
- o em x

Mantendo a classe como "Empate". Isso resulta em mais 16 configurações de empate, totalizando 32 casos únicos de empate.

### 3. Implementar função gerar_estados_intermediarios(df, n_geracao_por_linha=3, seed=42)
   - Identificar posições preenchidas para cada linha
   - Remover aleatoriamente 1-3 posições
   - Validar resultado com função classificar()
   - Adicionar ao novo dataframe com classe "Tem jogo"

### 4. Parametrizar a geração
   - n_geracao_por_linha: número de variações por amostra (recomendado 3-5)
   - seed: manter em 42 para reprodutibilidade
   - min_remove/max_remove: manter em 1-3 para evitar estados muito vazios

### 5. Validar estados gerados
   - Verificar que nenhum tem vitória detectada
   - Verificar que todos têm posição vazia
   - Verificar ausência de duplicatas
   - Confirmar distribuição das 4 classes

### 6. Combinar as 4 classes em um único dataframe
   - Concatenar X venceu, O venceu, Empate e Tem jogo
   - Salvar dataset expandido

### 7. Análise comparativa das 4 classes
   - Distribuição
   - Padrões estratégicos por classe
   - Visualizações