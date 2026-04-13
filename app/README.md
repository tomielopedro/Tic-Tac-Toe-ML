# Tic-Tac-Toe vs Bot com Análise de IA (MLP)

Este projeto é uma aplicação interativa desenvolvida em **Streamlit** onde o usuário joga o clássico Jogo da Velha (Tic-Tac-Toe) contra um bot. O grande diferencial do projeto é a integração com uma **Rede Neural Artificial (MLP - Multi-Layer Perceptron)** treinada para atuar como "analista" da partida, avaliando o tabuleiro em tempo real e prevendo o status do jogo.

## Principais Funcionalidades

- **Jogo Interativo:** Interface gráfica fluida para jogar contra um bot de jogadas aleatórias.
- **Análise em Tempo Real:** A cada clique, o tabuleiro é convertido em matriz e enviado para a IA, que classifica o estado atual em: `Tem jogo`, `X venceu`, `O venceu` ou `Empate`.
- **Carregamento Dinâmico de Modelos:** O sistema rastreia automaticamente o diretório de modelos do projeto. Novos modelos treinados (como Random Forest ou SVM) adicionados em suas respectivas pastas aparecerão instantaneamente no menu do aplicativo, sem necessidade de alterar o código.
- **Active Learning (Treinamento Contínuo):** Um painel integrado permite ao usuário atuar como supervisor da IA. Se o modelo errar uma previsão, o usuário pode reportar o status correto. O sistema salva automaticamente essa nova amostra em um CSV (`dataset_correcoes.csv`), criando um banco de dados focado nas fraquezas do modelo para re-treinamentos futuros.

---