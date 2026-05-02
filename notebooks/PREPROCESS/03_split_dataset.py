"""
Gera os splits físicos treino/validação/teste a partir do dataset pré-processado.

Saída: data/splits/{train,val,test}.csv

Estratégia:
    - Split estratificado por 'classe' (mantém a proporção das 4 classes nos 3 conjuntos)
    - Proporção 70% / 15% / 15%
    - random_state=42 para reprodutibilidade
    - Rodar UMA vez. Todos os notebooks de modelagem consomem desses arquivos.

Uso:
    cd notebooks/PREPROCESS
    python 03_split_dataset.py
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Caminhos relativos à raiz do projeto (script roda de qualquer lugar)
RAIZ = Path(__file__).resolve().parents[2]
ENTRADA = RAIZ / "data" / "processed" / "preprocessed_1.csv"
SAIDA = RAIZ / "data" / "splits"

PROPORCAO_TESTE = 0.15
PROPORCAO_VAL = 0.15  # Sobre o total. Logo 70% sobra para treino.
SEED = 42


def main():
    print(f"Lendo {ENTRADA.relative_to(RAIZ)}...")
    df = pd.read_csv(ENTRADA)
    print(f"Total: {len(df)} amostras\n")
    print("Distribuição original:")
    print(df["classe"].value_counts(), "\n")

    # 1) Separa o teste primeiro (15% do total)
    df_temp, df_test = train_test_split(
        df,
        test_size=PROPORCAO_TESTE,
        random_state=SEED,
        stratify=df["classe"],
    )

    # 2) Do que sobrou (85%), separa validação tal que vire 15% do total.
    #    val_size relativo = 0.15 / 0.85 ≈ 0.1765
    val_size_relativo = PROPORCAO_VAL / (1.0 - PROPORCAO_TESTE)
    df_train, df_val = train_test_split(
        df_temp,
        test_size=val_size_relativo,
        random_state=SEED,
        stratify=df_temp["classe"],
    )

    SAIDA.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(SAIDA / "train.csv", index=False)
    df_val.to_csv(SAIDA / "val.csv", index=False)
    df_test.to_csv(SAIDA / "test.csv", index=False)

    print("Splits gerados:")
    for nome, sub in [("train", df_train), ("val", df_val), ("test", df_test)]:
        pct = len(sub) / len(df) * 100
        print(f"  {nome:5s}: {len(sub):4d} amostras ({pct:5.1f}%)")
        print(sub["classe"].value_counts().to_string().replace("\n", "\n           "))
        print()

    print(f"Arquivos salvos em {SAIDA.relative_to(RAIZ)}/")


if __name__ == "__main__":
    main()
