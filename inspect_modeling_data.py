from pathlib import Path
import os
import pandas as pd


def find_data_dir() -> Path:
    cwd = Path.cwd().resolve()
    script_dir = Path(__file__).resolve().parent

    for base in [cwd, *cwd.parents, script_dir, *script_dir.parents]:
        if base.name.lower() == "data":
            return base
        p = base / "Data"
        if p.exists():
            return p

    raise FileNotFoundError("Pasta Data não encontrada.")


DATA_DIR = find_data_dir()
path = DATA_DIR / "Modeling" / "dataset_modelagem.csv"

df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")

print("=" * 90)
print("DATASET DE MODELAGEM")
print("=" * 90)
print(f"Arquivo: {path}")
print(f"Linhas: {len(df):,}")
print(f"Colunas: {len(df.columns):,}")
print()

print("TARGET:")
print(df["risco_nao_alfabetizado"].value_counts(dropna=False))
print()
print(
    "Taxa de risco:",
    round(100 * df["risco_nao_alfabetizado"].mean(), 2),
    "%"
)
print()

print("=" * 90)
print("COLUNAS E TIPOS")
print("=" * 90)
for col in df.columns:
    print(
        f"{col:45} | {str(df[col].dtype):12} | "
        f"nulos={df[col].isna().sum():5} | "
        f"unicos={df[col].nunique(dropna=True):5}"
    )

print()
print("=" * 90)
print("TOP 30 COLUNAS COM MAIS NULOS")
print("=" * 90)

missing = (
    df.isna().mean()
    .mul(100)
    .sort_values(ascending=False)
    .head(30)
)

print(missing.to_string())
