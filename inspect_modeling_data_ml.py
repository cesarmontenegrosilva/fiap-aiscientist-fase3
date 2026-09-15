from pathlib import Path
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
path = DATA_DIR / "Modeling" / "dataset_modelagem_ml.csv"

df = pd.read_csv(
    path,
    low_memory=False,
    encoding="utf-8-sig",
)

TARGET = "risco_nao_alfabetizado"

AUDIT = {
    "id_aluno",
    "id_escola",
    "codigo_municipio",
    "nome_municipio",
    "alfabetizado_oficial",
    TARGET,
}

features = [
    c for c in df.columns
    if c not in AUDIT
]

print("=" * 90)
print("DATASET CURADO PARA MACHINE LEARNING")
print("=" * 90)
print(f"Arquivo: {path}")
print(f"Linhas: {len(df):,}")
print(f"Colunas totais: {len(df.columns):,}")
print(f"Features efetivas: {len(features):,}")
print()

print("TARGET:")
print(df[TARGET].value_counts(dropna=False))
print()
print(
    "Taxa de risco:",
    round(100 * df[TARGET].mean(), 2),
    "%"
)
print()

print("=" * 90)
print("FEATURES")
print("=" * 90)

for col in features:
    print(
        f"{col:45} | {str(df[col].dtype):12} | "
        f"nulos={df[col].isna().sum():5} | "
        f"nulos%={100*df[col].isna().mean():6.2f} | "
        f"unicos={df[col].nunique(dropna=True):5}"
    )

print()
print("=" * 90)
print("CHECAGENS")
print("=" * 90)

constants = [
    c for c in features
    if df[c].nunique(dropna=True) <= 1
]

high_missing = [
    (
        c,
        round(100 * df[c].isna().mean(), 2)
    )
    for c in features
    if df[c].isna().mean() > 0.20
]

print("Features constantes:", constants if constants else "nenhuma")
print("Features com >20% nulos:", high_missing if high_missing else "nenhuma")
