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
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

pred = pd.read_csv(
    REPORTS_DIR / "predictions_test.csv",
    encoding="utf-8-sig",
)

required = [
    "codigo_municipio",
    "prob_risco_nao_alfabetizado",
]

missing = [
    c for c in required
    if c not in pred.columns
]

if missing:
    raise KeyError(
        f"Colunas ausentes em predictions_test.csv: {missing}"
    )

group_cols = ["codigo_municipio"]

for c in ["nome_municipio", "sigla_uf"]:
    if c in pred.columns:
        group_cols.append(c)

municipal = (
    pred.groupby(
        group_cols,
        dropna=False,
    )
    .agg(
        alunos_teste=(
            "prob_risco_nao_alfabetizado",
            "size",
        ),
        risco_medio=(
            "prob_risco_nao_alfabetizado",
            "mean",
        ),
        alunos_preditos_risco=(
            "y_pred",
            "sum",
        ),
        alunos_reais_risco=(
            "y_real",
            "sum",
        ),
    )
    .reset_index()
)

municipal["risco_medio_pct"] = (
    100 * municipal["risco_medio"]
)

# Evita ranking com pouquíssimos alunos.
municipal["elegivel_ranking"] = (
    municipal["alunos_teste"] >= 5
)

ranking = (
    municipal[
        municipal["elegivel_ranking"]
    ]
    .sort_values(
        "risco_medio",
        ascending=False,
    )
    .reset_index(drop=True)
)

ranking["ranking_risco_amostra"] = (
    ranking.index + 1
)

municipal.to_csv(
    REPORTS_DIR / "risco_municipal_todos.csv",
    index=False,
    encoding="utf-8-sig",
)

ranking.to_csv(
    REPORTS_DIR / "ranking_risco_municipal.csv",
    index=False,
    encoding="utf-8-sig",
)

print(
    "Agregação municipal concluída. "
    "O ranking é referente apenas ao conjunto de teste/amostra."
)
