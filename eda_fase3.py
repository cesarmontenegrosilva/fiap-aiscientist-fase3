from __future__ import annotations

from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
MODELING_DIR = DATA_DIR / "Modeling"
REPORTS_DIR = DATA_DIR / "reports"
IMAGES_DIR = DATA_DIR / "images"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

path = MODELING_DIR / "dataset_modelagem.csv"

df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")

TARGET = "risco_nao_alfabetizado"


# ============================================================
# 1. RESUMO GERAL
# ============================================================

summary = pd.DataFrame(
    [
        {
            "linhas": len(df),
            "colunas": len(df.columns),
            "duplicadas": int(df.duplicated().sum()),
            "target_0": int((df[TARGET] == 0).sum()),
            "target_1": int((df[TARGET] == 1).sum()),
            "taxa_target_1_pct": round(100 * df[TARGET].mean(), 2),
        }
    ]
)

summary.to_csv(
    REPORTS_DIR / "eda_resumo_geral.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 2. NULOS
# ============================================================

missing = pd.DataFrame(
    {
        "coluna": df.columns,
        "nulos": [df[c].isna().sum() for c in df.columns],
        "nulos_pct": [100 * df[c].isna().mean() for c in df.columns],
        "unicos": [df[c].nunique(dropna=True) for c in df.columns],
        "dtype": [str(df[c].dtype) for c in df.columns],
    }
).sort_values(
    "nulos_pct",
    ascending=False,
)

missing.to_csv(
    REPORTS_DIR / "eda_valores_ausentes.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 3. TARGET
# ============================================================

target_counts = (
    df[TARGET]
    .value_counts(dropna=False)
    .rename_axis("classe")
    .reset_index(name="quantidade")
)

target_counts["percentual"] = (
    100 * target_counts["quantidade"] / len(df)
)

target_counts.to_csv(
    REPORTS_DIR / "eda_distribuicao_target.csv",
    index=False,
    encoding="utf-8-sig",
)

fig, ax = plt.subplots(figsize=(7, 5))
target_counts.plot(
    x="classe",
    y="quantidade",
    kind="bar",
    legend=False,
    ax=ax,
)
ax.set_title("Distribuição do target: risco de não alfabetização")
ax.set_xlabel("Classe")
ax.set_ylabel("Quantidade de alunos")
fig.tight_layout()
fig.savefig(
    IMAGES_DIR / "eda_target_distribution.png",
    dpi=160,
)
plt.close(fig)


# ============================================================
# 4. NULOS - GRÁFICO
# ============================================================

top_missing = missing.head(20).sort_values("nulos_pct")

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(
    top_missing["coluna"],
    top_missing["nulos_pct"],
)
ax.set_title("Top 20 variáveis por percentual de valores ausentes")
ax.set_xlabel("% ausente")
fig.tight_layout()
fig.savefig(
    IMAGES_DIR / "eda_missing_values.png",
    dpi=160,
)
plt.close(fig)


# ============================================================
# 5. RELAÇÃO NUMÉRICAS X TARGET
# ============================================================

numeric_cols = [
    c for c in df.select_dtypes(include=np.number).columns
    if c != TARGET
]

corr_rows = []

for col in numeric_cols:
    valid = df[[col, TARGET]].dropna()

    if len(valid) < 20:
        continue

    if valid[col].nunique() <= 1:
        continue

    corr = valid[col].corr(valid[TARGET])

    corr_rows.append(
        {
            "variavel": col,
            "correlacao_com_target": corr,
            "abs_correlacao": abs(corr) if pd.notna(corr) else np.nan,
        }
    )

corr_df = (
    pd.DataFrame(corr_rows)
    .sort_values(
        "abs_correlacao",
        ascending=False,
    )
)

corr_df.to_csv(
    REPORTS_DIR / "eda_correlacoes_target.csv",
    index=False,
    encoding="utf-8-sig",
)

if not corr_df.empty:
    top_corr = corr_df.head(15).sort_values(
        "abs_correlacao"
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        top_corr["variavel"],
        top_corr["correlacao_com_target"],
    )
    ax.set_title(
        "Variáveis numéricas com maior correlação com o target"
    )
    ax.set_xlabel("Correlação")
    fig.tight_layout()
    fig.savefig(
        IMAGES_DIR / "eda_numeric_correlations.png",
        dpi=160,
    )
    plt.close(fig)


# ============================================================
# 6. RISCO POR UF
# ============================================================

if "sigla_uf" in df.columns:
    uf = (
        df.groupby("sigla_uf", dropna=False)[TARGET]
        .agg(["count", "mean"])
        .reset_index()
        .rename(
            columns={
                "count": "alunos",
                "mean": "taxa_risco",
            }
        )
    )

    uf["taxa_risco_pct"] = 100 * uf["taxa_risco"]

    uf.to_csv(
        REPORTS_DIR / "eda_risco_por_uf.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_uf = uf[uf["alunos"] >= 10].sort_values(
        "taxa_risco_pct"
    )

    if not plot_uf.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            plot_uf["sigla_uf"],
            plot_uf["taxa_risco_pct"],
        )
        ax.set_title(
            "Taxa de risco observada por UF na amostra"
        )
        ax.set_xlabel("% de alunos não alfabetizados")
        fig.tight_layout()
        fig.savefig(
            IMAGES_DIR / "eda_risk_by_uf.png",
            dpi=160,
        )
        plt.close(fig)


print("EDA concluída.")
print(f"Relatórios: {REPORTS_DIR}")
print(f"Imagens:    {IMAGES_DIR}")
