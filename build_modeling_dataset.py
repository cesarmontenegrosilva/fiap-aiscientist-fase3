from __future__ import annotations

from pathlib import Path
import os
import logging
import pandas as pd
import numpy as np


RANDOM_SEED = 42


def find_data_dir() -> Path:
    env = os.getenv("FIAP_DATA_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p

    cwd = Path.cwd().resolve()
    script_dir = Path(__file__).resolve().parent

    for base in [cwd, *cwd.parents, script_dir, *script_dir.parents]:
        if base.name.lower() == "data":
            return base
        candidate = base / "Data"
        if candidate.exists() and candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Não encontrei a pasta Data. Execute a partir da raiz do projeto."
    )


DATA_DIR = find_data_dir()
GOLD_DIR = DATA_DIR / "Gold"
EXTERNAL_DIR = DATA_DIR / "External"
MODELING_DIR = DATA_DIR / "Modeling"
REPORTS_DIR = DATA_DIR / "reports"
LOG_DIR = DATA_DIR / "logs"

for p in [MODELING_DIR, REPORTS_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "modeling_dataset.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def normalize_code(series: pd.Series, width: int) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(width)
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {path}")
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def coerce_numeric_like(df: pd.DataFrame, protected: set[str]) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if col in protected:
            continue

        if out[col].dtype != "object":
            continue

        s = out[col].astype("string").str.strip()
        candidate = pd.to_numeric(
            s.str.replace(",", ".", regex=False),
            errors="coerce",
        )

        non_null = s.notna().sum()
        if non_null == 0:
            continue

        ratio = candidate.notna().sum() / non_null

        if ratio >= 0.95:
            out[col] = candidate

    return out


def main():
    logger.info("=" * 78)
    logger.info("CONSTRUÇÃO DO DATASET DE MODELAGEM")
    logger.info("Data directory: %s", DATA_DIR)
    logger.info("=" * 78)

    alunos_path = GOLD_DIR / "gold_amostra_alunos.csv"
    municipios_path = GOLD_DIR / "gold_indicadores_municipios.csv"
    censo_path = (
        EXTERNAL_DIR / "Censo_Escolar" / "processed"
        / "censo_escolar_2023_municipios.csv"
    )
    ibge_path = (
        EXTERNAL_DIR / "IBGE" / "processed"
        / "ibge_municipios_2022.csv"
    )

    alunos = read_csv(alunos_path)
    municipios = read_csv(municipios_path)
    censo = read_csv(censo_path)
    ibge = read_csv(ibge_path)

    logger.info("Alunos: %s x %s", *alunos.shape)
    logger.info("Gold municipal: %s x %s", *municipios.shape)
    logger.info("Censo municipal: %s x %s", *censo.shape)
    logger.info("IBGE: %s x %s", *ibge.shape)

    for df_name, df in [
        ("alunos", alunos),
        ("municipios", municipios),
        ("censo", censo),
        ("ibge", ibge),
    ]:
        if "codigo_municipio" not in df.columns:
            raise KeyError(
                f"{df_name} não possui a coluna codigo_municipio."
            )
        df["codigo_municipio"] = normalize_code(
            df["codigo_municipio"], 7
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------
    if "alfabetizado_oficial" not in alunos.columns:
        raise KeyError(
            "gold_amostra_alunos.csv não possui alfabetizado_oficial."
        )

    alunos["alfabetizado_oficial"] = pd.to_numeric(
        alunos["alfabetizado_oficial"],
        errors="coerce",
    )

    alunos = alunos[
        alunos["alfabetizado_oficial"].isin([0, 1])
    ].copy()

    alunos["risco_nao_alfabetizado"] = (
        1 - alunos["alfabetizado_oficial"].astype(int)
    )

    # --------------------------------------------------------
    # Gold municipal: só variáveis conhecidas antes do target
    # --------------------------------------------------------
    municipal_safe_candidates = [
        "codigo_municipio",
        "pct_alfabetizados_2023",
        "meta_2024",
    ]

    municipal_safe = [
        c for c in municipal_safe_candidates
        if c in municipios.columns
    ]

    municipios_safe = municipios[municipal_safe].copy()

    # Uma linha por município
    municipios_safe = municipios_safe.drop_duplicates(
        subset=["codigo_municipio"]
    )

    # --------------------------------------------------------
    # Remove colunas duplicadas de identificação das externas
    # --------------------------------------------------------
    censo_drop = [
        c for c in [
            "nome_municipio_censo",
            "sigla_uf_censo",
            "codigo_uf_censo",
        ]
        if c in censo.columns
    ]

    censo_features = censo.drop(columns=censo_drop).drop_duplicates(
        subset=["codigo_municipio"]
    )

    ibge_features = ibge.drop_duplicates(
        subset=["codigo_municipio"]
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------
    dataset = alunos.merge(
        municipios_safe,
        on="codigo_municipio",
        how="left",
        validate="m:1",
    )

    dataset = dataset.merge(
        censo_features,
        on="codigo_municipio",
        how="left",
        validate="m:1",
        suffixes=("", "_censo"),
    )

    dataset = dataset.merge(
        ibge_features,
        on="codigo_municipio",
        how="left",
        validate="m:1",
        suffixes=("", "_ibge"),
    )

    # --------------------------------------------------------
    # Padronização de tipos
    # --------------------------------------------------------
    protected = {
        "id_aluno",
        "id_escola",
        "codigo_municipio",
        "codigo_uf",
        "sigla_uf",
        "nome_municipio",
    }

    dataset = coerce_numeric_like(
        dataset,
        protected=protected,
    )

    # --------------------------------------------------------
    # Cobertura dos merges
    # --------------------------------------------------------
    censo_signal = next(
        (
            c for c in censo_features.columns
            if c != "codigo_municipio"
        ),
        None,
    )

    ibge_signal = next(
        (
            c for c in ibge_features.columns
            if c != "codigo_municipio"
        ),
        None,
    )

    coverage_rows = []

    if censo_signal:
        coverage_rows.append(
            {
                "fonte": "Censo Escolar 2023",
                "linhas_dataset": len(dataset),
                "linhas_com_dado": int(dataset[censo_signal].notna().sum()),
                "cobertura_pct": round(
                    100 * dataset[censo_signal].notna().mean(), 2
                ),
            }
        )

    if ibge_signal:
        coverage_rows.append(
            {
                "fonte": "IBGE 2022",
                "linhas_dataset": len(dataset),
                "linhas_com_dado": int(dataset[ibge_signal].notna().sum()),
                "cobertura_pct": round(
                    100 * dataset[ibge_signal].notna().mean(), 2
                ),
            }
        )

    coverage = pd.DataFrame(coverage_rows)

    # --------------------------------------------------------
    # Lista formal de leakage
    # --------------------------------------------------------
    leakage_columns = [
        "proficiencia_lp",
        "alfabetizado_calculado_743",
        "divergencia_indicador_743",
        "pct_alfabetizados_2024",
        "variacao_2023_2024_pp",
        "gap_meta_2024_pp",
        "atingiu_meta_2024",
        "ranking_amostra_brasil_2024",
        "ranking_amostra_uf_2024",
        "ranking_ufs_2024",
        "nivel_alfabetizacao",
    ]

    leakage_present = [
        c for c in leakage_columns if c in dataset.columns
    ]

    # Não removemos alfabetizado_oficial porque ele é mantido apenas
    # para auditoria. O script de treinamento nunca o usa como feature.
    if leakage_present:
        dataset = dataset.drop(columns=leakage_present)

    output = MODELING_DIR / "dataset_modelagem.csv"
    dataset.to_csv(
        output,
        index=False,
        encoding="utf-8-sig",
    )

    coverage_path = REPORTS_DIR / "cobertura_integracao_modelagem.csv"
    coverage.to_csv(
        coverage_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame(
        [
            {
                "linhas": len(dataset),
                "colunas": len(dataset.columns),
                "target_1_risco": int(
                    (dataset["risco_nao_alfabetizado"] == 1).sum()
                ),
                "target_0_sem_risco": int(
                    (dataset["risco_nao_alfabetizado"] == 0).sum()
                ),
                "taxa_risco_pct": round(
                    100 * dataset["risco_nao_alfabetizado"].mean(), 2
                ),
                "leakage_removido": ";".join(leakage_present),
            }
        ]
    )

    summary_path = REPORTS_DIR / "resumo_dataset_modelagem.csv"
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info("Dataset salvo: %s", output)
    logger.info("Shape final: %s x %s", *dataset.shape)
    logger.info(
        "Taxa de risco: %.2f%%",
        100 * dataset["risco_nao_alfabetizado"].mean(),
    )

    if not coverage.empty:
        logger.info("\n%s", coverage.to_string(index=False))


if __name__ == "__main__":
    main()
