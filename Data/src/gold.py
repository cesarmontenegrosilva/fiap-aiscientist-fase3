import logging

import numpy as np
import pandas as pd

from src.common import setup_logging, write_csv
from src.config import (
    GOLD_DIR,
    SILVER_DIR,
    LOG_DIR,
    ALFABETIZACAO_CUTOFF,
)


setup_logging(
    LOG_DIR / "pipeline.log"
)


def read_silver(
    filename: str
) -> pd.DataFrame:
    path = (
        SILVER_DIR
        /
        filename
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo Silver não encontrado: {path}"
        )

    return pd.read_csv(
        path,
        low_memory=False,
        encoding="utf-8-sig"
    )


def build_gold_municipios():
    logging.info(
        "Gold: indicadores municipais"
    )

    df = read_silver(
        "silver_municipios.csv"
    )

    numeric_cols = [
        "pct_alfabetizados_2023",
        "pct_alfabetizados_2024",
        "meta_2024",
        "meta_2025",
        "meta_2026",
        "meta_2027",
        "meta_2028",
        "meta_2029",
        "meta_2030",
        "pct_participacao",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df[
        "variacao_2023_2024_pp"
    ] = (
        df["pct_alfabetizados_2024"]
        -
        df["pct_alfabetizados_2023"]
    )

    df[
        "gap_meta_2024_pp"
    ] = (
        df["pct_alfabetizados_2024"]
        -
        df["meta_2024"]
    )

    df[
        "atingiu_meta_2024"
    ] = np.where(
        (
            df["pct_alfabetizados_2024"].notna()
            &
            df["meta_2024"].notna()
        ),
        (
            df["pct_alfabetizados_2024"]
            >=
            df["meta_2024"]
        ).astype(int),
        pd.NA
    )

    # Atenção: como a Bronze municipal contém uma amostra de até 5.000 linhas,
    # este ranking é um ranking DENTRO DA AMOSTRA.
    df[
        "ranking_amostra_brasil_2024"
    ] = (
        df[
            "pct_alfabetizados_2024"
        ]
        .rank(
            method="min",
            ascending=False
        )
        .astype("Int64")
    )

    df[
        "ranking_amostra_uf_2024"
    ] = (
        df
        .groupby(
            "sigla_uf"
        )[
            "pct_alfabetizados_2024"
        ]
        .rank(
            method="min",
            ascending=False
        )
        .astype("Int64")
    )

    columns = [
        "ano_avaliacao",
        "codigo_uf",
        "sigla_uf",
        "codigo_municipio",
        "nome_municipio",
        "rede",
        "pct_alfabetizados_2023",
        "pct_alfabetizados_2024",
        "variacao_2023_2024_pp",
        "meta_2024",
        "gap_meta_2024_pp",
        "atingiu_meta_2024",
        "meta_2025",
        "meta_2026",
        "meta_2027",
        "meta_2028",
        "meta_2029",
        "meta_2030",
        "nivel_alfabetizacao",
        "pct_participacao",
        "ranking_amostra_brasil_2024",
        "ranking_amostra_uf_2024",
    ]

    df = df[
        [
            c
            for c in columns
            if c in df.columns
        ]
    ].copy()

    write_csv(
        df,
        GOLD_DIR / "gold_indicadores_municipios.csv"
    )

    return df


def build_gold_ufs():
    logging.info(
        "Gold: indicadores Brasil/UFs"
    )

    df = read_silver(
        "silver_ufs.csv"
    )

    numeric_cols = [
        "pct_alfabetizados_2023",
        "pct_alfabetizados_2024",
        "meta_2024",
        "meta_2025",
        "meta_2026",
        "meta_2027",
        "meta_2028",
        "meta_2029",
        "meta_2030",
        "pct_participacao",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df[
        "variacao_2023_2024_pp"
    ] = (
        df["pct_alfabetizados_2024"]
        -
        df["pct_alfabetizados_2023"]
    )

    df[
        "gap_meta_2024_pp"
    ] = (
        df["pct_alfabetizados_2024"]
        -
        df["meta_2024"]
    )

    df[
        "atingiu_meta_2024"
    ] = np.where(
        (
            df["pct_alfabetizados_2024"].notna()
            &
            df["meta_2024"].notna()
        ),
        (
            df["pct_alfabetizados_2024"]
            >=
            df["meta_2024"]
        ).astype(int),
        pd.NA
    )

    # Evita dar ranking para eventual linha "Brasil".
    mask_uf = (
        df["sigla_uf"].notna()
        &
        (
            df["sigla_uf"]
            .astype(str)
            .str.len()
            == 2
        )
    )

    df[
        "ranking_ufs_2024"
    ] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64"
    )

    df.loc[
        mask_uf,
        "ranking_ufs_2024"
    ] = (
        df.loc[
            mask_uf,
            "pct_alfabetizados_2024"
        ]
        .rank(
            method="min",
            ascending=False
        )
        .astype("Int64")
    )

    columns = [
        "ano_avaliacao",
        "codigo_uf",
        "sigla_uf",
        "nome_uf",
        "rede",
        "pct_alfabetizados_2023",
        "pct_alfabetizados_2024",
        "variacao_2023_2024_pp",
        "meta_2024",
        "gap_meta_2024_pp",
        "atingiu_meta_2024",
        "meta_2025",
        "meta_2026",
        "meta_2027",
        "meta_2028",
        "meta_2029",
        "meta_2030",
        "pct_participacao",
        "ranking_ufs_2024",
    ]

    df = df[
        [
            c
            for c in columns
            if c in df.columns
        ]
    ].copy()

    write_csv(
        df,
        GOLD_DIR / "gold_indicadores_ufs.csv"
    )

    return df


def build_gold_alunos():
    logging.info(
        "Gold: amostra analítica de alunos"
    )

    df = read_silver(
        "silver_alunos.csv"
    )

    df[
        "proficiencia_lp"
    ] = pd.to_numeric(
        df[
            "proficiencia_lp"
        ],
        errors="coerce"
    )

    df[
        "alfabetizado_oficial"
    ] = pd.to_numeric(
        df[
            "alfabetizado"
        ],
        errors="coerce"
    ).astype("Int64")

    df[
        "alfabetizado_calculado_743"
    ] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64"
    )

    mask = (
        df[
            "proficiencia_lp"
        ].notna()
    )

    df.loc[
        mask,
        "alfabetizado_calculado_743"
    ] = (
        df.loc[
            mask,
            "proficiencia_lp"
        ]
        >=
        ALFABETIZACAO_CUTOFF
    ).astype(int)

    df[
        "divergencia_indicador_743"
    ] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64"
    )

    compare_mask = (
        df[
            "alfabetizado_oficial"
        ].notna()
        &
        df[
            "alfabetizado_calculado_743"
        ].notna()
    )

    df.loc[
        compare_mask,
        "divergencia_indicador_743"
    ] = (
        df.loc[
            compare_mask,
            "alfabetizado_oficial"
        ]
        !=
        df.loc[
            compare_mask,
            "alfabetizado_calculado_743"
        ]
    ).astype(int)

    columns = [
        "ano_avaliacao",
        "codigo_uf",
        "sigla_uf",
        "codigo_municipio",
        "nome_municipio",
        "id_aluno",
        "id_escola",
        "tipo_serie",
        "tipo_dependencia",
        "presenca_lp",
        "preenchimento_lp",
        "codigo_caderno_lp",
        "peso_aluno_lp",
        "proficiencia_lp",
        "alfabetizado_oficial",
        "alfabetizado_calculado_743",
        "divergencia_indicador_743",
    ]

    df = df[
        [
            c
            for c in columns
            if c in df.columns
        ]
    ].copy()

    write_csv(
        df,
        GOLD_DIR / "gold_amostra_alunos.csv"
    )

    return df


def build_gold_evolucao(
    municipios: pd.DataFrame
):
    logging.info(
        "Gold: evolução temporal municipal"
    )

    id_cols = [
        "codigo_uf",
        "sigla_uf",
        "codigo_municipio",
        "nome_municipio",
    ]

    base = municipios[
        [
            c
            for c in (
                id_cols
                +
                [
                    "pct_alfabetizados_2023",
                    "pct_alfabetizados_2024",
                ]
            )
            if c in municipios.columns
        ]
    ].copy()

    evolution = base.melt(
        id_vars=[
            c
            for c in id_cols
            if c in base.columns
        ],
        value_vars=[
            c
            for c in [
                "pct_alfabetizados_2023",
                "pct_alfabetizados_2024",
            ]
            if c in base.columns
        ],
        var_name="indicador",
        value_name="pct_alfabetizados",
    )

    evolution[
        "ano"
    ] = (
        evolution[
            "indicador"
        ]
        .str.extract(
            r"(2023|2024)"
        )
        .astype("Int64")
    )

    evolution = (
        evolution
        .drop(
            columns=[
                "indicador"
            ]
        )
        .sort_values(
            [
                "codigo_municipio",
                "ano",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    write_csv(
        evolution,
        GOLD_DIR / "gold_evolucao_temporal_municipal.csv"
    )

    return evolution


def build_gold_resumo_uf(
    municipios: pd.DataFrame
):
    logging.info(
        "Gold: resumo analítico por UF"
    )

    resumo = (
        municipios
        .groupby(
            "sigla_uf",
            dropna=False
        )
        .agg(
            municipios_na_amostra=(
                "codigo_municipio",
                "nunique"
            ),
            media_pct_alfabetizados_2023=(
                "pct_alfabetizados_2023",
                "mean"
            ),
            media_pct_alfabetizados_2024=(
                "pct_alfabetizados_2024",
                "mean"
            ),
            media_variacao_2023_2024_pp=(
                "variacao_2023_2024_pp",
                "mean"
            ),
            media_gap_meta_2024_pp=(
                "gap_meta_2024_pp",
                "mean"
            ),
            pct_municipios_atingiram_meta_2024=(
                "atingiu_meta_2024",
                "mean"
            ),
            media_participacao=(
                "pct_participacao",
                "mean"
            ),
        )
        .reset_index()
    )

    resumo[
        "pct_municipios_atingiram_meta_2024"
    ] = (
        resumo[
            "pct_municipios_atingiram_meta_2024"
        ]
        * 100
    )

    write_csv(
        resumo,
        GOLD_DIR / "gold_resumo_municipal_por_uf.csv"
    )

    return resumo


def run_gold():
    logging.info(
        "=" * 72
    )

    logging.info(
        "CAMADA GOLD — DATASETS ANALÍTICOS"
    )

    logging.info(
        "=" * 72
    )

    municipios = (
        build_gold_municipios()
    )

    build_gold_ufs()

    alunos = (
        build_gold_alunos()
    )

    build_gold_evolucao(
        municipios
    )

    build_gold_resumo_uf(
        municipios
    )

    resumo = pd.DataFrame(
        [
            {
                "metrica":
                    "municipios_gold",
                "valor":
                    len(municipios),
            },
            {
                "metrica":
                    "alunos_gold",
                "valor":
                    len(alunos),
            },
            {
                "metrica":
                    "corte_proficiencia_alfabetizacao",
                "valor":
                    ALFABETIZACAO_CUTOFF,
            },
            {
                "metrica":
                    "observacao_ranking_municipal",
                "valor":
                    (
                        "O ranking municipal é calculado "
                        "sobre a amostra Bronze, não sobre "
                        "a totalidade dos municípios."
                    ),
            },
        ]
    )

    write_csv(
        resumo,
        GOLD_DIR / "_resumo_gold.csv"
    )

    logging.info(
        "CAMADA GOLD CONCLUÍDA"
    )


if __name__ == "__main__":
    run_gold()
