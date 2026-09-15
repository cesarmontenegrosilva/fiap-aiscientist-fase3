import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import setup_logging, write_csv
from src.config import BRONZE_DIR, SILVER_DIR, LOG_DIR


setup_logging(LOG_DIR / "pipeline.log")


def read_bronze(filename: str) -> pd.DataFrame:
    path = BRONZE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo Bronze não encontrado: {path}"
        )

    return pd.read_csv(
        path,
        dtype=str,
        low_memory=False,
        encoding="utf-8-sig"
    )


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
                "NULL": pd.NA,
            }
        )
    )


def normalize_code(series: pd.Series, digits: int | None = None) -> pd.Series:
    result = (
        clean_text(series)
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
    )

    if digits is not None:
        result = result.str.zfill(digits)

    return result


def to_number(series: pd.Series) -> pd.Series:
    """
    Converte números aceitando tanto ponto quanto vírgula decimal.
    """
    s = clean_text(series)

    # Se houver vírgula, assume padrão brasileiro.
    comma_mask = s.str.contains(",", na=False)

    s_br = (
        s[comma_mask]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    s_dot = s[~comma_mask]

    out = pd.Series(
        pd.NA,
        index=s.index,
        dtype="Float64"
    )

    out.loc[comma_mask] = pd.to_numeric(
        s_br,
        errors="coerce"
    ).astype("Float64")

    out.loc[~comma_mask] = pd.to_numeric(
        s_dot,
        errors="coerce"
    ).astype("Float64")

    return out


def to_int(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(
        clean_text(series),
        errors="coerce"
    )
    return numeric.astype("Int64")


def validate_range(
    df: pd.DataFrame,
    column: str,
    min_value: float,
    max_value: float
) -> int:
    if column not in df.columns:
        return 0

    s = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    invalid = (
        s.notna()
        & (
            (s < min_value)
            | (s > max_value)
        )
    )

    return int(
        invalid.sum()
    )


def validate_binary(
    df: pd.DataFrame,
    column: str
) -> int:
    if column not in df.columns:
        return 0

    s = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    invalid = (
        s.notna()
        & ~s.isin([0, 1])
    )

    return int(
        invalid.sum()
    )


def build_quality_rows(
    df: pd.DataFrame,
    table_name: str,
    duplicate_count: int,
    extra_tests: list[dict]
) -> list[dict]:

    rows = []

    rows.append(
        {
            "tabela": table_name,
            "teste": "duplicatas_removidas",
            "coluna": "",
            "resultado": duplicate_count,
            "status": (
                "OK"
                if duplicate_count == 0
                else "AJUSTADO"
            ),
        }
    )

    for column in df.columns:
        rows.append(
            {
                "tabela": table_name,
                "teste": "valores_nulos",
                "coluna": column,
                "resultado": int(
                    df[column].isna().sum()
                ),
                "status": "INFO",
            }
        )

    rows.extend(extra_tests)

    return rows


def transform_alunos():
    logging.info(
        "Silver: transformando alunos"
    )

    df = read_bronze(
        "bronze_alunos_5000.csv"
    )

    rename_map = {
        "NU_ANO_AVALIACAO": "ano_avaliacao",
        "CO_UF": "codigo_uf",
        "SG_UF": "sigla_uf",
        "ID_ALUNO": "id_aluno",
        "TP_SERIE": "tipo_serie",
        "ID_ESCOLA": "id_escola",
        "TP_DEPENDENCIA": "tipo_dependencia",
        "CO_MUNICIPIO": "codigo_municipio",
        "NO_MUNICIPIO": "nome_municipio",
        "IN_PRESENCA_LP": "presenca_lp",
        "IN_PREENCHIMENTO_LP": "preenchimento_lp",
        "CO_CADERNO_LP": "codigo_caderno_lp",
        "VL_PESO_ALUNO_LP": "peso_aluno_lp",
        "VL_PROFICIENCIA_LP": "proficiencia_lp",
        "IN_ALFABETIZADO": "alfabetizado",
    }

    df = df.rename(
        columns=rename_map
    )

    df = df[
        [
            col
            for col in rename_map.values()
            if col in df.columns
        ]
    ].copy()

    for col in [
        "sigla_uf",
        "id_aluno",
        "id_escola",
        "tipo_serie",
        "tipo_dependencia",
        "nome_municipio",
        "codigo_caderno_lp",
    ]:
        if col in df.columns:
            df[col] = clean_text(
                df[col]
            )

    if "sigla_uf" in df.columns:
        df["sigla_uf"] = (
            df["sigla_uf"]
            .str.upper()
        )

    if "codigo_uf" in df.columns:
        df["codigo_uf"] = normalize_code(
            df["codigo_uf"],
            digits=2
        )

    if "codigo_municipio" in df.columns:
        df["codigo_municipio"] = normalize_code(
            df["codigo_municipio"],
            digits=7
        )

    if "ano_avaliacao" in df.columns:
        df["ano_avaliacao"] = to_int(
            df["ano_avaliacao"]
        )

    for col in [
        "presenca_lp",
        "preenchimento_lp",
        "alfabetizado",
    ]:
        if col in df.columns:
            df[col] = to_int(
                df[col]
            )

    for col in [
        "peso_aluno_lp",
        "proficiencia_lp",
    ]:
        if col in df.columns:
            df[col] = to_number(
                df[col]
            )

    before = len(df)

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    duplicate_count = (
        before - len(df)
    )

    tests = []

    for col in [
        "presenca_lp",
        "preenchimento_lp",
        "alfabetizado",
    ]:
        invalid = validate_binary(
            df,
            col
        )

        tests.append(
            {
                "tabela": "alunos",
                "teste": "validacao_binaria_0_1",
                "coluna": col,
                "resultado": invalid,
                "status": (
                    "OK"
                    if invalid == 0
                    else "ERRO"
                ),
            }
        )

    quality = build_quality_rows(
        df,
        "alunos",
        duplicate_count,
        tests
    )

    write_csv(
        df,
        SILVER_DIR / "silver_alunos.csv"
    )

    return df, quality


def transform_ufs():
    logging.info(
        "Silver: transformando Brasil/UFs"
    )

    df = read_bronze(
        "bronze_brasil_ufs.csv"
    )

    # Remove colunas vazias criadas pelo Excel
    df = df.drop(
        columns=[
            c
            for c in df.columns
            if c.startswith("Unnamed:")
        ],
        errors="ignore"
    )

    rename_map = {
        "_aba_origem": "aba_origem",
        "ANO DA AVALIAÇÃO": "ano_avaliacao",
        "CÓDIGO UF": "codigo_uf",
        "SIGLA UF": "sigla_uf",
        "NOME UF": "nome_uf",
        "REDE": "rede",
        "PERCENTUAL DE ALUNOS ALFABETIZADOS Sistemas estaduais de avaliação 2023": "pct_alfabetizados_2023",
        "PERCENTUAL DE ALUNOS ALFABETIZADOS Sistemas estaduais de avaliação 2024": "pct_alfabetizados_2024",
        "META 2024 (2)": "meta_2024",
        "META 2025": "meta_2025",
        "META 2026": "meta_2026",
        "META 2027": "meta_2027",
        "META 2028": "meta_2028",
        "META 2029": "meta_2029",
        "META 2030": "meta_2030",
        "PERCENTUAL DE PARTICIPAÇÃO": "pct_participacao",
    }

    df = df.rename(
        columns=rename_map
    )

    for col in [
        "aba_origem",
        "sigla_uf",
        "nome_uf",
        "rede",
    ]:
        if col in df.columns:
            df[col] = clean_text(
                df[col]
            )

    if "sigla_uf" in df.columns:
        df["sigla_uf"] = (
            df["sigla_uf"]
            .str.upper()
        )

    if "codigo_uf" in df.columns:
        df["codigo_uf"] = normalize_code(
            df["codigo_uf"],
            digits=2
        )

    if "ano_avaliacao" in df.columns:
        df["ano_avaliacao"] = to_int(
            df["ano_avaliacao"]
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
            df[col] = to_number(
                df[col]
            )

    before = len(df)

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    duplicate_count = (
        before - len(df)
    )

    tests = []

    for col in numeric_cols:
        if col in df.columns:
            invalid = validate_range(
                df,
                col,
                0,
                100
            )

            tests.append(
                {
                    "tabela": "ufs",
                    "teste": "faixa_0_100",
                    "coluna": col,
                    "resultado": invalid,
                    "status": (
                        "OK"
                        if invalid == 0
                        else "ERRO"
                    ),
                }
            )

    quality = build_quality_rows(
        df,
        "ufs",
        duplicate_count,
        tests
    )

    write_csv(
        df,
        SILVER_DIR / "silver_ufs.csv"
    )

    return df, quality


def transform_municipios():
    logging.info(
        "Silver: transformando municípios"
    )

    df = read_bronze(
        "bronze_municipios.csv"
    )

    df = df.drop(
        columns=[
            c
            for c in df.columns
            if c.startswith("Unnamed:")
        ],
        errors="ignore"
    )

    rename_map = {
        "_aba_origem": "aba_origem",
        "ANO DA AVALIAÇÃO": "ano_avaliacao",
        "CÓDIGO UF": "codigo_uf",
        "SIGLA UF": "sigla_uf",
        "CÓDIGO MUNICÍPIO": "codigo_municipio",
        "NOME DO MUNICÍPIO": "nome_municipio",
        "REDE": "rede",
        "PERCENTUAL DE ALUNOS ALFABETIZADOS  - 2023 (1)": "pct_alfabetizados_2023",
        "PERCENTUAL DE ALUNOS ALFABETIZADOS  - 2024 (1)": "pct_alfabetizados_2024",
        "META 2024 (2)": "meta_2024",
        "META 2025": "meta_2025",
        "META 2026": "meta_2026",
        "META 2027": "meta_2027",
        "META 2028": "meta_2028",
        "META 2029": "meta_2029",
        "META 2030": "meta_2030",
        "NIVEL ALFABETIZAÇÃO": "nivel_alfabetizacao",
        "PERCENTUAL DE PARTICIPAÇÃO": "pct_participacao",
    }

    df = df.rename(
        columns=rename_map
    )

    for col in [
        "aba_origem",
        "sigla_uf",
        "nome_municipio",
        "rede",
        "nivel_alfabetizacao",
    ]:
        if col in df.columns:
            df[col] = clean_text(
                df[col]
            )

    if "sigla_uf" in df.columns:
        df["sigla_uf"] = (
            df["sigla_uf"]
            .str.upper()
        )

    if "codigo_uf" in df.columns:
        df["codigo_uf"] = normalize_code(
            df["codigo_uf"],
            digits=2
        )

    if "codigo_municipio" in df.columns:
        df["codigo_municipio"] = normalize_code(
            df["codigo_municipio"],
            digits=7
        )

    if "ano_avaliacao" in df.columns:
        df["ano_avaliacao"] = to_int(
            df["ano_avaliacao"]
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
            df[col] = to_number(
                df[col]
            )

    before = len(df)

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    duplicate_count = (
        before - len(df)
    )

    tests = []

    for col in numeric_cols:
        if col in df.columns:
            invalid = validate_range(
                df,
                col,
                0,
                100
            )

            tests.append(
                {
                    "tabela": "municipios",
                    "teste": "faixa_0_100",
                    "coluna": col,
                    "resultado": invalid,
                    "status": (
                        "OK"
                        if invalid == 0
                        else "ERRO"
                    ),
                }
            )

    quality = build_quality_rows(
        df,
        "municipios",
        duplicate_count,
        tests
    )

    write_csv(
        df,
        SILVER_DIR / "silver_municipios.csv"
    )

    return df, quality


def run_silver():
    logging.info(
        "=" * 72
    )

    logging.info(
        "CAMADA SILVER — "
        "LIMPEZA, TIPAGEM E PADRONIZAÇÃO"
    )

    logging.info(
        "=" * 72
    )

    all_quality = []

    _, quality = transform_alunos()
    all_quality.extend(
        quality
    )

    _, quality = transform_ufs()
    all_quality.extend(
        quality
    )

    _, quality = transform_municipios()
    all_quality.extend(
        quality
    )

    quality_df = pd.DataFrame(
        all_quality
    )

    write_csv(
        quality_df,
        SILVER_DIR / "_relatorio_qualidade.csv"
    )

    logging.info(
        "CAMADA SILVER CONCLUÍDA"
    )


if __name__ == "__main__":
    run_silver()
