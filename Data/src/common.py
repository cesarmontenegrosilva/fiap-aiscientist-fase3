import csv
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd


def setup_logging(log_file: Path):
    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)


def remove_accents(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    return "".join(
        ch for ch in value
        if not unicodedata.combining(ch)
    )


def normalize_name(value: str) -> str:
    value = remove_accents(str(value)).lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    seen = {}
    normalized = []

    for col in result.columns:
        name = normalize_name(col) or "coluna"
        seen[name] = seen.get(name, 0) + 1

        if seen[name] > 1:
            name = f"{name}_{seen[name]}"

        normalized.append(name)

    result.columns = normalized
    return result


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for col in result.columns:
        if (
            pd.api.types.is_object_dtype(result[col])
            or pd.api.types.is_string_dtype(result[col])
        ):
            result[col] = result[col].astype("string").str.strip()
            result[col] = result[col].replace(
                {
                    "": pd.NA,
                    "nan": pd.NA,
                    "None": pd.NA,
                    "NULL": pd.NA,
                }
            )

    return result


def sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)

    return (
        df.sample(
            n=n,
            random_state=seed
        )
        .reset_index(drop=True)
    )


def detect_encoding_and_separator(path: Path):
    """
    Detecta encoding e separador de CSV/TXT.

    Evita o falso positivo de ASCII quando o início do arquivo não possui
    caracteres acentuados, mas o restante possui.
    """

    file_size = path.stat().st_size
    samples = []

    with open(path, "rb") as handle:
        samples.append(handle.read(200_000))

        if file_size > 400_000:
            handle.seek(file_size // 2)
            samples.append(handle.read(200_000))

            handle.seek(max(0, file_size - 200_000))
            samples.append(handle.read(200_000))

    raw = b"\n".join(samples)

    try:
        text = raw.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        try:
            text = raw.decode("cp1252")
            encoding = "cp1252"
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
            encoding = "latin-1"

    try:
        dialect = csv.Sniffer().sniff(
            text[:50_000],
            delimiters=";,|\t,"
        )
        separator = dialect.delimiter
    except csv.Error:
        separator = ";"

    return encoding, separator


def read_csv_robust(path: Path, **kwargs):
    encoding, separator = detect_encoding_and_separator(path)

    return pd.read_csv(
        path,
        sep=separator,
        encoding=encoding,
        encoding_errors="replace",
        low_memory=False,
        on_bad_lines="skip",
        **kwargs,
    )


def to_numeric_when_safe(
    series: pd.Series,
    column_name: str
) -> pd.Series:
    protected_tokens = (
        "id_",
        "codigo",
        "cod_",
        "inep",
        "cpf",
        "cep",
    )

    if any(token in column_name for token in protected_tokens):
        return series

    if not (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        return series

    prepared = (
        series.astype("string")
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    numeric = pd.to_numeric(
        prepared,
        errors="coerce"
    )

    non_null = series.notna().sum()

    if non_null and numeric.notna().sum() / non_null >= 0.90:
        return numeric

    return series


def find_column(
    columns,
    required_terms=(),
    any_terms=(),
    excluded_terms=()
):
    for col in columns:
        low = normalize_name(col)

        if required_terms and not all(
            term in low for term in required_terms
        ):
            continue

        if any_terms and not any(
            term in low for term in any_terms
        ):
            continue

        if excluded_terms and any(
            term in low for term in excluded_terms
        ):
            continue

        return col

    return None


def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )
