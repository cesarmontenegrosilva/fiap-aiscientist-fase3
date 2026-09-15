from __future__ import annotations

import json
import logging
import os
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# ============================================================
# CONFIGURAÇÕES
# ============================================================

INEP_CENSO_ESCOLAR_2023_URL = (
    "https://download.inep.gov.br/dados_abertos/"
    "microdados_censo_escolar_2023.zip"
)

IBGE_LOCALIDADES_URL = (
    "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
)

IBGE_POPULACAO_2022_URL = (
    "https://apisidra.ibge.gov.br/values/"
    "t/4709/n6/all/v/93/p/2022"
)

IBGE_URBANO_RURAL_2022_URL = (
    "https://apisidra.ibge.gov.br/values/"
    "t/9923/n6/all/v/93/p/2022/c1/all"
)

IBGE_PIB_2022_URL = (
    "https://apisidra.ibge.gov.br/values/"
    "t/5938/n6/all/v/37/p/2022"
)

REQUEST_TIMEOUT = (30, 300)
MAX_ATTEMPTS = 6
CHUNK_SIZE = 100_000

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "close",
}


# ============================================================
# DIRETÓRIOS
# ============================================================

def find_data_dir() -> Path:
    env = os.getenv("FIAP_DATA_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"FIAP_DATA_DIR não existe: {p}")

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

CENSO_DIR = EXTERNAL_DIR / "Censo_Escolar"
CENSO_RAW_DIR = CENSO_DIR / "raw"
CENSO_EXTRACT_DIR = CENSO_RAW_DIR / "extracted"
CENSO_PROCESSED_DIR = CENSO_DIR / "processed"

IBGE_DIR = EXTERNAL_DIR / "IBGE"
IBGE_RAW_DIR = IBGE_DIR / "raw"
IBGE_PROCESSED_DIR = IBGE_DIR / "processed"

LOG_DIR = DATA_DIR / "logs"

for folder in [
    CENSO_RAW_DIR,
    CENSO_EXTRACT_DIR,
    CENSO_PROCESSED_DIR,
    IBGE_RAW_DIR,
    IBGE_PROCESSED_DIR,
    LOG_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOG
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "external_data.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


# ============================================================
# UTILITÁRIOS
# ============================================================

def normalize_code(value: Any, width: int | None = None) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    text = re.sub(r"\.0$", "", text)
    text = re.sub(r"\D", "", text)

    if not text:
        return None

    return text.zfill(width) if width else text


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def request_json(url: str, attempts: int = MAX_ATTEMPTS) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            logger.info("HTTP %s/%s: %s", attempt, attempts, url)

            r = requests.get(
                url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()

        except (requests.exceptions.RequestException, ValueError) as exc:
            logger.warning("Falha HTTP: %s", exc)

            if attempt == attempts:
                raise

            time.sleep(attempt * 5)


def download_file(url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        logger.info("Arquivo já existe; download ignorado: %s", destination)
        return destination

    part = Path(str(destination) + ".part")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            logger.info("Download %s/%s: %s", attempt, MAX_ATTEMPTS, url)

            with requests.get(
                url,
                headers=HEADERS,
                stream=True,
                timeout=REQUEST_TIMEOUT,
            ) as r:
                r.raise_for_status()

                with part.open("wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)

            if not part.exists() or part.stat().st_size == 0:
                raise RuntimeError("Arquivo baixado está vazio.")

            part.replace(destination)
            return destination

        except Exception as exc:
            logger.warning("Falha no download: %s", exc)

            if part.exists():
                try:
                    part.unlink()
                except OSError:
                    pass

            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    "Não foi possível baixar o Censo Escolar.\n"
                    f"Baixe manualmente e coloque em:\n{destination}"
                ) from exc

            time.sleep(attempt * 10)


def detect_encoding(path: Path) -> str:
    size = path.stat().st_size
    parts = []

    with path.open("rb") as f:
        parts.append(f.read(200_000))

        if size > 400_000:
            f.seek(size // 2)
            parts.append(f.read(200_000))

            f.seek(max(0, size - 200_000))
            parts.append(f.read(200_000))

    raw = b"\n".join(parts)

    for encoding in ["utf-8", "cp1252", "latin-1"]:
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            pass

    return "latin-1"


def parse_numeric(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()

    s = s.replace(
        {
            "-": "0",
            "...": pd.NA,
            "..": pd.NA,
            "X": pd.NA,
            "x": pd.NA,
        }
    )

    out = pd.to_numeric(s, errors="coerce")

    failed = out.isna() & s.notna()

    if failed.any():
        br = (
            s.loc[failed]
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )

        out.loc[failed] = pd.to_numeric(br, errors="coerce")

    return out


# ============================================================
# MUNICÍPIOS PRESENTES NA GOLD
# ============================================================

def load_gold_municipalities() -> set[str]:
    paths = [
        GOLD_DIR / "gold_indicadores_municipios.csv",
        GOLD_DIR / "gold_amostra_alunos.csv",
    ]

    ids: set[str] = set()

    for path in paths:
        if not path.exists():
            logger.warning("Gold não encontrada: %s", path)
            continue

        df = pd.read_csv(
            path,
            dtype=str,
            low_memory=False,
            encoding="utf-8-sig",
        )

        if "codigo_municipio" not in df.columns:
            continue

        codes = (
            df["codigo_municipio"]
            .map(lambda x: normalize_code(x, 7))
            .dropna()
        )

        ids.update(codes.tolist())

    if not ids:
        raise ValueError("Nenhum codigo_municipio encontrado na Gold.")

    logger.info("Municípios na Gold: %s", len(ids))
    return ids


# ============================================================
# CENSO ESCOLAR 2023
# ============================================================

CENSO_BINARY_FEATURES = {
    "IN_INTERNET": "pct_escolas_internet",
    "IN_BANDA_LARGA": "pct_escolas_banda_larga",
    "IN_BIBLIOTECA": "pct_escolas_biblioteca",
    "IN_SALA_LEITURA": "pct_escolas_sala_leitura",
    "IN_BIBLIOTECA_SALA_LEITURA": "pct_escolas_biblioteca_ou_sala_leitura",
    "IN_LABORATORIO_INFORMATICA": "pct_escolas_lab_informatica",
    "IN_AGUA_POTAVEL": "pct_escolas_agua_potavel",
    "IN_AGUA_REDE_PUBLICA": "pct_escolas_agua_rede_publica",
    "IN_ENERGIA_REDE_PUBLICA": "pct_escolas_energia_rede_publica",
    "IN_ESGOTO_REDE_PUBLICA": "pct_escolas_esgoto_rede_publica",
    "IN_LIXO_SERVICO_COLETA": "pct_escolas_coleta_lixo",
    "IN_QUADRA_ESPORTES": "pct_escolas_quadra_esportes",
    "IN_PATIO_COBERTO": "pct_escolas_patio_coberto",
    "IN_PATIO_DESCOBERTO": "pct_escolas_patio_descoberto",
    "IN_REFEITORIO": "pct_escolas_refeitorio",
    "IN_AUDITORIO": "pct_escolas_auditorio",
}

CENSO_NUMERIC_FEATURES = {
    "QT_SALAS_EXISTENTES": "media_salas_existentes",
    "QT_SALAS_UTILIZADAS": "media_salas_utilizadas",
    "QT_EQUIP_MULTIMIDIA": "media_equip_multimidia",
    "QT_DESKTOP_ALUNO": "media_desktop_aluno",
    "QT_COMP_PORTATIL_ALUNO": "media_notebook_aluno",
    "QT_TABLET_ALUNO": "media_tablet_aluno",
}


def locate_or_extract_censo_csv(zip_path: Path) -> Path:
    existing = list(CENSO_EXTRACT_DIR.glob("*.csv"))

    if existing:
        return max(existing, key=lambda p: p.stat().st_size)

    with zipfile.ZipFile(zip_path) as archive:
        csv_members = [
            n for n in archive.namelist()
            if n.lower().endswith(".csv")
        ]

        if not csv_members:
            raise FileNotFoundError("Nenhum CSV encontrado no ZIP do Censo.")

        main_member = max(
            csv_members,
            key=lambda n: archive.getinfo(n).file_size,
        )

        output = CENSO_EXTRACT_DIR / Path(main_member).name

        with archive.open(main_member) as src, output.open("wb") as dst:
            while True:
                block = src.read(1024 * 1024)

                if not block:
                    break

                dst.write(block)

    return output


def aggregate_censo_by_municipality(
    censo_csv: Path,
    municipality_ids: set[str],
) -> pd.DataFrame:

    encoding = detect_encoding(censo_csv)

    logger.info(
        "Censo Escolar | arquivo=%s | encoding=%s",
        censo_csv.name,
        encoding,
    )

    header = pd.read_csv(
        censo_csv,
        sep=";",
        encoding=encoding,
        encoding_errors="replace",
        dtype=str,
        nrows=0,
    )

    available = set(header.columns)

    if "CO_MUNICIPIO" not in available:
        raise KeyError("CO_MUNICIPIO não existe no Censo Escolar.")

    usecols = ["CO_MUNICIPIO"]

    identity_cols = [
        "NO_MUNICIPIO",
        "SG_UF",
        "CO_UF",
        "TP_LOCALIZACAO",
        "TP_DEPENDENCIA",
    ]

    usecols += [
        c for c in identity_cols
        if c in available
    ]

    selected_binary = {
        k: v for k, v in CENSO_BINARY_FEATURES.items()
        if k in available
    }

    selected_numeric = {
        k: v for k, v in CENSO_NUMERIC_FEATURES.items()
        if k in available
    }

    usecols += list(selected_binary)
    usecols += list(selected_numeric)
    usecols = list(dict.fromkeys(usecols))

    logger.info(
        "Censo | binários encontrados: %s",
        list(selected_binary),
    )

    logger.info(
        "Censo | numéricos encontrados: %s",
        list(selected_numeric),
    )

    chunks = []
    total = 0
    kept = 0

    reader = pd.read_csv(
        censo_csv,
        sep=";",
        encoding=encoding,
        encoding_errors="replace",
        dtype=str,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        on_bad_lines="skip",
    )

    for n, chunk in enumerate(reader, start=1):
        total += len(chunk)

        chunk["codigo_municipio"] = (
            chunk["CO_MUNICIPIO"]
            .map(lambda x: normalize_code(x, 7))
        )

        chunk = chunk[
            chunk["codigo_municipio"].isin(municipality_ids)
        ].copy()

        kept += len(chunk)

        if not chunk.empty:
            chunks.append(chunk)

        logger.info(
            "Censo | chunk=%s | lidas=%s | mantidas=%s",
            n,
            f"{total:,}",
            f"{kept:,}",
        )

    if not chunks:
        raise ValueError(
            "Nenhuma escola do Censo encontrada nos municípios da Gold."
        )

    schools = pd.concat(chunks, ignore_index=True)

    for col in selected_binary:
        schools[col] = pd.to_numeric(
            schools[col],
            errors="coerce",
        )

    for col in selected_numeric:
        schools[col] = pd.to_numeric(
            schools[col],
            errors="coerce",
        )

    # Indicadores de localização
    if "TP_LOCALIZACAO" in schools.columns:
        loc = pd.to_numeric(
            schools["TP_LOCALIZACAO"],
            errors="coerce",
        )

        schools["escola_urbana"] = (loc == 1).astype("Float64")
        schools["escola_rural"] = (loc == 2).astype("Float64")

    # Dependência administrativa
    if "TP_DEPENDENCIA" in schools.columns:
        dep = pd.to_numeric(
            schools["TP_DEPENDENCIA"],
            errors="coerce",
        )

        schools["dep_federal"] = (dep == 1).astype("Float64")
        schools["dep_estadual"] = (dep == 2).astype("Float64")
        schools["dep_municipal"] = (dep == 3).astype("Float64")
        schools["dep_privada"] = (dep == 4).astype("Float64")

    grouped = schools.groupby(
        "codigo_municipio",
        dropna=False,
    )

    result = (
        grouped.size()
        .rename("numero_escolas")
        .reset_index()
    )

    # Identificadores
    identity_map = {
        "NO_MUNICIPIO": "nome_municipio_censo",
        "SG_UF": "sigla_uf_censo",
        "CO_UF": "codigo_uf_censo",
    }

    for source, output in identity_map.items():
        if source in schools.columns:
            aux = (
                grouped[source]
                .first()
                .rename(output)
                .reset_index()
            )

            result = result.merge(
                aux,
                on="codigo_municipio",
                how="left",
            )

    # Percentuais estruturais
    derived_pct = {
        "escola_urbana": "pct_escolas_urbanas",
        "escola_rural": "pct_escolas_rurais",
        "dep_federal": "pct_escolas_federais",
        "dep_estadual": "pct_escolas_estaduais",
        "dep_municipal": "pct_escolas_municipais",
        "dep_privada": "pct_escolas_privadas",
    }

    for source, output in derived_pct.items():
        if source in schools.columns:
            aux = (
                grouped[source]
                .mean()
                .mul(100)
                .rename(output)
                .reset_index()
            )

            result = result.merge(
                aux,
                on="codigo_municipio",
                how="left",
            )

    # Infraestrutura binária -> percentual municipal de escolas
    for source, output in selected_binary.items():
        aux = (
            grouped[source]
            .mean()
            .mul(100)
            .rename(output)
            .reset_index()
        )

        result = result.merge(
            aux,
            on="codigo_municipio",
            how="left",
        )

    # Variáveis quantitativas -> média municipal
    for source, output in selected_numeric.items():
        aux = (
            grouped[source]
            .mean()
            .rename(output)
            .reset_index()
        )

        result = result.merge(
            aux,
            on="codigo_municipio",
            how="left",
        )

    return (
        result
        .sort_values("codigo_municipio")
        .reset_index(drop=True)
    )


def collect_censo_escolar(
    municipality_ids: set[str],
) -> Path:

    zip_path = (
        CENSO_RAW_DIR
        / "microdados_censo_escolar_2023.zip"
    )

    if not zip_path.exists():
        download_file(
            INEP_CENSO_ESCOLAR_2023_URL,
            zip_path,
        )
    else:
        logger.info(
            "ZIP do Censo Escolar já existe: %s",
            zip_path,
        )

    censo_csv = locate_or_extract_censo_csv(
        zip_path
    )

    municipal = aggregate_censo_by_municipality(
        censo_csv,
        municipality_ids,
    )

    output = (
        CENSO_PROCESSED_DIR
        / "censo_escolar_2023_municipios.csv"
    )

    write_csv(municipal, output)

    logger.info(
        "Censo municipal: %s municípios e %s colunas.",
        len(municipal),
        len(municipal.columns),
    )

    return output


# ============================================================
# IBGE - LOCALIDADES
# ============================================================

def flatten_localidades(
    payload: list[dict],
) -> pd.DataFrame:

    rows = []

    for item in payload:
        immediate = item.get("regiao-imediata") or {}
        intermediate = immediate.get("regiao-intermediaria") or {}
        uf = intermediate.get("UF") or {}
        region = uf.get("regiao") or {}

        micro = item.get("microrregiao") or {}
        meso = micro.get("mesorregiao") or {}

        if not uf:
            uf = meso.get("UF") or {}
            region = uf.get("regiao") or {}

        rows.append(
            {
                "codigo_municipio": normalize_code(item.get("id"), 7),
                "nome_municipio_ibge": item.get("nome"),
                "codigo_uf_ibge": normalize_code(uf.get("id"), 2),
                "sigla_uf_ibge": uf.get("sigla"),
                "nome_uf_ibge": uf.get("nome"),
                "codigo_regiao": region.get("id"),
                "sigla_regiao": region.get("sigla"),
                "nome_regiao": region.get("nome"),
                "codigo_regiao_imediata": immediate.get("id"),
                "nome_regiao_imediata": immediate.get("nome"),
                "codigo_regiao_intermediaria": intermediate.get("id"),
                "nome_regiao_intermediaria": intermediate.get("nome"),
                "codigo_microrregiao": micro.get("id"),
                "nome_microrregiao": micro.get("nome"),
                "codigo_mesorregiao": meso.get("id"),
                "nome_mesorregiao": meso.get("nome"),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# IBGE - SIDRA
# ============================================================

def sidra_to_df(payload: list[dict]) -> pd.DataFrame:
    if not payload or len(payload) < 2:
        raise ValueError("SIDRA retornou resposta vazia.")

    metadata = payload[0]
    df = pd.DataFrame(payload[1:])

    rename = {
        internal: label
        for internal, label in metadata.items()
        if internal in df.columns
    }

    return df.rename(columns=rename)


def find_column(
    columns,
    include: list[str],
    exclude: list[str] | None = None,
) -> str | None:

    exclude = exclude or []

    for col in columns:
        low = str(col).lower()

        if not all(term.lower() in low for term in include):
            continue

        if any(term.lower() in low for term in exclude):
            continue

        return col

    return None


def parse_sidra_simple(
    payload: list[dict],
    output_name: str,
) -> pd.DataFrame:

    df = sidra_to_df(payload)

    code_col = find_column(
        df.columns,
        ["município", "código"],
    )

    value_col = find_column(
        df.columns,
        ["valor"],
    )

    if code_col is None or value_col is None:
        raise KeyError(
            "Não identifiquei Município (Código) e Valor no SIDRA."
        )

    result = pd.DataFrame(
        {
            "codigo_municipio": (
                df[code_col]
                .map(lambda x: normalize_code(x, 7))
            ),
            output_name: parse_numeric(
                df[value_col]
            ),
        }
    )

    return (
        result
        .dropna(subset=["codigo_municipio"])
        .drop_duplicates(subset=["codigo_municipio"])
        .reset_index(drop=True)
    )


def parse_sidra_urban_rural(
    payload: list[dict],
) -> pd.DataFrame:

    df = sidra_to_df(payload)

    code_col = find_column(
        df.columns,
        ["município", "código"],
    )

    value_col = find_column(
        df.columns,
        ["valor"],
    )

    category_col = find_column(
        df.columns,
        ["situação", "domicílio"],
        exclude=["código"],
    )

    if (
        code_col is None
        or value_col is None
        or category_col is None
    ):
        raise KeyError(
            "Não identifiquei os campos da tabela urbana/rural do SIDRA."
        )

    tmp = pd.DataFrame(
        {
            "codigo_municipio": (
                df[code_col]
                .map(lambda x: normalize_code(x, 7))
            ),
            "categoria": (
                df[category_col]
                .astype("string")
                .str.strip()
                .str.lower()
            ),
            "valor": parse_numeric(
                df[value_col]
            ),
        }
    )

    pivot = (
        tmp.pivot_table(
            index="codigo_municipio",
            columns="categoria",
            values="valor",
            aggfunc="first",
        )
        .reset_index()
    )

    rename = {}

    for col in pivot.columns:
        low = str(col).lower()

        if low == "urbana":
            rename[col] = "pop_urbana_2022"
        elif low == "rural":
            rename[col] = "pop_rural_2022"
        elif low == "total":
            rename[col] = "pop_total_urbano_rural_2022"

    pivot = pivot.rename(columns=rename)

    if (
        "pop_urbana_2022" in pivot.columns
        and
        "pop_rural_2022" in pivot.columns
    ):
        total = (
            pivot["pop_urbana_2022"]
            + pivot["pop_rural_2022"]
        )

        pivot["pct_urbana_2022"] = (
            100 * pivot["pop_urbana_2022"] / total
        )

        pivot["pct_rural_2022"] = (
            100 * pivot["pop_rural_2022"] / total
        )

    return pivot


def safe_json(
    label: str,
    url: str,
    raw_path: Path,
) -> list[dict] | None:

    try:
        logger.info("IBGE | %s", label)

        payload = request_json(url)

        write_json(
            payload,
            raw_path,
        )

        return payload

    except Exception as exc:
        logger.error(
            "IBGE | falha em %s: %s",
            label,
            exc,
        )

        return None


def collect_ibge(
    municipality_ids: set[str],
) -> Path:

    # Localidades
    loc_payload = safe_json(
        "Localidades",
        IBGE_LOCALIDADES_URL,
        IBGE_RAW_DIR / "localidades_municipios.json",
    )

    if loc_payload is not None:
        base = flatten_localidades(loc_payload)
    else:
        base = pd.DataFrame(
            {"codigo_municipio": sorted(municipality_ids)}
        )

    # População
    pop_payload = safe_json(
        "População 2022",
        IBGE_POPULACAO_2022_URL,
        IBGE_RAW_DIR / "sidra_populacao_2022.json",
    )

    pop = (
        parse_sidra_simple(
            pop_payload,
            "populacao_2022",
        )
        if pop_payload is not None
        else pd.DataFrame(
            columns=[
                "codigo_municipio",
                "populacao_2022",
            ]
        )
    )

    # Urbano / rural
    urban_payload = safe_json(
        "População urbana/rural 2022",
        IBGE_URBANO_RURAL_2022_URL,
        IBGE_RAW_DIR / "sidra_urbano_rural_2022.json",
    )

    urban = (
        parse_sidra_urban_rural(
            urban_payload
        )
        if urban_payload is not None
        else pd.DataFrame(
            columns=["codigo_municipio"]
        )
    )

    # PIB
    pib_payload = safe_json(
        "PIB municipal 2022",
        IBGE_PIB_2022_URL,
        IBGE_RAW_DIR / "sidra_pib_2022.json",
    )

    pib = (
        parse_sidra_simple(
            pib_payload,
            "pib_2022_mil_reais",
        )
        if pib_payload is not None
        else pd.DataFrame(
            columns=[
                "codigo_municipio",
                "pib_2022_mil_reais",
            ]
        )
    )

    ibge = (
        base
        .merge(
            pop,
            on="codigo_municipio",
            how="left",
        )
        .merge(
            urban,
            on="codigo_municipio",
            how="left",
        )
        .merge(
            pib,
            on="codigo_municipio",
            how="left",
        )
    )

    if (
        "populacao_2022" in ibge.columns
        and
        "pib_2022_mil_reais" in ibge.columns
    ):
        pop_num = pd.to_numeric(
            ibge["populacao_2022"],
            errors="coerce",
        )

        pib_num = pd.to_numeric(
            ibge["pib_2022_mil_reais"],
            errors="coerce",
        )

        ibge["pib_per_capita_2022"] = (
            pib_num * 1000 / pop_num
        )

    ibge = (
        ibge[
            ibge["codigo_municipio"].isin(
                municipality_ids
            )
        ]
        .drop_duplicates(
            subset=["codigo_municipio"]
        )
        .sort_values("codigo_municipio")
        .reset_index(drop=True)
    )

    output = (
        IBGE_PROCESSED_DIR
        / "ibge_municipios_2022.csv"
    )

    write_csv(
        ibge,
        output,
    )

    logger.info(
        "IBGE: %s municípios e %s colunas.",
        len(ibge),
        len(ibge.columns),
    )

    return output


# ============================================================
# MANIFESTO
# ============================================================

def build_manifest(
    censo_output: Path | None,
    ibge_output: Path | None,
) -> Path:

    now = datetime.now().isoformat(
        timespec="seconds"
    )

    df = pd.DataFrame(
        [
            {
                "fonte": "INEP - Censo Escolar 2023",
                "ano": 2023,
                "arquivo": str(censo_output) if censo_output else "",
                "chave_integracao": "codigo_municipio",
                "uso": "Infraestrutura e contexto educacional municipal",
                "processado_em": now,
            },
            {
                "fonte": "IBGE - Censo 2022 / SIDRA / Localidades",
                "ano": 2022,
                "arquivo": str(ibge_output) if ibge_output else "",
                "chave_integracao": "codigo_municipio",
                "uso": (
                    "População, urbanização, PIB, PIB per capita "
                    "e atributos territoriais"
                ),
                "processado_em": now,
            },
        ]
    )

    output = (
        EXTERNAL_DIR
        / "_manifesto_fontes_externas.csv"
    )

    write_csv(
        df,
        output,
    )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info("=" * 78)
    logger.info(
        "DADOS EXTERNOS - TECH CHALLENGE FIAP FASE 3"
    )
    logger.info(
        "Data directory: %s",
        DATA_DIR,
    )
    logger.info("=" * 78)

    municipality_ids = (
        load_gold_municipalities()
    )

    censo_output = None
    ibge_output = None

    # --------------------------------------------------------
    # 1/2 CENSO ESCOLAR
    # --------------------------------------------------------

    logger.info(
        "[1/2] CENSO ESCOLAR 2023 - AGREGAÇÃO MUNICIPAL"
    )

    try:
        censo_output = (
            collect_censo_escolar(
                municipality_ids
            )
        )

    except Exception as exc:
        logger.exception(
            "Falha no Censo Escolar: %s",
            exc,
        )

        logger.warning(
            "A execução continuará para o IBGE."
        )

    # --------------------------------------------------------
    # 2/2 IBGE
    # --------------------------------------------------------

    logger.info(
        "[2/2] IBGE 2022"
    )

    try:
        ibge_output = (
            collect_ibge(
                municipality_ids
            )
        )

    except Exception as exc:
        logger.exception(
            "Falha no IBGE: %s",
            exc,
        )

    manifest = build_manifest(
        censo_output,
        ibge_output,
    )

    logger.info("=" * 78)
    logger.info("PROCESSAMENTO FINALIZADO")
    logger.info(
        "Censo Escolar: %s",
        censo_output if censo_output else "FALHOU",
    )
    logger.info(
        "IBGE: %s",
        ibge_output if ibge_output else "FALHOU",
    )
    logger.info(
        "Manifesto: %s",
        manifest,
    )
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
