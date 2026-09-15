import logging
import time
import zipfile

from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.common import (
    detect_encoding_and_separator,
    sample_df,
    setup_logging,
    write_csv,
)

from src.config import (
    BASE_DIR,
    BRONZE_DIR,
    CHUNK_SIZE,
    FALLBACK_URLS,
    INEP_RESULTADOS_2024,
    LOG_DIR,
    ORIGINAL_DIR,
    RANDOM_SEED,
    REQUEST_TIMEOUT,
    SAMPLE_SIZE,
)


setup_logging(
    LOG_DIR / "pipeline.log"
)


HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/151.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,"
        "application/xhtml+xml,"
        "application/xml;q=0.9,"
        "*/*;q=0.8"
    ),
    "Accept-Language": (
        "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7"
    ),
    "Connection": "close",
}


def discover_official_links():
    logging.info(
        "Lendo página oficial do INEP: %s",
        INEP_RESULTADOS_2024
    )

    try:
        response = requests.get(
            INEP_RESULTADOS_2024,
            timeout=REQUEST_TIMEOUT,
            headers=HTTP_HEADERS,
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        found = {}

        for anchor in soup.find_all("a", href=True):
            text = " ".join(
                anchor.stripped_strings
            ).lower()

            href = urljoin(
                INEP_RESULTADOS_2024,
                anchor["href"]
            )

            if (
                "resultado" in text
                and "meta" in text
                and (
                    "uf" in text
                    or "brasil" in text
                )
                and "munic" not in text
            ):
                found["brasil_ufs"] = href

            if (
                "resultado" in text
                and "meta" in text
                and "munic" in text
            ):
                found["municipios"] = href

            if (
                "microdados" in text
                and "alfabet" in text
            ):
                found["microdados"] = href

        for key, fallback_url in FALLBACK_URLS.items():
            found.setdefault(
                key,
                fallback_url
            )

        logging.info(
            "Links identificados."
        )

        for key, url in found.items():
            logging.info(
                "%s -> %s",
                key,
                url
            )

        return found

    except Exception as exc:
        logging.warning(
            "Não foi possível descobrir automaticamente "
            "os links na página do INEP."
        )
        logging.warning(
            "Motivo: %s",
            exc
        )
        logging.info(
            "Usando URLs de fallback."
        )

        return FALLBACK_URLS.copy()


def download(
    url: str,
    destination: Path,
    max_attempts: int = 6
):
    if (
        destination.exists()
        and destination.stat().st_size > 0
    ):
        logging.info(
            "Arquivo já existe. Download ignorado: %s",
            destination.name
        )
        return destination

    temporary_file = Path(
        str(destination) + ".part"
    )

    if temporary_file.exists():
        temporary_file.unlink()

    for attempt in range(
        1,
        max_attempts + 1
    ):
        try:
            logging.info(
                "Download tentativa %s/%s",
                attempt,
                max_attempts
            )
            logging.info(
                "URL: %s",
                url
            )

            response = requests.get(
                url,
                stream=True,
                headers=HTTP_HEADERS,
                timeout=(30, 300),
            )

            response.raise_for_status()

            total_bytes = 0

            with open(
                temporary_file,
                "wb"
            ) as output:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        output.write(chunk)
                        total_bytes += len(chunk)

            if total_bytes == 0:
                raise RuntimeError(
                    "Servidor retornou arquivo vazio."
                )

            temporary_file.replace(
                destination
            )

            logging.info(
                "Download concluído: %s",
                destination.name
            )
            logging.info(
                "Tamanho: %.2f MB",
                destination.stat().st_size
                / 1024
                / 1024
            )

            return destination

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
            ConnectionResetError,
            RuntimeError,
        ) as exc:
            logging.warning(
                "Falha no download na tentativa %s/%s",
                attempt,
                max_attempts
            )
            logging.warning(
                "Erro: %s",
                exc
            )

            if temporary_file.exists():
                try:
                    temporary_file.unlink()
                except Exception:
                    pass

            if attempt == max_attempts:
                raise RuntimeError(
                    "\n"
                    "DOWNLOAD DO INEP NÃO CONCLUÍDO.\n"
                    f"URL: {url}\n"
                    "Baixe o arquivo manualmente e coloque em:\n"
                    f"{ORIGINAL_DIR}\n"
                    "Depois execute novamente:\n"
                    "python -m src.bronze\n"
                ) from exc

            wait_seconds = attempt * 10

            logging.info(
                "Aguardando %s segundos antes da próxima tentativa...",
                wait_seconds
            )

            time.sleep(
                wait_seconds
            )


def file_name_from_url(
    url: str,
    fallback_name: str
):
    name = Path(
        urlparse(url).path
    ).name

    return (
        name
        if name
        else fallback_name
    )


def workbook_to_dataframe(
    path: Path
):
    logging.info(
        "Lendo planilha: %s",
        path.name
    )

    sheets = pd.read_excel(
        path,
        sheet_name=None,
        dtype=str,
        engine="openpyxl"
    )

    frames = []

    for sheet_name, dataframe in sheets.items():
        if dataframe is None or dataframe.empty:
            continue

        dataframe = (
            dataframe
            .dropna(how="all")
            .dropna(axis=1, how="all")
        )

        if dataframe.empty:
            continue

        dataframe.insert(
            0,
            "_aba_origem",
            sheet_name
        )

        frames.append(
            dataframe
        )

    if not frames:
        raise ValueError(
            f"Nenhuma tabela legível foi encontrada em {path.name}"
        )

    result = pd.concat(
        frames,
        ignore_index=True,
        sort=False
    )

    logging.info(
        "Planilha %s: %s registros",
        path.name,
        len(result)
    )

    return result


def find_largest_tabular_file(
    extracted_dir: Path
):
    candidates = []

    for extension in (
        "*.csv",
        "*.CSV",
        "*.txt",
        "*.TXT",
    ):
        candidates.extend(
            extracted_dir.rglob(
                extension
            )
        )

    if not candidates:
        raise FileNotFoundError(
            "O ZIP dos microdados não contém CSV ou TXT reconhecível."
        )

    largest_file = max(
        candidates,
        key=lambda path:
            path.stat().st_size
    )

    logging.info(
        "Arquivo principal dos microdados: %s",
        largest_file
    )

    return largest_file


def random_sample_large_csv(
    path: Path,
    sample_size: int,
    seed: int
):
    encoding, separator = (
        detect_encoding_and_separator(
            path
        )
    )

    logging.info(
        "Encoding detectado: %s",
        encoding
    )
    logging.info(
        "Separador detectado: %r",
        separator
    )

    rng = np.random.default_rng(
        seed
    )

    selected = None
    total_rows = 0

    reader = pd.read_csv(
        path,
        sep=separator,
        encoding=encoding,
        encoding_errors="replace",
        dtype=str,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        on_bad_lines="skip",
        engine="c",
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1
    ):
        total_rows += len(chunk)

        chunk[
            "__amostra_aleatoria__"
        ] = rng.random(
            len(chunk)
        )

        if selected is None:
            selected = chunk
        else:
            selected = pd.concat(
                [
                    selected,
                    chunk
                ],
                ignore_index=True
            )

        if len(selected) > sample_size:
            selected = (
                selected
                .nsmallest(
                    sample_size,
                    "__amostra_aleatoria__"
                )
            )

        logging.info(
            "Microdados | chunk %s | linhas processadas: %s",
            chunk_number,
            f"{total_rows:,}"
        )

    if (
        selected is None
        or selected.empty
    ):
        raise ValueError(
            "Nenhuma linha foi lida dos microdados."
        )

    selected = selected.nsmallest(
        min(
            sample_size,
            len(selected)
        ),
        "__amostra_aleatoria__"
    )

    selected = selected.drop(
        columns=[
            "__amostra_aleatoria__"
        ]
    )

    selected = selected.reset_index(
        drop=True
    )

    logging.info(
        "Amostra final: %s registros",
        len(selected)
    )

    return (
        selected,
        total_rows
    )


def run_bronze():
    logging.info(
        "=" * 72
    )
    logging.info(
        "CAMADA BRONZE — FONTE PRIMÁRIA OFICIAL INEP"
    )
    logging.info(
        "=" * 72
    )

    logging.info(
        "Diretório do projeto: %s",
        BASE_DIR
    )
    logging.info(
        "Bronze: %s",
        BRONZE_DIR
    )
    logging.info(
        "Originais: %s",
        ORIGINAL_DIR
    )

    links = discover_official_links()
    source_rows = []

    # --------------------------------------------------------
    # 1. Brasil e UFs
    # --------------------------------------------------------

    logging.info(
        "-" * 72
    )
    logging.info(
        "PROCESSANDO BRASIL E UFs"
    )
    logging.info(
        "-" * 72
    )

    url = links[
        "brasil_ufs"
    ]

    source_path = download(
        url,
        ORIGINAL_DIR
        /
        file_name_from_url(
            url,
            "resultados_e_metas_ufs_2024_2.xlsx"
        )
    )

    dataframe = workbook_to_dataframe(
        source_path
    )

    original_count = len(
        dataframe
    )

    dataframe = sample_df(
        dataframe,
        SAMPLE_SIZE,
        RANDOM_SEED
    )

    output_path = (
        BRONZE_DIR
        /
        "bronze_brasil_ufs.csv"
    )

    write_csv(
        dataframe,
        output_path
    )

    source_rows.append(
        {
            "fonte": "brasil_ufs",
            "url": url,
            "arquivo_original": str(source_path),
            "linhas_origem": original_count,
            "linhas_bronze": len(dataframe),
        }
    )

    logging.info(
        "Brasil/UF concluído."
    )

    # --------------------------------------------------------
    # 2. Municípios
    # --------------------------------------------------------

    logging.info(
        "-" * 72
    )
    logging.info(
        "PROCESSANDO MUNICÍPIOS"
    )
    logging.info(
        "-" * 72
    )

    url = links[
        "municipios"
    ]

    source_path = download(
        url,
        ORIGINAL_DIR
        /
        file_name_from_url(
            url,
            "resultados_e_metas_municipios_2024.xlsx"
        )
    )

    dataframe = workbook_to_dataframe(
        source_path
    )

    original_count = len(
        dataframe
    )

    dataframe = sample_df(
        dataframe,
        SAMPLE_SIZE,
        RANDOM_SEED
    )

    output_path = (
        BRONZE_DIR
        /
        "bronze_municipios.csv"
    )

    write_csv(
        dataframe,
        output_path
    )

    source_rows.append(
        {
            "fonte": "municipios",
            "url": url,
            "arquivo_original": str(source_path),
            "linhas_origem": original_count,
            "linhas_bronze": len(dataframe),
        }
    )

    logging.info(
        "Municípios concluído."
    )

    # --------------------------------------------------------
    # 3. Microdados
    # --------------------------------------------------------

    logging.info(
        "-" * 72
    )
    logging.info(
        "PROCESSANDO MICRODADOS"
    )
    logging.info(
        "-" * 72
    )

    url = links[
        "microdados"
    ]

    zip_path = download(
        url,
        ORIGINAL_DIR
        /
        file_name_from_url(
            url,
            "microdados_avaliacao_da_alfabetizacao_2024.zip"
        )
    )

    extracted_dir = (
        ORIGINAL_DIR
        /
        "microdados_extraidos"
    )

    extracted_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    existing_files = list(
        extracted_dir.rglob("*")
    )

    if not any(
        path.is_file()
        for path in existing_files
    ):
        logging.info(
            "Extraindo ZIP dos microdados..."
        )

        with zipfile.ZipFile(
            zip_path
        ) as archive:
            archive.extractall(
                extracted_dir
            )

        logging.info(
            "Extração concluída."
        )
    else:
        logging.info(
            "Microdados já extraídos. Extração ignorada."
        )

    microdata_file = (
        find_largest_tabular_file(
            extracted_dir
        )
    )

    sample, total_rows = (
        random_sample_large_csv(
            microdata_file,
            SAMPLE_SIZE,
            RANDOM_SEED
        )
    )

    output_path = (
        BRONZE_DIR
        /
        "bronze_alunos_5000.csv"
    )

    write_csv(
        sample,
        output_path
    )

    source_rows.append(
        {
            "fonte": "microdados",
            "url": url,
            "arquivo_original": str(zip_path),
            "arquivo_tabular": str(microdata_file),
            "linhas_origem": total_rows,
            "linhas_bronze": len(sample),
        }
    )

    logging.info(
        "Microdados concluídos."
    )

    # --------------------------------------------------------
    # Manifesto
    # --------------------------------------------------------

    manifest = pd.DataFrame(
        source_rows
    )

    manifest_path = (
        BRONZE_DIR
        /
        "_manifesto_fontes.csv"
    )

    write_csv(
        manifest,
        manifest_path
    )

    logging.info(
        "=" * 72
    )
    logging.info(
        "CAMADA BRONZE CONCLUÍDA"
    )
    logging.info(
        "=" * 72
    )

    logging.info(
        "Arquivos gerados:"
    )
    logging.info(
        "%s",
        BRONZE_DIR
        /
        "bronze_brasil_ufs.csv"
    )
    logging.info(
        "%s",
        BRONZE_DIR
        /
        "bronze_municipios.csv"
    )
    logging.info(
        "%s",
        BRONZE_DIR
        /
        "bronze_alunos_5000.csv"
    )
    logging.info(
        "%s",
        manifest_path
    )

    return manifest


if __name__ == "__main__":
    run_bronze()
