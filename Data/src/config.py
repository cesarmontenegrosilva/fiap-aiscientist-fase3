from pathlib import Path


# ============================================================
# DIRETÓRIO BASE DO PROJETO
# ============================================================
#
# config.py está dentro da pasta:
#
# techallenge1/src/config.py
#
# Portanto:
# Path(__file__).resolve().parent.parent
#
# retorna automaticamente:
#
# G:\Meu Drive\FIAP\AI_Scientist\Fase3\techallenge1
#
# Assim não precisamos deixar caminho absoluto fixo.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# CAMADAS DA ARQUITETURA MEDALLION
# ============================================================

BRONZE_DIR = BASE_DIR / "Bronze"

SILVER_DIR = BASE_DIR / "Silver"

GOLD_DIR = BASE_DIR / "Gold"

LOG_DIR = BASE_DIR / "logs"


# Pasta onde os arquivos originais do INEP serão preservados
ORIGINAL_DIR = BRONZE_DIR / "_originais"


# ============================================================
# CRIAÇÃO AUTOMÁTICA DAS PASTAS
# ============================================================

for directory in [
    BRONZE_DIR,
    SILVER_DIR,
    GOLD_DIR,
    LOG_DIR,
    ORIGINAL_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# FONTE PRIMÁRIA
# ============================================================
#
# Página institucional oficial do INEP.
#
# A partir desta página o programa tenta localizar os links
# dos arquivos de resultados, metas e microdados.
# ============================================================

INEP_RESULTADOS_2024 = (
    "https://www.gov.br/inep/pt-br/"
    "areas-de-atuacao/"
    "avaliacao-e-exames-educacionais/"
    "avaliacao-da-alfabetizacao/"
    "resultados/2024"
)


# ============================================================
# LINKS DE FALLBACK
# ============================================================
#
# Caso o programa não consiga descobrir automaticamente
# os links na página do INEP, tenta estes endereços.
# ============================================================

FALLBACK_URLS = {

    "brasil_ufs": (
        "https://download.inep.gov.br/"
        "alfabetiza_brasil/"
        "resultados_e_metas_ufs_2024_2.xlsx"
    ),

    "municipios": (
        "https://download.inep.gov.br/"
        "alfabetiza_brasil/"
        "resultados_e_metas_municipios_2024.xlsx"
    ),

    "microdados": (
        "https://download.inep.gov.br/"
        "dados_abertos/"
        "microdados_avaliacao_da_alfabetizacao_2024.zip"
    ),
}


# ============================================================
# CONFIGURAÇÕES DA AMOSTRA
# ============================================================

# Quantidade máxima de linhas que queremos na Bronze
SAMPLE_SIZE = 5000


# Seed fixa:
# permite reproduzir a mesma amostra aleatória
RANDOM_SEED = 42


# ============================================================
# DOWNLOAD
# ============================================================

# Timeout utilizado nas requisições
REQUEST_TIMEOUT = 120


# ============================================================
# LEITURA DOS MICRODADOS
# ============================================================

# Quantas linhas serão processadas por vez
#
# Isso evita carregar o arquivo inteiro na memória.
CHUNK_SIZE = 100_000


# ============================================================
# CRITÉRIO DE ALFABETIZAÇÃO
# ============================================================

# Ponto de corte estabelecido no desafio
ALFABETIZACAO_CUTOFF = 743