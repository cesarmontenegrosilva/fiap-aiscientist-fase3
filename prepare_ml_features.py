from __future__ import annotations

from pathlib import Path
import pandas as pd


def find_data_dir() -> Path:
    cwd = Path.cwd().resolve()
    script_dir = Path(__file__).resolve().parent

    for base in [cwd, *cwd.parents, script_dir, *script_dir.parents]:
        if base.name.lower() == "data":
            return base

        candidate = base / "Data"
        if candidate.exists() and candidate.is_dir():
            return candidate

    raise FileNotFoundError("Pasta Data não encontrada.")


DATA_DIR = find_data_dir()
MODELING_DIR = DATA_DIR / "Modeling"
REPORTS_DIR = DATA_DIR / "reports"

MODELING_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT = MODELING_DIR / "dataset_modelagem.csv"
OUTPUT = MODELING_DIR / "dataset_modelagem_ml.csv"
REPORT = REPORTS_DIR / "feature_selection_report.csv"


# ============================================================
# FEATURES CURADAS PARA O MODELO
# ============================================================
#
# Objetivos:
# - remover identificadores;
# - remover variáveis constantes;
# - remover variáveis observadas no momento da prova;
# - remover leakage direto/indireto;
# - reduzir redundância geográfica;
# - reduzir multicolinearidade óbvia;
# - preservar variáveis educacionais, territoriais e
#   socioeconômicas interpretáveis.
#
# id_escola é mantido APENAS como coluna de grupo para split.
# codigo_municipio é mantido APENAS para auditoria/agregação.
#

MODEL_FEATURES = [
    # ----------------------------
    # Contexto individual/escolar
    # ----------------------------
    "tipo_dependencia",

    # ----------------------------
    # Contexto educacional anterior
    # ----------------------------
    "pct_alfabetizados_2023",
    "meta_2024",

    # ----------------------------
    # Censo Escolar 2023 - município
    # ----------------------------
    "numero_escolas",
    "pct_escolas_urbanas",

    # Dependência administrativa:
    # não usamos todas as categorias porque somam 100%.
    "pct_escolas_estaduais",
    "pct_escolas_municipais",
    "pct_escolas_privadas",

    "pct_escolas_internet",
    "pct_escolas_banda_larga",
    "pct_escolas_biblioteca_ou_sala_leitura",
    "pct_escolas_lab_informatica",
    "pct_escolas_agua_potavel",
    "pct_escolas_agua_rede_publica",
    "pct_escolas_energia_rede_publica",
    "pct_escolas_esgoto_rede_publica",
    "pct_escolas_coleta_lixo",
    "pct_escolas_quadra_esportes",
    "pct_escolas_patio_coberto",
    "pct_escolas_patio_descoberto",
    "pct_escolas_refeitorio",
    "pct_escolas_auditorio",

    "media_salas_utilizadas",
    "media_equip_multimidia",
    "media_desktop_aluno",
    "media_notebook_aluno",
    "media_tablet_aluno",

    # ----------------------------
    # IBGE 2022
    # ----------------------------
    "sigla_uf",
    "populacao_2022",
    "pct_urbana_2022",
    "pib_per_capita_2022",
]

TARGET = "risco_nao_alfabetizado"

AUDIT_COLUMNS = [
    "id_aluno",
    "id_escola",
    "codigo_municipio",
    "nome_municipio",
    "alfabetizado_oficial",
]


EXCLUSION_REASONS = {
    # Constantes
    "ano_avaliacao": "Constante na amostra; não adiciona informação preditiva.",
    "tipo_serie": "Constante na amostra; não adiciona informação preditiva.",

    # Identificadores
    "id_aluno": "Identificador; usado apenas para auditoria.",
    "id_escola": "Identificador; usado apenas como grupo no split.",
    "codigo_municipio": "Identificador territorial; usado apenas para auditoria/agregação.",
    "nome_municipio": "Alta cardinalidade; risco de memorizar municípios.",
    "codigo_uf": "Código identificador; sigla_uf representa a UF de forma categórica.",

    # Variáveis do momento da avaliação
    "presenca_lp": "Observada no momento da prova; não disponível em cenário preventivo.",
    "preenchimento_lp": "Observada no momento da prova; não disponível em cenário preventivo.",
    "codigo_caderno_lp": "Característica operacional da prova; não é fator explicativo estrutural.",
    "peso_aluno_lp": "Peso amostral/estatístico; não é característica preditiva do aluno.",

    # Target/leakage
    "alfabetizado_oficial": "Target original; mantido apenas para auditoria.",
    "proficiencia_lp": "Leakage direto; a proficiência determina a classificação.",
    "alfabetizado_calculado_743": "Reconstrução do target.",
    "divergencia_indicador_743": "Derivada diretamente do target/proficiência.",
    "pct_alfabetizados_2024": "Resultado agregado do mesmo período do target.",
    "variacao_2023_2024_pp": "Contém o resultado de 2024.",
    "gap_meta_2024_pp": "Contém o resultado de 2024.",
    "atingiu_meta_2024": "Contém o resultado de 2024.",
    "ranking_amostra_brasil_2024": "Derivado do resultado de 2024.",
    "ranking_amostra_uf_2024": "Derivado do resultado de 2024.",
    "ranking_ufs_2024": "Derivado do resultado de 2024.",
    "nivel_alfabetizacao": "Pode ser derivado do desempenho agregado.",

    # Redundâncias do Censo Escolar
    "pct_escolas_rurais": "Redundante com pct_escolas_urbanas (somam aproximadamente 100%).",
    "pct_escolas_federais": "Retirada para evitar dependência linear entre categorias administrativas.",
    "pct_escolas_biblioteca": "Sobreposição com indicador combinado biblioteca/sala de leitura.",
    "pct_escolas_sala_leitura": "Sobreposição com indicador combinado biblioteca/sala de leitura.",

    # Redundâncias geográficas IBGE
    "nome_municipio_ibge": "Duplicata/alta cardinalidade; não usada no modelo.",
    "codigo_uf_ibge": "Duplicata do contexto de UF.",
    "sigla_uf_ibge": "Duplicata de sigla_uf.",
    "nome_uf_ibge": "Duplicata de sigla_uf.",
    "codigo_regiao": "Código territorial; não usado como variável numérica.",
    "sigla_regiao": "Derivável da UF; removida para reduzir redundância.",
    "nome_regiao": "Derivável da UF; removida para reduzir redundância.",
    "codigo_regiao_imediata": "Alta cardinalidade e código identificador.",
    "nome_regiao_imediata": "Alta cardinalidade; risco de overfitting.",
    "codigo_regiao_intermediaria": "Código identificador territorial.",
    "nome_regiao_intermediaria": "Alta cardinalidade; removida por parcimônia.",
    "codigo_microrregiao": "Código identificador territorial.",
    "nome_microrregiao": "Alta cardinalidade; risco de overfitting.",
    "codigo_mesorregiao": "Código identificador territorial.",
    "nome_mesorregiao": "Alta cardinalidade; removida por parcimônia.",

    # Redundâncias demográficas/econômicas
    "pop_rural_2022": "Redundante com população total e percentual urbano.",
    "pop_total_urbano_rural_2022": "Redundante com populacao_2022.",
    "pop_urbana_2022": "Redundante com população total e percentual urbano.",
    "pct_rural_2022": "Redundante com pct_urbana_2022 (somam aproximadamente 100%).",
    "pib_2022_mil_reais": "Redundante com população e PIB per capita.",
}


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {INPUT}\n"
            "Execute primeiro: python build_modeling_dataset.py"
        )

    df = pd.read_csv(
        INPUT,
        low_memory=False,
        encoding="utf-8-sig",
    )

    missing_features = [
        c for c in MODEL_FEATURES
        if c not in df.columns
    ]

    if missing_features:
        print(
            "AVISO: algumas features planejadas não existem e serão ignoradas:"
        )
        for c in missing_features:
            print(" -", c)

    selected_features = [
        c for c in MODEL_FEATURES
        if c in df.columns
    ]

    audit = [
        c for c in AUDIT_COLUMNS
        if c in df.columns
    ]

    if TARGET not in df.columns:
        raise KeyError(f"Target ausente: {TARGET}")

    final_columns = audit + selected_features + [TARGET]

    # Remove duplicação de nomes caso uma coluna apareça em mais de um grupo.
    final_columns = list(dict.fromkeys(final_columns))

    ml = df[final_columns].copy()

    # Remove colunas constantes que tenham passado pela seleção por acaso.
    constant_features = [
        c for c in selected_features
        if ml[c].nunique(dropna=True) <= 1
    ]

    if constant_features:
        ml = ml.drop(columns=constant_features)
        selected_features = [
            c for c in selected_features
            if c not in constant_features
        ]

    ml.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    report_rows = []

    for col in df.columns:
        if col == TARGET:
            status = "target"
            reason = "Target do modelo: 1 = risco de não alfabetização."
        elif col in audit:
            status = "auditoria/grupo"
            reason = EXCLUSION_REASONS.get(
                col,
                "Mantida apenas para auditoria ou agrupamento."
            )
        elif col in selected_features:
            status = "feature"
            reason = "Selecionada para modelagem."
        elif col in constant_features:
            status = "excluida"
            reason = "Constante na amostra."
        else:
            status = "excluida"
            reason = EXCLUSION_REASONS.get(
                col,
                "Excluída por parcimônia, redundância ou ausência de justificativa preditiva."
            )

        report_rows.append(
            {
                "coluna": col,
                "status": status,
                "motivo": reason,
                "dtype_original": str(df[col].dtype),
                "nulos": int(df[col].isna().sum()),
                "nulos_pct": round(100 * df[col].isna().mean(), 2),
                "unicos": int(df[col].nunique(dropna=True)),
            }
        )

    report = pd.DataFrame(report_rows)

    report.to_csv(
        REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("PREPARAÇÃO DE FEATURES CONCLUÍDA")
    print("=" * 80)
    print(f"Arquivo original: {INPUT}")
    print(f"Arquivo ML:       {OUTPUT}")
    print(f"Relatório:        {REPORT}")
    print()
    print(f"Linhas: {len(ml):,}")
    print(f"Features selecionadas: {len(selected_features)}")
    print(f"Colunas de auditoria/grupo: {len(audit)}")
    print()
    print("FEATURES:")
    for col in selected_features:
        print(" -", col)


if __name__ == "__main__":
    main()
