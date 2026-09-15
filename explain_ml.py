from __future__ import annotations

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance


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
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = DATA_DIR / "reports"
IMAGES_DIR = DATA_DIR / "images"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def to_dense(matrix):
    """
    Converte matriz esparsa em ndarray somente na pequena amostra
    utilizada pelo SHAP.
    """
    if hasattr(matrix, "toarray"):
        return matrix.toarray()

    return np.asarray(matrix)


def clean_transformed_feature_name(name: str) -> str:
    """
    Remove prefixos gerados pelo ColumnTransformer:
    num__feature -> feature
    cat__sigla_uf_PE -> sigla_uf_PE
    """
    if "__" in name:
        return name.split("__", 1)[1]

    return name


def original_feature_from_transformed(
    transformed_name: str,
    original_features: list[str],
) -> str:
    """
    Mapeia a feature transformada para a feature original.

    Exemplos:
    sigla_uf_PE -> sigla_uf
    pib_per_capita_2022 -> pib_per_capita_2022
    """
    clean = clean_transformed_feature_name(transformed_name)

    if clean in original_features:
        return clean

    # Usa primeiro nomes maiores para evitar matches parciais indevidos.
    for feature in sorted(original_features, key=len, reverse=True):
        if clean.startswith(feature + "_"):
            return feature

    return clean


def select_positive_class_shap(values_array: np.ndarray) -> np.ndarray:
    """
    Normaliza diferentes formatos retornados pelo SHAP.

    Para classificadores binários, algumas versões retornam:
    (n_amostras, n_features, 2).

    Nesse caso usamos a classe positiva = 1 (risco).
    """
    arr = np.asarray(values_array)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        # Formato moderno: amostras x features x outputs
        if arr.shape[-1] == 2:
            return arr[:, :, 1]

        if arr.shape[-1] == 1:
            return arr[:, :, 0]

        # Fallback para formatos incomuns.
        return arr[:, :, -1]

    raise ValueError(
        f"Formato SHAP inesperado: shape={arr.shape}"
    )


def main():
    warnings.filterwarnings("ignore")

    dataset_path = MODELING_DIR / "dataset_modelagem_ml.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_path} não existe.\n"
            "Execute primeiro: python prepare_ml_features.py"
        )

    df = pd.read_csv(
        dataset_path,
        low_memory=False,
        encoding="utf-8-sig",
    )

    pred = pd.read_csv(
        REPORTS_DIR / "predictions_test.csv",
        low_memory=False,
        encoding="utf-8-sig",
    )

    model = joblib.load(
        MODELS_DIR / "best_model.joblib"
    )

    with (
        MODELS_DIR / "model_metadata.json"
    ).open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    features = metadata["features"]
    target = metadata["target"]
    best_model_name = metadata.get("best_model", "desconhecido")

    test_indices = (
        pred["row_index"]
        .astype(int)
        .to_numpy()
    )

    X_test = df.loc[
        test_indices,
        features,
    ].copy()

    y_test = (
        df.loc[
            test_indices,
            target,
        ]
        .astype(int)
    )

    # ========================================================
    # 1. PERMUTATION IMPORTANCE
    # ========================================================

    print("Calculando Permutation Importance...")

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values(
        "importance_mean",
        ascending=False,
    )

    importance.to_csv(
        REPORTS_DIR / "permutation_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top = (
        importance
        .head(25)
        .sort_values("importance_mean")
    )

    fig, ax = plt.subplots(figsize=(10, 9))

    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
    )

    ax.set_title(
        "Top 25 variáveis — Permutation Importance (PR-AUC)"
    )

    ax.set_xlabel(
        "Queda média de PR-AUC ao permutar a variável"
    )

    fig.tight_layout()

    fig.savefig(
        IMAGES_DIR / "ml_permutation_importance.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # 2. SHAP APÓS O PREPROCESSAMENTO
    # ========================================================

    shap_status = {
        "executado": False,
        "modelo": best_model_name,
        "mensagem": "",
    }

    try:
        import shap

        if not hasattr(model, "named_steps"):
            raise TypeError(
                "O modelo salvo não é uma Pipeline do Scikit-learn."
            )

        if (
            "preprocess" not in model.named_steps
            or "model" not in model.named_steps
        ):
            raise KeyError(
                "A Pipeline precisa conter as etapas "
                "'preprocess' e 'model'."
            )

        preprocessor = model.named_steps["preprocess"]
        estimator = model.named_steps["model"]

        # Amostra pequena o suficiente para interpretação rápida,
        # mas grande o suficiente para representar o holdout.
        sample_raw = X_test.sample(
            n=min(150, len(X_test)),
            random_state=42,
        )

        background_raw = X_test.sample(
            n=min(60, len(X_test)),
            random_state=7,
        )

        print(
            "Transformando dados para o espaço numérico usado pelo modelo..."
        )

        sample_transformed = to_dense(
            preprocessor.transform(sample_raw)
        )

        background_transformed = to_dense(
            preprocessor.transform(background_raw)
        )

        # Recupera os nomes depois de OneHot/Ordinal/etc.
        try:
            transformed_feature_names = (
                preprocessor.get_feature_names_out()
                .astype(str)
                .tolist()
            )

        except Exception:
            transformed_feature_names = [
                f"feature_{i}"
                for i in range(sample_transformed.shape[1])
            ]

        if (
            len(transformed_feature_names)
            != sample_transformed.shape[1]
        ):
            transformed_feature_names = [
                f"feature_{i}"
                for i in range(sample_transformed.shape[1])
            ]

        print(
            f"Calculando SHAP para {best_model_name} "
            f"com {sample_transformed.shape[0]} amostras e "
            f"{sample_transformed.shape[1]} features transformadas..."
        )

        # O SHAP recebe somente números, exatamente como o estimador final.
        explainer = shap.Explainer(
            estimator,
            background_transformed,
            feature_names=transformed_feature_names,
        )

        shap_values = explainer(
            sample_transformed
        )

        positive_values = select_positive_class_shap(
            shap_values.values
        )

        # ----------------------------------------------------
        # Importância SHAP no espaço transformado
        # ----------------------------------------------------

        mean_abs = np.abs(
            positive_values
        ).mean(axis=0)

        shap_transformed = pd.DataFrame(
            {
                "feature_transformada": transformed_feature_names,
                "mean_abs_shap": mean_abs,
            }
        )

        shap_transformed["feature_original"] = (
            shap_transformed[
                "feature_transformada"
            ].map(
                lambda name:
                    original_feature_from_transformed(
                        name,
                        features,
                    )
            )
        )

        shap_transformed = shap_transformed.sort_values(
            "mean_abs_shap",
            ascending=False,
        )

        shap_transformed.to_csv(
            REPORTS_DIR / "shap_importance_transformed.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # ----------------------------------------------------
        # Agrega OneHot/etc. de volta para a feature original
        # ----------------------------------------------------

        shap_original = (
            shap_transformed
            .groupby(
                "feature_original",
                as_index=False,
            )["mean_abs_shap"]
            .sum()
            .sort_values(
                "mean_abs_shap",
                ascending=False,
            )
        )

        shap_original.to_csv(
            REPORTS_DIR / "shap_importance.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # ----------------------------------------------------
        # Gráfico agregado — mais fácil de explicar no trabalho
        # ----------------------------------------------------

        top_shap = (
            shap_original
            .head(25)
            .sort_values("mean_abs_shap")
        )

        fig, ax = plt.subplots(figsize=(10, 9))

        ax.barh(
            top_shap["feature_original"],
            top_shap["mean_abs_shap"],
        )

        ax.set_title(
            "Top 25 variáveis — importância média absoluta SHAP"
        )

        ax.set_xlabel(
            "Média de |SHAP value|"
        )

        fig.tight_layout()

        fig.savefig(
            IMAGES_DIR / "ml_shap_importance.png",
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(fig)

        # ----------------------------------------------------
        # Beeswarm transformado (opcional, mas útil visualmente)
        # ----------------------------------------------------

        try:
            explanation = shap.Explanation(
                values=positive_values,
                data=sample_transformed,
                feature_names=transformed_feature_names,
            )

            plt.figure()

            shap.plots.beeswarm(
                explanation,
                max_display=20,
                show=False,
            )

            plt.tight_layout()

            plt.savefig(
                IMAGES_DIR / "ml_shap_beeswarm.png",
                dpi=160,
                bbox_inches="tight",
            )

            plt.close()

        except Exception as plot_exc:
            # Não invalida o SHAP; só registra que o beeswarm falhou.
            print(
                "AVISO: SHAP foi calculado, mas o beeswarm "
                f"não pôde ser gerado: {type(plot_exc).__name__}: {plot_exc}"
            )

        shap_status = {
            "executado": True,
            "modelo": best_model_name,
            "amostras_explicadas": int(
                sample_transformed.shape[0]
            ),
            "features_transformadas": int(
                sample_transformed.shape[1]
            ),
            "mensagem": (
                "SHAP executado com sucesso após o pré-processamento. "
                "A interpretação principal foi agregada novamente "
                "para as features originais."
            ),
        }

    except Exception as exc:
        shap_status = {
            "executado": False,
            "modelo": best_model_name,
            "mensagem": (
                "SHAP não foi executado. "
                "Permutation Importance continua válida. "
                f"Motivo: {type(exc).__name__}: {exc}"
            ),
        }

    # ========================================================
    # STATUS
    # ========================================================

    with (
        REPORTS_DIR / "shap_status.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            shap_status,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 78)
    print("INTERPRETABILIDADE CONCLUÍDA")
    print("=" * 78)
    print(shap_status["mensagem"])

    if shap_status["executado"]:
        print()
        print(
            "Arquivos SHAP:"
        )
        print(
            REPORTS_DIR / "shap_importance.csv"
        )
        print(
            REPORTS_DIR / "shap_importance_transformed.csv"
        )
        print(
            IMAGES_DIR / "ml_shap_importance.png"
        )
        print(
            IMAGES_DIR / "ml_shap_beeswarm.png"
        )


if __name__ == "__main__":
    main()
