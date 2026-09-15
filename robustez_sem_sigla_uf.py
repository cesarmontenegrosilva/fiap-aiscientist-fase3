from __future__ import annotations

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder


RANDOM_SEED = 42
TARGET = "risco_nao_alfabetizado"
GROUP_COL = "id_escola"
FEATURE_TO_REMOVE = "sigla_uf"


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

for p in [REPORTS_DIR, IMAGES_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def metrics_dict(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_risco": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall_risco": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_risco": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def build_pipeline(
    best_model_name: str,
    best_params: dict,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:

    numeric_scaled = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    numeric_plain = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_onehot = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                ),
            ),
        ]
    )

    categorical_ordinal = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    clean_params = {
        key.replace("model__", "", 1): value
        for key, value in best_params.items()
        if key.startswith("model__")
    }

    if best_model_name == "logistic_regression":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_scaled, numeric_cols),
                ("cat", categorical_onehot, categorical_cols),
            ],
            remainder="drop",
        )

        estimator = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            **clean_params,
        )

    elif best_model_name == "random_forest":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_plain, numeric_cols),
                ("cat", categorical_onehot, categorical_cols),
            ],
            remainder="drop",
        )

        estimator = RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            **clean_params,
        )

    elif best_model_name == "hist_gradient_boosting":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_plain, numeric_cols),
                ("cat", categorical_ordinal, categorical_cols),
            ],
            remainder="drop",
            sparse_threshold=0,
        )

        estimator = HistGradientBoostingClassifier(
            random_state=RANDOM_SEED,
            **clean_params,
        )

    else:
        raise ValueError(
            f"Modelo não suportado pelo teste de robustez: {best_model_name}"
        )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )


def main():
    warnings.filterwarnings("ignore")

    dataset_path = MODELING_DIR / "dataset_modelagem_ml.csv"
    metadata_path = MODELS_DIR / "model_metadata.json"
    baseline_metrics_path = REPORTS_DIR / "metricas_teste_final.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    if not baseline_metrics_path.exists():
        raise FileNotFoundError(baseline_metrics_path)

    df = pd.read_csv(
        dataset_path,
        low_memory=False,
        encoding="utf-8-sig",
    )

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    baseline = pd.read_csv(
        baseline_metrics_path,
        encoding="utf-8-sig",
    )

    best_model_name = metadata["best_model"]
    best_params = metadata.get("best_params", {})
    original_features = metadata["features"]

    if FEATURE_TO_REMOVE not in original_features:
        raise ValueError(
            f"{FEATURE_TO_REMOVE} não está entre as features do modelo original."
        )

    robust_features = [
        c for c in original_features
        if c != FEATURE_TO_REMOVE
    ]

    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    # Mesma construção de grupos usada no treino principal.
    fallback_groups = pd.Series(
        [f"SEM_ESCOLA_{i}" for i in df.index],
        index=df.index,
        dtype="string",
    )

    groups = (
        df[GROUP_COL]
        .astype("string")
        .fillna(fallback_groups)
    )

    X = df[robust_features].copy()
    y = df[TARGET].copy()

    numeric_cols = (
        X.select_dtypes(include=[np.number])
        .columns
        .tolist()
    )

    categorical_cols = [
        c for c in X.columns
        if c not in numeric_cols
    ]

    # IMPORTANTE:
    # mesmo StratifiedGroupKFold, mesmo random_state, mesmo primeiro fold.
    # Assim o conjunto de teste é o mesmo da execução principal.
    outer = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    train_idx, test_idx = next(
        outer.split(X, y, groups)
    )

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    pipeline = build_pipeline(
        best_model_name=best_model_name,
        best_params=best_params,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    print("=" * 80)
    print("TESTE DE ROBUSTEZ — MODELO SEM sigla_uf")
    print("=" * 80)
    print(f"Modelo original: {best_model_name}")
    print(f"Feature removida: {FEATURE_TO_REMOVE}")
    print(f"Features restantes: {len(robust_features)}")
    print(f"Treino: {len(train_idx):,}")
    print(f"Teste:  {len(test_idx):,}")
    print()

    pipeline.fit(
        X_train,
        y_train,
    )

    y_pred = pipeline.predict(
        X_test
    )

    y_prob = pipeline.predict_proba(
        X_test
    )[:, 1]

    robust_metrics = metrics_dict(
        y_test,
        y_pred,
        y_prob,
    )

    # --------------------------------------------------------
    # Comparação com baseline
    # --------------------------------------------------------
    metrics_order = [
        "accuracy",
        "balanced_accuracy",
        "precision_risco",
        "recall_risco",
        "f1_risco",
        "roc_auc",
        "pr_auc",
    ]

    comparison_rows = []

    for metric in metrics_order:
        baseline_value = float(
            baseline.iloc[0][metric]
        )

        robust_value = float(
            robust_metrics[metric]
        )

        delta = (
            robust_value
            - baseline_value
        )

        delta_pct = (
            100 * delta / baseline_value
            if baseline_value != 0
            else np.nan
        )

        comparison_rows.append(
            {
                "metrica": metric,
                "modelo_com_sigla_uf": baseline_value,
                "modelo_sem_sigla_uf": robust_value,
                "delta_absoluto": delta,
                "delta_percentual": delta_pct,
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison_path = (
        REPORTS_DIR
        / "robustez_sem_sigla_uf.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Predições
    # --------------------------------------------------------
    pred = pd.DataFrame(
        {
            "row_index": df.index[test_idx],
            "y_real": y_test.to_numpy(),
            "y_pred_sem_uf": y_pred,
            "prob_risco_sem_uf": y_prob,
        }
    )

    for c in [
        "id_aluno",
        "id_escola",
        "codigo_municipio",
        "nome_municipio",
        "sigla_uf",
    ]:
        if c in df.columns:
            pred[c] = df.iloc[test_idx][c].to_numpy()

    pred.to_csv(
        REPORTS_DIR
        / "predictions_test_sem_sigla_uf.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Modelo alternativo
    # --------------------------------------------------------
    joblib.dump(
        pipeline,
        MODELS_DIR
        / "robustness_model_sem_sigla_uf.joblib",
    )

    # --------------------------------------------------------
    # Gráfico
    # --------------------------------------------------------
    plot_df = comparison[
        comparison["metrica"].isin(
            [
                "balanced_accuracy",
                "recall_risco",
                "f1_risco",
                "roc_auc",
                "pr_auc",
            ]
        )
    ].copy()

    x = np.arange(
        len(plot_df)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width / 2,
        plot_df["modelo_com_sigla_uf"],
        width,
        label="Com sigla_uf",
    )

    ax.bar(
        x + width / 2,
        plot_df["modelo_sem_sigla_uf"],
        width,
        label="Sem sigla_uf",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        plot_df["metrica"],
        rotation=30,
        ha="right",
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_ylabel(
        "Métrica"
    )

    ax.set_title(
        "Teste de robustez: desempenho com e sem sigla_uf"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        IMAGES_DIR
        / "ml_robustez_sem_sigla_uf.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # Resumo automático
    # --------------------------------------------------------
    pr_delta = comparison.loc[
        comparison["metrica"] == "pr_auc",
        "delta_absoluto",
    ].iloc[0]

    roc_delta = comparison.loc[
        comparison["metrica"] == "roc_auc",
        "delta_absoluto",
    ].iloc[0]

    bal_delta = comparison.loc[
        comparison["metrica"] == "balanced_accuracy",
        "delta_absoluto",
    ].iloc[0]

    summary = {
        "modelo_original": best_model_name,
        "feature_removida": FEATURE_TO_REMOVE,
        "features_restantes": len(robust_features),
        "delta_pr_auc": float(pr_delta),
        "delta_roc_auc": float(roc_delta),
        "delta_balanced_accuracy": float(bal_delta),
        "interpretacao": (
            "Queda pequena: o modelo não depende fortemente de sigla_uf."
            if abs(pr_delta) < 0.02 and abs(roc_delta) < 0.02
            else
            "Queda relevante: parte importante do desempenho depende de sigla_uf."
        ),
    }

    with (
        REPORTS_DIR
        / "robustez_sem_sigla_uf_resumo.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(comparison.to_string(index=False))
    print()
    print("Interpretação automática:")
    print(summary["interpretacao"])
    print()
    print("Arquivo principal:")
    print(comparison_path)


if __name__ == "__main__":
    main()
