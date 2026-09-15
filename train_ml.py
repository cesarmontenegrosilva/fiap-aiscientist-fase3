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
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_validate,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    OrdinalEncoder,
)


RANDOM_SEED = 42
TARGET = "risco_nao_alfabetizado"
GROUP_COL = "id_escola"


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
MODELS_DIR = DATA_DIR / "models"
IMAGES_DIR = DATA_DIR / "images"

for p in [REPORTS_DIR, MODELS_DIR, IMAGES_DIR]:
    p.mkdir(parents=True, exist_ok=True)


AUDIT_NOT_FEATURES = {
    TARGET,
    "alfabetizado_oficial",
    "id_aluno",
    "id_escola",
    "codigo_municipio",
    "nome_municipio",
}


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


def main():
    warnings.filterwarnings("ignore")

    dataset_path = (
        MODELING_DIR / "dataset_modelagem_ml.csv"
    )

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

    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    if GROUP_COL not in df.columns:
        raise KeyError(
            f"A coluna {GROUP_COL} é necessária para separação por grupos."
        )

    # Grupos usados no StratifiedGroupKFold.
    # O fillna precisa receber um escalar ou uma Series alinhada ao índice.
    # Embora id_escola não tenha nulos na base atual, mantemos um fallback
    # robusto para futuras execuções.
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

    features = [
        c for c in df.columns
        if c not in AUDIT_NOT_FEATURES
    ]

    X = df[features].copy()
    y = df[TARGET].copy()

    # Segurança extra: remove qualquer constante que ainda tenha restado.
    constant_features = [
        c for c in X.columns
        if X[c].nunique(dropna=True) <= 1
    ]

    if constant_features:
        X = X.drop(columns=constant_features)

    numeric_cols = (
        X.select_dtypes(include=[np.number])
        .columns
        .tolist()
    )

    categorical_cols = [
        c for c in X.columns
        if c not in numeric_cols
    ]

    print("=" * 80)
    print("CONFIGURAÇÃO DO TREINAMENTO")
    print("=" * 80)
    print(f"Linhas: {len(X):,}")
    print(f"Features: {len(X.columns)}")
    print(f"Numéricas: {len(numeric_cols)}")
    print(f"Categóricas: {len(categorical_cols)}")
    print(f"Target positivo: {100*y.mean():.2f}%")
    print(f"Grupos (id_escola): {groups.nunique():,}")

    # ========================================================
    # HOLDOUT FINAL: 1 fold de 5 = ~20%
    # ========================================================
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
    groups_train = groups.iloc[train_idx].copy()

    # ========================================================
    # PREPROCESSAMENTO
    # ========================================================
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

    preprocessor_linear = ColumnTransformer(
        transformers=[
            ("num", numeric_scaled, numeric_cols),
            ("cat", categorical_onehot, categorical_cols),
        ],
        remainder="drop",
    )

    preprocessor_tree = ColumnTransformer(
        transformers=[
            ("num", numeric_plain, numeric_cols),
            ("cat", categorical_onehot, categorical_cols),
        ],
        remainder="drop",
    )

    preprocessor_hgb = ColumnTransformer(
        transformers=[
            ("num", numeric_plain, numeric_cols),
            ("cat", categorical_ordinal, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", preprocessor_linear),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", preprocessor_tree),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocess", preprocessor_hgb),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.08,
                        max_iter=250,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }

    # ========================================================
    # CV APENAS NO TREINO
    # ========================================================
    cv = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    rows = []

    for name, pipeline in models.items():
        print(f"\nAvaliando {name}...")

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            groups=groups_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        row = {"modelo": name}

        for metric in scoring:
            values = scores[f"test_{metric}"]

            row[f"{metric}_media"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))

        rows.append(row)

    cv_results = (
        pd.DataFrame(rows)
        .sort_values(
            "pr_auc_media",
            ascending=False,
        )
    )

    cv_results.to_csv(
        REPORTS_DIR / "metricas_modelos_cv.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_name = cv_results.iloc[0]["modelo"]
    best_pipeline = models[best_name]

    print("\nMelhor modelo inicial:", best_name)

    # ========================================================
    # TUNING DO VENCEDOR
    # ========================================================
    param_distributions = {
        "logistic_regression": {
            "model__C": np.logspace(-2, 1, 20),
        },
        "random_forest": {
            "model__n_estimators": [300, 400, 500, 700],
            "model__max_depth": [None, 8, 12, 18, 25],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", 0.5],
        },
        "hist_gradient_boosting": {
            "model__learning_rate": [0.03, 0.05, 0.08, 0.10],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__max_iter": [150, 250, 350],
            "model__l2_regularization": [0.0, 0.1, 1.0, 5.0],
        },
    }

    search = RandomizedSearchCV(
        estimator=best_pipeline,
        param_distributions=param_distributions[best_name],
        n_iter=10,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    search.fit(
        X_train,
        y_train,
        groups=groups_train,
    )

    final_model = search.best_estimator_

    # ========================================================
    # TESTE FINAL
    # ========================================================
    y_pred = final_model.predict(X_test)

    y_prob = final_model.predict_proba(X_test)[:, 1]

    test_metrics = metrics_dict(
        y_test,
        y_pred,
        y_prob,
    )

    test_metrics["modelo"] = best_name
    test_metrics["best_params"] = json.dumps(
        search.best_params_,
        ensure_ascii=False,
    )

    pd.DataFrame(
        [test_metrics]
    ).to_csv(
        REPORTS_DIR / "metricas_teste_final.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pred = pd.DataFrame(
        {
            "row_index": df.index[test_idx],
            "y_real": y_test.to_numpy(),
            "y_pred": y_pred,
            "prob_risco_nao_alfabetizado": y_prob,
        }
    )

    for c in [
        "id_aluno",
        "id_escola",
        "codigo_municipio",
        "nome_municipio",
    ]:
        if c in df.columns:
            pred[c] = df.iloc[test_idx][c].to_numpy()

    # sigla_uf é feature, mas também útil na auditoria.
    if "sigla_uf" in df.columns:
        pred["sigla_uf"] = df.iloc[test_idx]["sigla_uf"].to_numpy()

    pred.to_csv(
        REPORTS_DIR / "predictions_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    joblib.dump(
        final_model,
        MODELS_DIR / "best_model.joblib",
    )

    metadata = {
        "best_model": best_name,
        "target": TARGET,
        "positive_class": "1 = risco de não alfabetização",
        "group_column": GROUP_COL,
        "features": X.columns.tolist(),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "constant_features_removed": constant_features,
        "best_params": search.best_params_,
        "test_metrics": test_metrics,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "random_seed": RANDOM_SEED,
    }

    with (
        MODELS_DIR / "model_metadata.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    # ========================================================
    # GRÁFICOS
    # ========================================================
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        ax=ax,
    )
    ax.set_title("Matriz de confusão — teste")
    fig.tight_layout()
    fig.savefig(
        IMAGES_DIR / "ml_confusion_matrix.png",
        dpi=160,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(
        y_test,
        y_prob,
        ax=ax,
    )
    ax.set_title("Curva ROC — teste")
    fig.tight_layout()
    fig.savefig(
        IMAGES_DIR / "ml_roc_curve.png",
        dpi=160,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(
        y_test,
        y_prob,
        ax=ax,
    )
    ax.set_title("Curva Precision-Recall — teste")
    fig.tight_layout()
    fig.savefig(
        IMAGES_DIR / "ml_pr_curve.png",
        dpi=160,
    )
    plt.close(fig)

    print("\n" + "=" * 80)
    print("TREINAMENTO CONCLUÍDO")
    print("=" * 80)
    print("Melhor modelo:", best_name)
    print("Melhores parâmetros:", search.best_params_)
    print("\nMétricas de teste:")

    for k, v in test_metrics.items():
        if isinstance(v, (int, float, np.floating)):
            print(f"{k}: {float(v):.4f}")


if __name__ == "__main__":
    main()
