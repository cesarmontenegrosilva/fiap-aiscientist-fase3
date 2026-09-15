from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "build_modeling_dataset.py",
    "prepare_ml_features.py",
    "inspect_modeling_data_ml.py",
    "eda_fase3.py",
    "train_ml.py",
    "explain_ml.py",
    "aggregate_risk.py",
]


def main():
    base = Path(__file__).resolve().parent

    for script in SCRIPTS:
        path = base / script

        print("\n" + "=" * 80)
        print(f"EXECUTANDO: {script}")
        print("=" * 80)

        subprocess.run(
            [sys.executable, str(path)],
            check=True,
        )

    print("\n" + "=" * 80)
    print("PIPELINE DE MACHINE LEARNING FINALIZADA")
    print("=" * 80)


if __name__ == "__main__":
    main()
