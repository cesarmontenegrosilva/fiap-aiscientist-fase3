from pathlib import Path

import pandas as pd

from src.config import BRONZE_DIR, SILVER_DIR


def print_schema(folder: Path):
    print(f"\n{'=' * 80}\n{folder.name.upper()}\n{'=' * 80}")
    for path in sorted(folder.glob("*.csv")):
        if path.name.startswith("_"):
            continue
        df = pd.read_csv(path, nrows=5, low_memory=False)
        print(f"\n{path.name}")
        print("-" * len(path.name))
        for col in df.columns:
            print(col)


if __name__ == "__main__":
    print_schema(BRONZE_DIR)
    print_schema(SILVER_DIR)
