import time
from datetime import datetime

from src.bronze import run_bronze
from src.silver import run_silver
from src.gold import run_gold


def main():

    print(
        "=" * 72
    )

    print(
        "FIAP TECH CHALLENGE — PIPELINE MEDALLION LOCAL"
    )

    print(
        "INEP -> Bronze -> Silver -> Gold"
    )

    print(
        "=" * 72
    )

    start = time.time()

    print(
        "\n[1/3] BRONZE\n"
    )

    run_bronze()

    print(
        "\n[2/3] SILVER\n"
    )

    run_silver()

    print(
        "\n[3/3] GOLD\n"
    )

    run_gold()

    elapsed = (
        time.time()
        -
        start
    )

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "PIPELINE CONCLUÍDO"
    )

    print(
        f"Data/hora: "
        f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    )

    print(
        f"Tempo total: "
        f"{elapsed:.1f} segundos"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
