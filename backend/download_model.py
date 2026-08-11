from pathlib import Path
import sys
import time

import requests


# ============================================================
# CONFIGURAÇÃO
# ============================================================

MODEL_URL = (
    "https://github.com/1nfinityLoop/GreenSorter/"
    "releases/download/v0.1/model.pt"
)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "model.pt"
)


# ============================================================
# VERIFICAR MODELO EXISTENTE
# ============================================================

if (
    MODEL_PATH.exists()
    and MODEL_PATH.stat().st_size > 1_000_000
):

    size_mb = (
        MODEL_PATH.stat().st_size
        / 1024
        / 1024
    )

    print(
        f"[EcoScan] model.pt já existe."
    )

    print(
        f"[EcoScan] Tamanho: {size_mb:.1f} MB"
    )

    sys.exit(0)


# ============================================================
# DOWNLOAD
# ============================================================

print("=" * 60)

print(
    "[EcoScan] Baixando modelo oficial GreenSorter..."
)

print(
    f"[EcoScan] URL: {MODEL_URL}"
)

print("=" * 60)


MAX_ATTEMPTS = 3


for attempt in range(
    1,
    MAX_ATTEMPTS + 1
):

    try:

        print(
            f"[EcoScan] Tentativa "
            f"{attempt}/{MAX_ATTEMPTS}"
        )


        with requests.get(
            MODEL_URL,
            stream=True,
            timeout=(30, 300),
            allow_redirects=True,
        ) as response:

            response.raise_for_status()


            total = int(
                response.headers.get(
                    "content-length",
                    "0"
                )
            )


            downloaded = 0


            with MODEL_PATH.open(
                "wb"
            ) as f:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if not chunk:
                        continue


                    f.write(chunk)

                    downloaded += len(chunk)


                    if total:

                        percent = (
                            downloaded
                            / total
                            * 100
                        )

                        print(
                            f"\r[EcoScan] "
                            f"{percent:.1f}%",
                            end="",
                            flush=True
                        )


        print()


        # ----------------------------------------------------
        # VALIDAR DOWNLOAD
        # ----------------------------------------------------

        if not MODEL_PATH.exists():

            raise RuntimeError(
                "Arquivo model.pt não foi criado."
            )


        size = MODEL_PATH.stat().st_size


        if size <= 1_000_000:

            MODEL_PATH.unlink(
                missing_ok=True
            )

            raise RuntimeError(
                "Modelo baixado parece inválido "
                "ou está incompleto."
            )


        size_mb = (
            size
            / 1024
            / 1024
        )


        print(
            f"[EcoScan] Modelo salvo."
        )

        print(
            f"[EcoScan] Tamanho: "
            f"{size_mb:.1f} MB"
        )

        print(
            "[EcoScan] Download concluído."
        )


        sys.exit(0)


    except Exception as error:

        print()

        print(
            f"[EcoScan] Erro no download: "
            f"{error}"
        )


        if attempt < MAX_ATTEMPTS:

            print(
                "[EcoScan] "
                "Tentando novamente..."
            )

            time.sleep(5)

        else:

            print(
                "[EcoScan] "
                "Não foi possível baixar "
                "o modelo."
            )

            sys.exit(1)