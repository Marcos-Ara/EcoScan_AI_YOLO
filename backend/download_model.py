import os
import sys
import time
import zipfile
from pathlib import Path

import requests


# ============================================================
# EcoScan AI - Download seguro do checkpoint GreenSorter
# ============================================================

MODEL_URL = (
    "https://github.com/1nfinityLoop/GreenSorter/"
    "releases/download/v0.1/model.pt"
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pt"
PART_PATH = ROOT / "model.pt.part"

MIN_MODEL_SIZE = 1_000_000
EXPECTED_MODEL_SIZE = 74_768_207
MAX_ATTEMPTS = 3


def size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def validate_model(path: Path) -> None:
    """Valida o arquivo sem carregar os 72 MB na memória."""

    if not path.is_file():
        raise RuntimeError("model.pt não foi criado.")

    size = path.stat().st_size
    if size < MIN_MODEL_SIZE:
        raise RuntimeError(
            f"model.pt está pequeno demais: {size} bytes."
        )

    with path.open("rb") as handle:
        signature = handle.read(4)

    # O checkpoint oficial usado neste projeto é um ZIP do PyTorch.
    if signature != b"PK\x03\x04" or not zipfile.is_zipfile(path):
        raise RuntimeError(
            "O download não parece ser um checkpoint PyTorch válido. "
            "O arquivo recebido não é um ZIP válido."
        )

    # Não exigimos tamanho exatamente igual porque uma nova versão do
    # release pode mudar, mas avisamos se estiver diferente do conhecido.
    if size != EXPECTED_MODEL_SIZE:
        print(
            "[EcoScan] Aviso: tamanho diferente do release conhecido: "
            f"{size:,} bytes (esperado historicamente "
            f"{EXPECTED_MODEL_SIZE:,}).",
            flush=True,
        )


def download() -> None:
    print("=" * 60, flush=True)
    print("[EcoScan] Download do modelo oficial GreenSorter", flush=True)
    print(f"[EcoScan] URL: {MODEL_URL}", flush=True)
    print("=" * 60, flush=True)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(
                f"[EcoScan] Tentativa {attempt}/{MAX_ATTEMPTS}",
                flush=True,
            )

            PART_PATH.unlink(missing_ok=True)

            with requests.get(
                MODEL_URL,
                stream=True,
                timeout=(30, 300),
                allow_redirects=True,
            ) as response:
                response.raise_for_status()

                total = int(response.headers.get("content-length", "0"))
                downloaded = 0

                with PART_PATH.open("wb") as handle:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if not chunk:
                            continue

                        handle.write(chunk)
                        downloaded += len(chunk)

                        if total:
                            percent = downloaded / total * 100
                            print(
                                f"\r[EcoScan] {percent:6.1f}%",
                                end="",
                                flush=True,
                            )

            print(flush=True)

            validate_model(PART_PATH)
            os.replace(PART_PATH, MODEL_PATH)

            print("[EcoScan] Modelo válido.", flush=True)
            print(
                f"[EcoScan] Tamanho: {size_mb(MODEL_PATH):.1f} MB",
                flush=True,
            )
            print("[EcoScan] Download concluído.", flush=True)
            return

        except Exception as exc:
            PART_PATH.unlink(missing_ok=True)
            print(
                f"[EcoScan] Erro: {type(exc).__name__}: {exc}",
                flush=True,
            )

            if attempt < MAX_ATTEMPTS:
                print("[EcoScan] Tentando novamente em 5s...", flush=True)
                time.sleep(5)

    print("[EcoScan] Não foi possível baixar um model.pt válido.", flush=True)
    sys.exit(1)


def main() -> None:
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > MIN_MODEL_SIZE:
        try:
            validate_model(MODEL_PATH)
            print("[EcoScan] model.pt já existe e é válido.", flush=True)
            print(
                f"[EcoScan] Tamanho: {size_mb(MODEL_PATH):.1f} MB",
                flush=True,
            )
            return
        except Exception as exc:
            print(
                f"[EcoScan] model.pt existente é inválido: {exc}",
                flush=True,
            )
            MODEL_PATH.unlink(missing_ok=True)

    download()


if __name__ == "__main__":
    main()
