from pathlib import Path
import requests

MODEL_URL = "https://github.com/1nfinityLoop/GreenSorter/releases/download/v0.1/model.pt"
MODEL_PATH = Path(__file__).resolve().parent / "model.pt"

if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
    print(f"[EcoScan] model.pt já existe: {MODEL_PATH}")
    raise SystemExit(0)

print("[EcoScan] Baixando model.pt oficial do GreenSorter...")
with requests.get(MODEL_URL, stream=True, timeout=180, allow_redirects=True) as response:
    response.raise_for_status()
    total = int(response.headers.get("content-length", "0"))
    downloaded = 0

    with MODEL_PATH.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r[EcoScan] {downloaded / total * 100:.1f}%", end="")

print(f"\n[EcoScan] Modelo salvo: {MODEL_PATH}")
print(f"[EcoScan] Tamanho: {MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB")
