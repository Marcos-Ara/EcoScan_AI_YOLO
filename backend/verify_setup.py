import importlib
import os
import sys

REQUIRED = [
    "fastapi",
    "uvicorn",
    "cv2",
    "numpy",
    "inference_sdk",
]

print("============================================================")
print("EcoScan AI - Verificação Roboflow YOLO11")
print("============================================================")

failed = False

for module_name in REQUIRED:
    try:
        importlib.import_module(module_name)
        print(f"[OK] {module_name}")
    except Exception as exc:
        failed = True
        print(f"[ERRO] {module_name}: {exc}")

api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
model_id = os.getenv("ROBOFLOW_MODEL_ID", "waste-sorting-smyr8/2")
api_url = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")

print(f"[INFO] Modelo: {model_id}")
print(f"[INFO] API: {api_url}")
print(f"[INFO] API key: {'configurada' if api_key else 'NÃO configurada'}")

if not api_key:
    failed = True
    print("[ERRO] Defina ROBOFLOW_API_KEY.")

if failed:
    sys.exit(1)

print("[EcoScan] Ambiente local válido.")
