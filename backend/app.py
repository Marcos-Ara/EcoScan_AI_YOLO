import asyncio
import base64
import gc
import os
import threading
import time
import unicodedata
from typing import Any

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com").rstrip("/")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "").strip()
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "waste-sorting-smyr8/2").strip()
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.40"))
IOU = float(os.getenv("IOU", "0.45"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "1500000"))
MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "960"))
MAX_DETECTIONS = max(1, min(20, int(os.getenv("MAX_DETECTIONS", "1"))))
FOCUS_MODE = os.getenv("FOCUS_MODE", "true").lower() in {"1", "true", "yes", "on"}
FOCUS_CROP_RATIO = min(1.0, max(0.55, float(os.getenv("FOCUS_CROP_RATIO", "0.90"))))
FOCUS_MAX_DETECTIONS = max(1, min(3, int(os.getenv("FOCUS_MAX_DETECTIONS", "1"))))
DEBUG_PREDICTIONS = os.getenv("DEBUG_PREDICTIONS", "false").lower() in {"1", "true", "yes", "on"}
ROBOFLOW_TIMEOUT_SECONDS = float(os.getenv("ROBOFLOW_TIMEOUT_SECONDS", "30"))
INFERENCE_LOCK = threading.Lock()

CLASS_MAP = {
    "paper": "papel",
    "paper_board": "papel",
    "paperboard": "papel",
    "cardboard": "papel",
    "plastic": "plastico",
    "plastics": "plastico",
    "glass": "vidro",
    "metal": "metal",
}

RULES = {
    "papel": {
        "category": "Papel", "bin": "🔵 Azul", "destination": "Reciclagem",
        "decomposition": "3–6 meses",
        "fact": "Papel e papelão devem estar, de preferência, secos e sem restos de comida.",
    },
    "plastico": {
        "category": "Plástico", "bin": "🔴 Vermelha", "destination": "Reciclagem",
        "decomposition": "Varia por material",
        "fact": "Garrafas PET, embalagens e outros plásticos devem ir para a coleta seletiva.",
    },
    "vidro": {
        "category": "Vidro", "bin": "🟢 Verde", "destination": "Reciclagem",
        "decomposition": "Muito longo",
        "fact": "Vidro deve ser encaminhado para a coleta seletiva.",
    },
    "metal": {
        "category": "Metal", "bin": "🟡 Amarela", "destination": "Reciclagem",
        "decomposition": "Varia por material",
        "fact": "Latas, tampas e outros metais devem ser encaminhados para reciclagem.",
    },
}

print("=" * 60, flush=True)
print("[EcoScan] Inicializando runtime Roboflow...", flush=True)
print(f"[EcoScan] API: {ROBOFLOW_API_URL}", flush=True)
print(f"[EcoScan] Modelo: {ROBOFLOW_MODEL_ID}", flush=True)
print(f"[EcoScan] Confidence: {CONFIDENCE}", flush=True)
print(f"[EcoScan] IoU: {IOU}", flush=True)
print(f"[EcoScan] Focus mode: {FOCUS_MODE}", flush=True)
print(f"[EcoScan] Focus crop: {FOCUS_CROP_RATIO:.2f}", flush=True)
print(f"[EcoScan] Focus max detections: {FOCUS_MAX_DETECTIONS}", flush=True)
print(f"[EcoScan] Roboflow API key: {'configured' if ROBOFLOW_API_KEY else 'missing'}", flush=True)
print("=" * 60, flush=True)

app = FastAPI(
    title="EcoScan AI YOLO API",
    version="3.1.0",
    description="EcoScan usando waste-sorting-smyr8/2 hospedado pelo Roboflow.",
)

allowed_origin = os.getenv("ALLOWED_ORIGIN", "https://marcos-ara.github.io")
origins = [x.strip() for x in allowed_origin.split(",") if x.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "EcoScan AI YOLO API",
        "status": "online",
        "model": ROBOFLOW_MODEL_ID,
        "runtime": "Roboflow Serverless",
        "version": "3.1.0",
        "roboflow_configured": bool(ROBOFLOW_API_KEY),
        "health": "/health",
        "predict": "/predict",
        "docs": "/docs",
    }

@app.head("/")
def root_head() -> dict[str, Any]:
    return {}

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_loaded": bool(ROBOFLOW_API_KEY),
        "model": ROBOFLOW_MODEL_ID,
        "runtime": "roboflow-serverless",
        "device": "remote",
        "confidence": CONFIDENCE,
        "iou": IOU,
        "max_file_size": MAX_FILE_SIZE,
        "max_image_dimension": MAX_IMAGE_DIMENSION,
        "max_detections": MAX_DETECTIONS,
        "focus_mode": FOCUS_MODE,
        "focus_crop_ratio": FOCUS_CROP_RATIO,
        "focus_max_detections": FOCUS_MAX_DETECTIONS,
        "roboflow_configured": bool(ROBOFLOW_API_KEY),
        "model_id": ROBOFLOW_MODEL_ID,
    }

def normalize_key(value: Any) -> str:
    return (
        unicodedata.normalize("NFD", str(value or ""))
        .strip().encode("ascii", "ignore").decode("ascii")
        .lower().replace("-", "_").replace(" ", "_")
    )

def prettify_class_name(value: Any) -> str:
    text = str(value or "").replace("_", " ").strip()
    return " ".join(part.capitalize() for part in text.split()) or "Objeto"

def resolve_rule(source_class: Any) -> tuple[str | None, dict[str, str] | None]:
    normalized = normalize_key(source_class)
    category_key = CLASS_MAP.get(normalized)
    if category_key is None and normalized in RULES:
        category_key = normalized
    return (category_key, RULES.get(category_key)) if category_key else (None, None)

def center_crop(image: np.ndarray, crop_ratio: float) -> tuple[np.ndarray, int, int]:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Imagem sem dimensões válidas.")
    side = max(32, min(int(round(min(h, w) * crop_ratio)), h, w))
    x0 = max(0, (w - side) // 2)
    y0 = max(0, (h - side) // 2)
    return image[y0:y0 + side, x0:x0 + side].copy(), x0, y0

def normalize_prediction(
    prediction: dict[str, Any], crop_offset_x: int, crop_offset_y: int,
    original_width: int, original_height: int,
) -> dict[str, Any] | None:
    source_class = prediction.get("class", prediction.get("label"))
    category_key, rule = resolve_rule(source_class)
    if not rule or not category_key:
        return None
    try:
        score = float(prediction.get("confidence", prediction.get("score", 0.0)))
        x = float(prediction.get("x", 0.0))
        y = float(prediction.get("y", 0.0))
        width = float(prediction.get("width", 0.0))
        height = float(prediction.get("height", 0.0))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(score) or score < CONFIDENCE or width <= 0 or height <= 0:
        return None

    x1 = max(0.0, min(x - width / 2.0 + crop_offset_x, float(original_width)))
    y1 = max(0.0, min(y - height / 2.0 + crop_offset_y, float(original_height)))
    width = min(width, float(original_width) - x1)
    height = min(height, float(original_height) - y1)
    if width <= 0 or height <= 0:
        return None

    return {
        "source_class": normalize_key(source_class),
        "class_name": prettify_class_name(source_class),
        "category": rule["category"],
        "category_key": category_key,
        "bin": rule["bin"],
        "destination": rule["destination"],
        "decomposition": rule["decomposition"],
        "fact": rule["fact"],
        "score": score,
        "bbox": [int(round(x1)), int(round(y1)), int(round(width)), int(round(height))],
    }

def focus_score(prediction: dict[str, Any], image_shape: tuple[int, int, int]) -> float:
    bbox = prediction.get("bbox") or []
    if len(bbox) < 4:
        return -1.0
    h, w = image_shape[:2]
    x, y, bw, bh = [float(v) for v in bbox[:4]]
    cx, cy = x + bw / 2.0, y + bh / 2.0
    tx, ty = w / 2.0, h / 2.0
    diag = max(1.0, (w * w + h * h) ** 0.5)
    distance = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
    center_bonus = max(0.0, 1.0 - min(0.35, (distance / diag) * 0.35))
    return float(prediction.get("score", 0.0)) * center_bonus

def normalize_roboflow_result(
    result: Any, crop_offset_x: int, crop_offset_y: int,
    original_width: int, original_height: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(result, dict):
        raise ValueError(f"Resposta inesperada do Roboflow: {type(result).__name__}")
    raw_predictions = result.get("predictions")
    if raw_predictions is None:
        raw_predictions = result.get("results", [])
    if not isinstance(raw_predictions, list):
        raw_predictions = []

    predictions: list[dict[str, Any]] = []
    for raw in raw_predictions:
        if isinstance(raw, dict):
            item = normalize_prediction(raw, crop_offset_x, crop_offset_y, original_width, original_height)
            if item is not None:
                predictions.append(item)

    predictions.sort(
        key=lambda item: (
            focus_score(item, (original_height, original_width, 3)) if FOCUS_MODE else float(item["score"]),
            float(item["score"]),
        ),
        reverse=True,
    )
    limit = FOCUS_MAX_DETECTIONS if FOCUS_MODE else MAX_DETECTIONS
    predictions = predictions[:limit]
    meta: dict[str, Any] = {
        "raw_prediction_count": len(raw_predictions),
        "accepted_prediction_count": len(predictions),
    }
    if DEBUG_PREDICTIONS:
        meta["raw_predictions"] = raw_predictions
    return predictions, meta

def run_roboflow_inference(image: np.ndarray) -> dict[str, Any]:
    if not ROBOFLOW_API_KEY:
        raise RuntimeError("ROBOFLOW_API_KEY não configurada no Render.")

    success, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not success:
        raise RuntimeError("Não foi possível codificar a imagem para JPEG.")

    image_base64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    url = f"{ROBOFLOW_API_URL}/{ROBOFLOW_MODEL_ID}"
    params = {
        "api_key": ROBOFLOW_API_KEY,
        "confidence": int(round(CONFIDENCE * 100)),
        "overlap": int(round(IOU * 100)),
    }

    print(f"[EcoScan] Endpoint: {url}", flush=True)
    try:
        response = requests.post(
            url,
            params=params,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=image_base64,
            timeout=ROBOFLOW_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise RuntimeError(f"Falha de conexão com Roboflow: {error}") from error

    print(f"[EcoScan] Roboflow HTTP: {response.status_code}", flush=True)
    if not response.ok:
        body = response.text[:4000]
        print(f"[EcoScan] Resposta de erro do Roboflow: {body}", flush=True)
        raise RuntimeError(f"Roboflow retornou HTTP {response.status_code}: {body}")

    try:
        result = response.json()
    except ValueError as error:
        raise RuntimeError(
            f"Roboflow retornou uma resposta que não é JSON: {response.text[:2000]}"
        ) from error
    if DEBUG_PREDICTIONS:
        print(f"[EcoScan] Resposta completa do Roboflow: {result}", flush=True)
    return result

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    request_id = id(file)
    request_start = time.perf_counter()
    raw: bytes | None = None
    frame: np.ndarray | None = None
    inference_frame: np.ndarray | None = None

    print("=" * 60, flush=True)
    print(f"[EcoScan][{request_id}] NOVA REQUISIÇÃO /predict", flush=True)
    print(f"[EcoScan][{request_id}] Modelo: {ROBOFLOW_MODEL_ID}", flush=True)

    try:
        if not ROBOFLOW_API_KEY:
            raise HTTPException(503, "ROBOFLOW_API_KEY não está configurada no Render.")
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(400, "Envie uma imagem.")

        raw = await file.read()
        if not raw:
            raise HTTPException(400, "Imagem vazia.")
        if len(raw) > MAX_FILE_SIZE:
            raise HTTPException(413, f"Imagem muito grande. Limite: {MAX_FILE_SIZE} bytes.")

        array = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        del array
        if frame is None:
            raise HTTPException(400, "Imagem inválida.")

        original_h, original_w = frame.shape[:2]
        if original_w > MAX_IMAGE_DIMENSION or original_h > MAX_IMAGE_DIMENSION:
            scale = min(MAX_IMAGE_DIMENSION / original_w, MAX_IMAGE_DIMENSION / original_h)
            new_w = max(1, int(round(original_w * scale)))
            new_h = max(1, int(round(original_h * scale)))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            original_h, original_w = frame.shape[:2]

        crop_offset_x = 0
        crop_offset_y = 0
        inference_frame = frame
        if FOCUS_MODE:
            inference_frame, crop_offset_x, crop_offset_y = center_crop(frame, FOCUS_CROP_RATIO)

        print(
            f"[EcoScan][{request_id}] Imagem enviada ao Roboflow: "
            f"{inference_frame.shape[1]}x{inference_frame.shape[0]}",
            flush=True,
        )

        if not INFERENCE_LOCK.acquire(timeout=25):
            raise HTTPException(503, "Detector ocupado. Tente novamente em alguns segundos.")

        try:
            inference_start = time.perf_counter()
            result = await asyncio.wait_for(
                asyncio.to_thread(run_roboflow_inference, inference_frame),
                timeout=ROBOFLOW_TIMEOUT_SECONDS,
            )
            inference_time = time.perf_counter() - inference_start
        finally:
            INFERENCE_LOCK.release()

        predictions, result_meta = normalize_roboflow_result(
            result,
            crop_offset_x,
            crop_offset_y,
            original_w,
            original_h,
        )
        total_time = time.perf_counter() - request_start

        print(f"[EcoScan][{request_id}] Detecções aceitas: {len(predictions)}", flush=True)
        print(f"[EcoScan][{request_id}] Inferência Roboflow: {inference_time:.3f}s", flush=True)
        print(f"[EcoScan][{request_id}] Tempo total: {total_time:.3f}s", flush=True)
        print("=" * 60, flush=True)

        return {
            "predictions": predictions,
            "model": ROBOFLOW_MODEL_ID,
            "runtime": "roboflow-serverless",
            "image": {"width": original_w, "height": original_h},
            "focus": {
                "enabled": FOCUS_MODE,
                "crop_ratio": FOCUS_CROP_RATIO,
                "max_detections": FOCUS_MAX_DETECTIONS,
            },
            "performance": {
                "inference_time_seconds": round(inference_time, 3),
                "total_time_seconds": round(total_time, 3),
            },
            "meta": result_meta,
        }

    except HTTPException:
        raise
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, "O Roboflow demorou mais que o limite configurado. Tente novamente.") from exc
    except MemoryError as exc:
        raise HTTPException(503, "Memória insuficiente para processar esta imagem.") from exc
    except Exception as exc:
        print(
            f"[EcoScan][{request_id}] ERRO: {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise HTTPException(
            502,
            f"Falha ao executar inferência no Roboflow: {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        raw = None
        frame = None
        inference_frame = None
        gc.collect()
        try:
            await file.close()
        except Exception:
            pass
