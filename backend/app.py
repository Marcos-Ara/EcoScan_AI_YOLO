import asyncio
import gc
import os
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from inference_sdk import InferenceHTTPClient

# ============================================================
# EcoScan AI - Runtime usando Roboflow YOLO11
# ============================================================
# O modelo agora é executado no Roboflow Serverless.
# O Render fica responsável somente por:
#   - FastAPI
#   - OpenCV
#   - envio da imagem ao Roboflow
#   - normalização da resposta
#   - modo de foco / melhor detecção
#
# Isso elimina PyTorch, YOLOv7, model.pt e ONNX do runtime.
# ============================================================

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

ROOT = Path(__file__).resolve().parent

# ============================================================
# CONFIGURAÇÃO
# ============================================================

ROBOFLOW_API_URL = os.getenv(
    "ROBOFLOW_API_URL",
    "https://serverless.roboflow.com",
).rstrip("/")

ROBOFLOW_API_KEY = os.getenv(
    "ROBOFLOW_API_KEY",
    "",
).strip()

ROBOFLOW_MODEL_ID = os.getenv(
    "ROBOFLOW_MODEL_ID",
    "waste-sorting-smyr8/2",
).strip()

CONFIDENCE = float(
    os.getenv("CONFIDENCE", "0.40")
)

IOU = float(
    os.getenv("IOU", "0.45")
)

MAX_FILE_SIZE = int(
    os.getenv("MAX_FILE_SIZE", "1500000")
)

MAX_IMAGE_DIMENSION = int(
    os.getenv("MAX_IMAGE_DIMENSION", "960")
)

MAX_DETECTIONS = max(
    1,
    min(
        20,
        int(os.getenv("MAX_DETECTIONS", "1")),
    ),
)

FOCUS_MODE = os.getenv(
    "FOCUS_MODE",
    "true",
).lower() in {
    "1",
    "true",
    "yes",
    "on",
}

FOCUS_CROP_RATIO = float(
    os.getenv("FOCUS_CROP_RATIO", "0.90")
)

FOCUS_CROP_RATIO = min(
    1.0,
    max(0.55, FOCUS_CROP_RATIO),
)

FOCUS_MAX_DETECTIONS = max(
    1,
    min(
        3,
        int(os.getenv("FOCUS_MAX_DETECTIONS", "1")),
    ),
)

DEBUG_PREDICTIONS = os.getenv(
    "DEBUG_PREDICTIONS",
    "false",
).lower() in {
    "1",
    "true",
    "yes",
    "on",
}

ROBOFLOW_TIMEOUT_SECONDS = float(
    os.getenv("ROBOFLOW_TIMEOUT_SECONDS", "30")
)

# Uma inferência por vez. Isso ajuda o plano Free do Render.
INFERENCE_LOCK = threading.Lock()

# ============================================================
# CLASSES DO MODELO -> REGRAS DO ECOSCAN
# ============================================================

# O modelo escolhido trabalha com:
# paper, plastic, glass, metal, cardboard.
# As aliases abaixo tornam a integração tolerante a variações
# de capitalização/nome retornadas pelo Roboflow.

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
        "category": "Papel",
        "bin": "🔵 Azul",
        "destination": "Reciclagem",
        "decomposition": "3–6 meses",
        "fact": (
            "Papel e papelão devem estar, de preferência, secos "
            "e sem restos de comida."
        ),
    },
    "plastico": {
        "category": "Plástico",
        "bin": "🔴 Vermelha",
        "destination": "Reciclagem",
        "decomposition": "Varia por material",
        "fact": (
            "Garrafas PET, embalagens e outros plásticos devem ir "
            "para a coleta seletiva."
        ),
    },
    "vidro": {
        "category": "Vidro",
        "bin": "🟢 Verde",
        "destination": "Reciclagem",
        "decomposition": "Muito longo",
        "fact": "Vidro deve ser encaminhado para a coleta seletiva.",
    },
    "metal": {
        "category": "Metal",
        "bin": "🟡 Amarela",
        "destination": "Reciclagem",
        "decomposition": "Varia por material",
        "fact": (
            "Latas, tampas e outros metais devem ser encaminhados "
            "para reciclagem."
        ),
    },
    "organico": {
        "category": "Orgânico",
        "bin": "🟤 Marrom",
        "destination": "Compostagem",
        "decomposition": "Varia",
        "fact": (
            "Restos de alimentos e cascas podem ser destinados à compostagem."
        ),
    },
    "rejeito": {
        "category": "Rejeito",
        "bin": "⚫ Cinza/Preta",
        "destination": "Rejeitos",
        "decomposition": "Varia",
        "fact": (
            "Resíduos que não podem ser reciclados devem ser destinados "
            "aos rejeitos."
        ),
    },
}

# ============================================================
# ROBOfLOW CLIENT
# ============================================================

roboflow_client: InferenceHTTPClient | None = None

if ROBOFLOW_API_KEY:
    roboflow_client = InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL,
        api_key=ROBOFLOW_API_KEY,
    )
else:
    print(
        "[EcoScan] AVISO: ROBOFLOW_API_KEY não configurada.",
        flush=True,
    )

print("=" * 60, flush=True)
print("[EcoScan] Inicializando runtime Roboflow...", flush=True)
print(f"[EcoScan] API: {ROBOFLOW_API_URL}", flush=True)
print(f"[EcoScan] Modelo: {ROBOFLOW_MODEL_ID}", flush=True)
print(f"[EcoScan] Confidence: {CONFIDENCE}", flush=True)
print(f"[EcoScan] IoU: {IOU}", flush=True)
print(f"[EcoScan] Max file size: {MAX_FILE_SIZE} bytes", flush=True)
print(f"[EcoScan] Max image dimension: {MAX_IMAGE_DIMENSION}px", flush=True)
print(f"[EcoScan] Focus mode: {FOCUS_MODE}", flush=True)
print(f"[EcoScan] Focus crop: {FOCUS_CROP_RATIO:.2f}", flush=True)
print(f"[EcoScan] Focus max detections: {FOCUS_MAX_DETECTIONS}", flush=True)
print(
    f"[EcoScan] Roboflow API key: {'configured' if ROBOFLOW_API_KEY else 'missing'}",
    flush=True,
)
print("=" * 60, flush=True)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="EcoScan AI YOLO API",
    version="3.0.0",
    description=(
        "API do EcoScan usando o modelo waste-sorting-smyr8/2 "
        "hospedado pelo Roboflow."
    ),
)

allowed_origin = os.getenv(
    "ALLOWED_ORIGIN",
    "https://marcos-ara.github.io",
)

origins = [
    item.strip()
    for item in allowed_origin.split(",")
    if item.strip()
]

if not origins:
    origins = ["*"]

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
        "version": "3.0.0",
        "roboflow_configured": bool(ROBOFLOW_API_KEY),
        "health": "/health",
        "predict": "/predict",
        "docs": "/docs",
    }


@app.head("/")
def root_head():
    # O Render envia HEAD / durante o health/probe.
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


# ============================================================
# HELPERS
# ============================================================


def normalize_key(value: Any) -> str:
    return (
        unicodedata.normalize("NFD", str(value or ""))
        .strip()
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def prettify_class_name(value: Any) -> str:
    text = str(value or "").replace("_", " ").strip()
    return " ".join(
        part.capitalize()
        for part in text.split()
    ) or "Objeto"


def resolve_rule(source_class: Any) -> tuple[str | None, dict[str, str] | None]:
    normalized = normalize_key(source_class)
    category_key = CLASS_MAP.get(normalized)

    if category_key is None and normalized in RULES:
        category_key = normalized

    if category_key is None:
        return None, None

    return category_key, RULES.get(category_key)


def center_crop(
    image: np.ndarray,
    crop_ratio: float,
) -> tuple[np.ndarray, int, int]:
    h, w = image.shape[:2]

    if h <= 0 or w <= 0:
        raise ValueError("Imagem sem dimensões válidas.")

    side = int(round(min(h, w) * crop_ratio))
    side = max(32, min(side, h, w))

    x0 = max(0, (w - side) // 2)
    y0 = max(0, (h - side) // 2)

    crop = image[
        y0:y0 + side,
        x0:x0 + side,
    ].copy()

    return crop, x0, y0


def focus_score(prediction: dict[str, Any], image_shape: tuple[int, int, int]) -> float:
    bbox = prediction.get("bbox") or []
    if len(bbox) < 4:
        return -1.0

    h, w = image_shape[:2]
    x, y, bw, bh = [float(v) for v in bbox[:4]]

    center_x = x + bw / 2.0
    center_y = y + bh / 2.0

    target_x = w / 2.0
    target_y = h / 2.0
    diag = max(1.0, (w * w + h * h) ** 0.5)

    distance = (
        (center_x - target_x) ** 2
        + (center_y - target_y) ** 2
    ) ** 0.5

    center_bonus = max(
        0.0,
        1.0 - min(0.35, (distance / diag) * 0.35),
    )

    score = float(prediction.get("score", 0.0))
    return score * center_bonus


def normalize_prediction(
    prediction: dict[str, Any],
    crop_offset_x: int,
    crop_offset_y: int,
    original_width: int,
    original_height: int,
) -> dict[str, Any] | None:
    source_class = prediction.get("class")
    if source_class is None:
        source_class = prediction.get("label")

    category_key, rule = resolve_rule(source_class)
    if rule is None or category_key is None:
        return None

    try:
        score = float(
            prediction.get(
                "confidence",
                prediction.get("score", 0.0),
            )
        )
    except (TypeError, ValueError):
        return None

    if not np.isfinite(score) or score < CONFIDENCE:
        return None

    try:
        x = float(prediction.get("x", 0.0))
        y = float(prediction.get("y", 0.0))
        width = float(prediction.get("width", 0.0))
        height = float(prediction.get("height", 0.0))
    except (TypeError, ValueError):
        return None

    if width <= 0 or height <= 0:
        return None

    x1 = x - width / 2.0 + crop_offset_x
    y1 = y - height / 2.0 + crop_offset_y

    x1 = max(0.0, min(x1, float(original_width)))
    y1 = max(0.0, min(y1, float(original_height)))

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
        "bbox": [
            int(round(x1)),
            int(round(y1)),
            int(round(width)),
            int(round(height)),
        ],
    }


def normalize_roboflow_result(
    result: Any,
    crop_offset_x: int,
    crop_offset_y: int,
    original_width: int,
    original_height: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(result, dict):
        raise ValueError(
            f"Resposta inesperada do Roboflow: {type(result).__name__}"
        )

    raw_predictions = result.get("predictions")

    if raw_predictions is None:
        # Alguns formatos podem usar uma chave diferente.
        raw_predictions = result.get("results", [])

    if not isinstance(raw_predictions, list):
        raw_predictions = []

    predictions: list[dict[str, Any]] = []

    for raw_prediction in raw_predictions:
        if not isinstance(raw_prediction, dict):
            continue

        normalized = normalize_prediction(
            raw_prediction,
            crop_offset_x,
            crop_offset_y,
            original_width,
            original_height,
        )

        if normalized is not None:
            predictions.append(normalized)

    # Ordena por confiança e, em foco, privilegia o objeto central.
    predictions.sort(
        key=lambda item: (
            focus_score(item, (original_height, original_width, 3))
            if FOCUS_MODE else float(item["score"]),
            float(item["score"]),
        ),
        reverse=True,
    )

    limit = (
        FOCUS_MAX_DETECTIONS
        if FOCUS_MODE
        else MAX_DETECTIONS
    )

    predictions = predictions[:limit]

    meta = {
        "raw_prediction_count": len(raw_predictions),
        "accepted_prediction_count": len(predictions),
    }

    if DEBUG_PREDICTIONS:
        meta["raw_predictions"] = raw_predictions

    return predictions, meta


def run_roboflow_inference(image: np.ndarray) -> Any:
    if roboflow_client is None:
        raise RuntimeError(
            "ROBOFLOW_API_KEY não configurada no Render."
        )

    # O SDK oficial aceita NumPy arrays diretamente.
    return roboflow_client.infer(
        image,
        model_id=ROBOFLOW_MODEL_ID,
    )


# ============================================================
# PREDICT
# ============================================================


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    request_id = id(file)
    request_start = time.perf_counter()

    raw: bytes | None = None
    frame: np.ndarray | None = None
    inference_frame: np.ndarray | None = None

    print("=" * 60, flush=True)
    print(
        f"[EcoScan][{request_id}] NOVA REQUISIÇÃO /predict",
        flush=True,
    )
    print(
        f"[EcoScan][{request_id}] Modelo: {ROBOFLOW_MODEL_ID}",
        flush=True,
    )

    try:
        if roboflow_client is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "ROBOFLOW_API_KEY não está configurada no Render. "
                    "Configure a variável de ambiente e faça um novo deploy."
                ),
            )

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Envie uma imagem.",
            )

        raw = await file.read()

        if not raw:
            raise HTTPException(
                status_code=400,
                detail="Imagem vazia.",
            )

        if len(raw) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Imagem muito grande. "
                    f"Limite: {MAX_FILE_SIZE} bytes."
                ),
            )

        array = np.frombuffer(
            raw,
            dtype=np.uint8,
        )

        frame = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR,
        )

        del array

        if frame is None:
            raise HTTPException(
                status_code=400,
                detail="Imagem inválida.",
            )

        original_h, original_w = frame.shape[:2]

        if original_w > MAX_IMAGE_DIMENSION or original_h > MAX_IMAGE_DIMENSION:
            scale = min(
                MAX_IMAGE_DIMENSION / original_w,
                MAX_IMAGE_DIMENSION / original_h,
            )

            new_w = max(1, int(round(original_w * scale)))
            new_h = max(1, int(round(original_h * scale)))

            frame = cv2.resize(
                frame,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA,
            )

        original_h, original_w = frame.shape[:2]

        # ----------------------------------------------------
        # FOCO EM UM OBJETO
        # ----------------------------------------------------
        crop_offset_x = 0
        crop_offset_y = 0
        inference_frame = frame

        if FOCUS_MODE:
            inference_frame, crop_offset_x, crop_offset_y = center_crop(
                frame,
                FOCUS_CROP_RATIO,
            )

        print(
            f"[EcoScan][{request_id}] "
            f"Imagem enviada ao Roboflow: "
            f"{inference_frame.shape[1]}x{inference_frame.shape[0]}",
            flush=True,
        )

        # ----------------------------------------------------
        # UMA INFERÊNCIA POR VEZ
        # ----------------------------------------------------
        acquired = INFERENCE_LOCK.acquire(timeout=25)
        if not acquired:
            raise HTTPException(
                status_code=503,
                detail="Detector ocupado. Tente novamente em alguns segundos.",
            )

        try:
            inference_start = time.perf_counter()

            result = await asyncio.wait_for(
                asyncio.to_thread(
                    run_roboflow_inference,
                    inference_frame,
                ),
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

        if DEBUG_PREDICTIONS:
            print(
                f"[EcoScan][{request_id}] Roboflow result: {result}",
                flush=True,
            )

        print(
            f"[EcoScan][{request_id}] "
            f"Detecções aceitas: {len(predictions)}",
            flush=True,
        )
        print(
            f"[EcoScan][{request_id}] "
            f"Inferência Roboflow: {inference_time:.3f}s",
            flush=True,
        )
        print(
            f"[EcoScan][{request_id}] "
            f"Tempo total: {total_time:.3f}s",
            flush=True,
        )
        print("=" * 60, flush=True)

        return {
            "predictions": predictions,
            "model": ROBOFLOW_MODEL_ID,
            "runtime": "roboflow-serverless",
            "image": {
                "width": original_w,
                "height": original_h,
            },
            "focus": {
                "enabled": FOCUS_MODE,
                "crop_ratio": FOCUS_CROP_RATIO,
                "max_detections": FOCUS_MAX_DETECTIONS,
            },
            "performance": {
                "inference_time_seconds": round(
                    inference_time,
                    3,
                ),
                "total_time_seconds": round(
                    total_time,
                    3,
                ),
            },
            "meta": result_meta,
        }

    except HTTPException:
        raise

    except asyncio.TimeoutError as exc:
        print(
            f"[EcoScan][{request_id}] Timeout Roboflow.",
            flush=True,
        )
        raise HTTPException(
            status_code=504,
            detail=(
                "O Roboflow demorou mais que o limite configurado. "
                "Tente novamente."
            ),
        ) from exc

    except MemoryError as exc:
        print(
            f"[EcoScan][{request_id}] MEMORY ERROR.",
            flush=True,
        )
        raise HTTPException(
            status_code=503,
            detail="Memória insuficiente para processar esta imagem.",
        ) from exc

    except Exception as exc:
        print(
            f"[EcoScan][{request_id}] "
            f"ERRO: {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "Não foi possível executar a inferência no Roboflow. "
                f"{type(exc).__name__}"
            ),
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
