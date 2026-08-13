import os
import gc
import time
import threading
from pathlib import Path

# ============================================================
# EcoScan AI - Runtime leve para Render Free
# ============================================================
# Esta versão NÃO carrega PyTorch/YOLOv7 em produção.
# O modelo é convertido para ONNX durante o build e o runtime
# utiliza apenas ONNX Runtime + OpenCV.
#
# Objetivo: reduzir drasticamente o consumo de RAM do serviço
# Free (512 MiB) e evitar OOM durante o startup.
# ============================================================

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np
import onnxruntime as ort

cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.onnx"

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Modelo ONNX não encontrado: {MODEL_PATH}. "
        "O Dockerfile deve convertê-lo durante o build."
    )

# ============================================================
# CONFIGURAÇÃO
# ============================================================

CONFIDENCE = float(os.getenv("CONFIDENCE", "0.35"))
IOU = float(os.getenv("IOU", "0.45"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "1500000"))
MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "960"))
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "20"))

# Modo de foco: o modelo recebe apenas uma região central da imagem.
# Isso reduz falsos positivos do fundo e força o uso em um único objeto.
FOCUS_MODE = os.getenv("FOCUS_MODE", "true").lower() in {
    "1", "true", "yes", "on"
}
FOCUS_CROP_RATIO = float(
    os.getenv("FOCUS_CROP_RATIO", "0.90")
)
FOCUS_CROP_RATIO = min(
    1.0,
    max(0.55, FOCUS_CROP_RATIO)
)

# Em modo de foco, retornamos apenas a melhor detecção.
FOCUS_MAX_DETECTIONS = int(
    os.getenv("FOCUS_MAX_DETECTIONS", "1")
)
FOCUS_MAX_DETECTIONS = max(
    1,
    min(3, FOCUS_MAX_DETECTIONS)
)

DEBUG_PREDICTIONS = os.getenv(
    "DEBUG_PREDICTIONS", "false"
).lower() in {
    "1", "true", "yes", "on"
}

# Uma única inferência por vez.
INFERENCE_LOCK = threading.Lock()

CLASS_MAP = {
    "cardboard": "papel",
    "metal": "metal",
    "rigid_plastic": "plastico",
    "soft_plastic": "plastico",
}

RULES = {
    "papel": {
        "category": "Papel",
        "bin": "🔵 Azul",
        "destination": "Reciclagem",
        "decomposition": "3–6 meses",
    },
    "plastico": {
        "category": "Plástico",
        "bin": "🔴 Vermelha",
        "destination": "Reciclagem",
        "decomposition": "Varia por material",
    },
    "metal": {
        "category": "Metal",
        "bin": "🟡 Amarela",
        "destination": "Reciclagem",
        "decomposition": "Varia por material",
    },
    "vidro": {
        "category": "Vidro",
        "bin": "🟢 Verde",
        "destination": "Reciclagem",
        "decomposition": "Muito longo",
    },
    "organico": {
        "category": "Orgânico",
        "bin": "🟤 Marrom",
        "destination": "Compostagem",
        "decomposition": "Varia",
    },
    "rejeito": {
        "category": "Rejeito",
        "bin": "⚫ Cinza/Preta",
        "destination": "Rejeitos",
        "decomposition": "Varia",
    },
}

# Ordem oficial do GreenSorter.
MODEL_NAMES = [
    "cardboard",
    "metal",
    "rigid_plastic",
    "soft_plastic",
]

# ============================================================
# ONNX RUNTIME
# ============================================================

print("=" * 60, flush=True)
print("[EcoScan] Inicializando runtime ONNX...", flush=True)
print(f"[EcoScan] Modelo: {MODEL_PATH}", flush=True)
print(f"[EcoScan] Tamanho do modelo: {MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB", flush=True)
print(f"[EcoScan] Image size: {IMG_SIZE}", flush=True)
print(f"[EcoScan] Confidence: {CONFIDENCE}", flush=True)
print(f"[EcoScan] IoU: {IOU}", flush=True)
print(f"[EcoScan] Max file size: {MAX_FILE_SIZE} bytes", flush=True)
print(f"[EcoScan] Max image dimension: {MAX_IMAGE_DIMENSION}px", flush=True)
print(f"[EcoScan] Focus mode: {FOCUS_MODE} ({FOCUS_CROP_RATIO:.2f})", flush=True)
print(f"[EcoScan] Focus max detections: {FOCUS_MAX_DETECTIONS}", flush=True)

session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
session_options.enable_mem_pattern = False
session_options.enable_cpu_mem_arena = False

load_start = time.perf_counter()

try:
    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
except Exception as exc:
    print(f"[EcoScan] ERRO AO CARREGAR ONNX: {exc!r}", flush=True)
    raise

load_time = time.perf_counter() - load_start

inputs = session.get_inputs()
outputs = session.get_outputs()

if not inputs:
    raise RuntimeError("Modelo ONNX não possui entrada.")

if not outputs:
    raise RuntimeError("Modelo ONNX não possui saída.")

input_meta = inputs[0]
INPUT_NAME = input_meta.name
OUTPUT_NAMES = [item.name for item in outputs]

output_shape = outputs[0].shape
if len(output_shape) == 3:
    output_values = output_shape[-1]
    if isinstance(output_values, int) and output_values != 9:
        raise RuntimeError(
            "Saída ONNX incompatível com o GreenSorter de 4 classes: "
            f"shape={output_shape}. Esperado [1,N,9]."
        )

# O export é estático em 224x224. Se alguém configurar outro tamanho
# no ambiente, é melhor falhar no startup do que enviar um tensor que
# o ONNX não aceita.
input_shape = input_meta.shape
if len(input_shape) == 4:
    static_h = input_shape[2]
    static_w = input_shape[3]
    if isinstance(static_h, int) and isinstance(static_w, int):
        if static_h != IMG_SIZE or static_w != IMG_SIZE:
            raise RuntimeError(
                "IMG_SIZE incompatível com o modelo ONNX: "
                f"ambiente={IMG_SIZE}, modelo={static_h}x{static_w}. "
                "Use o mesmo tamanho usado na conversão."
            )

print(f"[EcoScan] ONNX carregado em {load_time:.2f}s", flush=True)
print(f"[EcoScan] Input: {INPUT_NAME} shape={input_meta.shape}", flush=True)
print(f"[EcoScan] Outputs: {OUTPUT_NAMES}", flush=True)
print("[EcoScan] Runtime: CPU / 1 thread", flush=True)
print("=" * 60, flush=True)

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="EcoScan AI YOLO API",
    version="2.0.0",
    description=(
        "API leve de detecção de resíduos usando "
        "GreenSorter exportado para ONNX."
    ),
)

allowed_origin = os.getenv("ALLOWED_ORIGIN", "*")
origins = [item.strip() for item in allowed_origin.split(",") if item.strip()]
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
def root():
    return {
        "service": "EcoScan AI YOLO API",
        "status": "online",
        "model": "GreenSorter ONNX",
        "version": "2.0.0",
        "health": "/health",
        "predict": "/predict",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": True,
        "model": "GreenSorter ONNX",
        "runtime": "onnxruntime",
        "device": "cpu",
        "classes": MODEL_NAMES,
        "confidence": CONFIDENCE,
        "iou": IOU,
        "img_size": IMG_SIZE,
        "max_file_size": MAX_FILE_SIZE,
        "max_image_dimension": MAX_IMAGE_DIMENSION,
        "inference_lock": "enabled",
        "focus_mode": FOCUS_MODE,
        "focus_crop_ratio": FOCUS_CROP_RATIO,
        "focus_max_detections": FOCUS_MAX_DETECTIONS,
        "debug_predictions": DEBUG_PREDICTIONS,
    }


# ============================================================
# MODO DE FOCO
# ============================================================

def center_crop(
    image: np.ndarray,
    crop_ratio: float,
):
    """
    Recorta uma região central quadrada.
    Retorna:
      crop, offset_x, offset_y
    """
    h, w = image.shape[:2]

    if h <= 0 or w <= 0:
        raise ValueError("Imagem sem dimensões válidas.")

    side = int(
        round(
            min(h, w) * crop_ratio
        )
    )

    side = max(
        32,
        min(side, h, w)
    )

    x0 = max(
        0,
        (w - side) // 2
    )

    y0 = max(
        0,
        (h - side) // 2
    )

    x1 = min(
        w,
        x0 + side
    )

    y1 = min(
        h,
        y0 + side
    )

    crop = image[
        y0:y1,
        x0:x1
    ].copy()

    return crop, x0, y0


def select_focus_predictions(
    predictions,
    image_shape,
):
    """
    Seleciona somente as melhores detecções em modo de foco.

    Dá pequeno peso à proximidade do centro para evitar que
    uma detecção menor nas bordas do recorte seja escolhida
    no lugar do objeto central.
    """
    if not predictions:
        return []

    h, w = image_shape[:2]

    cx_target = w / 2.0
    cy_target = h / 2.0

    diag = max(
        1.0,
        (w * w + h * h) ** 0.5
    )

    ranked = []

    for item in predictions:
        bbox = item.get("bbox")

        if not bbox or len(bbox) < 4:
            continue

        x, y, bw, bh = [
            float(value)
            for value in bbox[:4]
        ]

        center_x = x + bw / 2.0
        center_y = y + bh / 2.0

        distance = (
            (
                (center_x - cx_target) ** 2
                + (center_y - cy_target) ** 2
            ) ** 0.5
        ) / diag

        score = float(
            item.get("score", 0.0)
        )

        # Mantém a confiança dominante e usa a posição
        # apenas como desempate suave.
        focus_score = (
            score
            * (
                1.0
                - min(
                    0.30,
                    distance * 0.30
                )
            )
        )

        ranked.append(
            (
                focus_score,
                score,
                item
            )
        )

    ranked.sort(
        key=lambda value: (
            value[0],
            value[1]
        ),
        reverse=True
    )

    return [
        item
        for _, _, item
        in ranked[
            :FOCUS_MAX_DETECTIONS
        ]
    ]


# ============================================================
# PREPROCESSAMENTO
# ============================================================

def letterbox(
    image: np.ndarray,
    new_size: int,
):
    """Redimensiona mantendo proporção e completa com cinza 114."""

    h, w = image.shape[:2]

    if h <= 0 or w <= 0:
        raise ValueError("Imagem sem dimensões válidas.")

    ratio = min(new_size / w, new_size / h)

    new_w = max(1, int(round(w * ratio)))
    new_h = max(1, int(round(h * ratio)))

    if (new_w, new_h) != (w, h):
        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = image

    pad_w = new_size - new_w
    pad_h = new_size - new_h

    left = pad_w // 2
    top = pad_h // 2

    output = np.full(
        (new_size, new_size, 3),
        114,
        dtype=np.uint8,
    )

    output[
        top:top + new_h,
        left:left + new_w
    ] = resized

    return output, ratio, left, top


def prepare_input(frame: np.ndarray):
    image, ratio, pad_x, pad_y = letterbox(
        frame,
        IMG_SIZE,
    )

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    # float32 é o formato esperado pelo modelo exportado.
    tensor = rgb.transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)
    tensor /= 255.0
    tensor = np.expand_dims(tensor, axis=0)

    return tensor, ratio, pad_x, pad_y


# ============================================================
# PÓS-PROCESSAMENTO YOLO
# ============================================================

def _normalize_output(output):
    """
    Aceita os formatos mais comuns do export YOLOv7:
      [1, N, 5+C]
      [N, 5+C]
      [1, 5+C, N]
    """
    arr = np.asarray(output)

    # Remove dimensões unitárias extras.
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(
            f"Formato de saída ONNX inesperado: {arr.shape}"
        )

    # Para GreenSorter são 9 valores:
    # x, y, w, h, objectness + 4 classes.
    expected = 5 + len(MODEL_NAMES)

    if arr.shape[1] == expected:
        return arr

    if arr.shape[0] == expected:
        return arr.T

    # Alguns exports end-to-end podem entregar 6 valores:
    # x1,y1,x2,y2,score,class.
    if arr.shape[1] == 6:
        return arr

    if arr.shape[0] == 6:
        return arr.T

    raise ValueError(
        f"Saída ONNX incompatível: {arr.shape}. "
        f"Esperado [N,{expected}] ou [N,6]."
    )


def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(
        0.0,
        (box[2] - box[0]) * (box[3] - box[1]),
    )

    area_b = np.maximum(
        0.0,
        (boxes[:, 2] - boxes[:, 0])
        * (boxes[:, 3] - boxes[:, 1]),
    )

    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-9)


def nms_class_aware(boxes, scores, class_ids, iou_threshold):
    """NMS simples e controlado para manter baixo o uso de RAM."""

    if not len(boxes):
        return []

    keep = []

    for class_id in np.unique(class_ids):
        indices = np.where(class_ids == class_id)[0]
        order = indices[
            np.argsort(scores[indices])[::-1]
        ]

        while order.size:
            current = int(order[0])
            keep.append(current)

            if order.size == 1:
                break

            rest = order[1:]
            ious = compute_iou(
                boxes[current],
                boxes[rest],
            )

            order = rest[
                ious <= iou_threshold
            ]

    keep.sort(
        key=lambda index: float(scores[index]),
        reverse=True,
    )

    return keep[:MAX_DETECTIONS]


def decode_predictions(
    output,
    original_shape,
    ratio,
    pad_x,
    pad_y,
):
    raw = _normalize_output(output)

    if raw.shape[1] == 6:
        # End-to-end: x1,y1,x2,y2,score,class
        boxes = raw[:, :4].astype(np.float32, copy=False)
        scores = raw[:, 4].astype(np.float32, copy=False)
        class_ids = raw[:, 5].astype(np.int32, copy=False)

        valid = (
            np.isfinite(scores)
            & (scores >= CONFIDENCE)
            & (class_ids >= 0)
            & (class_ids < len(MODEL_NAMES))
        )

        boxes = boxes[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]

    else:
        # YOLOv7 padrão:
        # x, y, w, h, objectness, class scores...
        cx = raw[:, 0]
        cy = raw[:, 1]
        bw = raw[:, 2]
        bh = raw[:, 3]
        objectness = raw[:, 4]

        class_scores = raw[:, 5:]

        class_ids = np.argmax(
            class_scores,
            axis=1,
        ).astype(np.int32)

        class_conf = class_scores[
            np.arange(len(class_scores)),
            class_ids,
        ]

        scores = objectness * class_conf

        valid = (
            np.isfinite(scores)
            & (scores >= CONFIDENCE)
            & (objectness > 0)
        )

        cx = cx[valid]
        cy = cy[valid]
        bw = bw[valid]
        bh = bh[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]

        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0

        boxes = np.column_stack(
            (x1, y1, x2, y2)
        ).astype(np.float32, copy=False)

    if not len(boxes):
        return []

    # Coordenadas do modelo -> imagem original.
    boxes[:, [0, 2]] = (
        boxes[:, [0, 2]] - pad_x
    ) / ratio

    boxes[:, [1, 3]] = (
        boxes[:, [1, 3]] - pad_y
    ) / ratio

    height, width = original_shape[:2]

    boxes[:, [0, 2]] = np.clip(
        boxes[:, [0, 2]],
        0,
        width,
    )

    boxes[:, [1, 3]] = np.clip(
        boxes[:, [1, 3]],
        0,
        height,
    )

    valid_size = (
        (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )

    boxes = boxes[valid_size]
    scores = scores[valid_size]
    class_ids = class_ids[valid_size]

    if not len(boxes):
        return []

    keep = nms_class_aware(
        boxes,
        scores,
        class_ids,
        IOU,
    )

    results = []

    for index in keep:
        class_id = int(class_ids[index])

        source_class = MODEL_NAMES[class_id]
        category_key = CLASS_MAP.get(source_class)

        if category_key is None:
            continue

        x1, y1, x2, y2 = boxes[index].tolist()

        rule = RULES[category_key]

        results.append({
            "source_class": source_class,
            "category": rule["category"],
            "category_key": category_key,
            "bin": rule["bin"],
            "destination": rule["destination"],
            "decomposition": rule["decomposition"],
            "score": float(scores[index]),
            "bbox": [
                int(round(x1)),
                int(round(y1)),
                max(0, int(round(x2 - x1))),
                max(0, int(round(y2 - y1))),
            ],
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:MAX_DETECTIONS]


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    request_id = id(file)
    request_start = time.perf_counter()

    raw = None
    frame = None
    input_tensor = None
    output = None

    print("=" * 60, flush=True)
    print(
        f"[EcoScan][{request_id}] NOVA REQUISIÇÃO /predict",
        flush=True,
    )
    print(
        f"[EcoScan][{request_id}] Filename: {file.filename}",
        flush=True,
    )
    print(
        f"[EcoScan][{request_id}] Content-Type: {file.content_type}",
        flush=True,
    )

    try:
        if (
            not file.content_type
            or not file.content_type.startswith("image/")
        ):
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

        print(
            f"[EcoScan][{request_id}] "
            f"Arquivo recebido: {len(raw)} bytes",
            flush=True,
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

        print(
            f"[EcoScan][{request_id}] "
            f"Resolução original: "
            f"{original_w}x{original_h}",
            flush=True,
        )

        if (
            original_w > MAX_IMAGE_DIMENSION
            or original_h > MAX_IMAGE_DIMENSION
        ):
            scale = min(
                MAX_IMAGE_DIMENSION / original_w,
                MAX_IMAGE_DIMENSION / original_h,
            )

            new_w = max(
                1,
                int(original_w * scale),
            )
            new_h = max(
                1,
                int(original_h * scale),
            )

            frame = cv2.resize(
                frame,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA,
            )

            print(
                f"[EcoScan][{request_id}] "
                f"Redimensionada para {new_w}x{new_h}",
                flush=True,
            )

        # ----------------------------------------------------
        # MODO DE FOCO
        # ----------------------------------------------------
        # Por padrão usamos somente a região central.
        # Isso reduz a influência de objetos/background nas bordas
        # e funciona melhor para o uso "um objeto por vez".
        inference_frame = frame
        focus_offset_x = 0
        focus_offset_y = 0

        if FOCUS_MODE:
            (
                inference_frame,
                focus_offset_x,
                focus_offset_y,
            ) = center_crop(
                frame,
                FOCUS_CROP_RATIO,
            )

            print(
                f"[EcoScan][{request_id}] "
                f"FOCUS crop: "
                f"{inference_frame.shape[1]}x"
                f"{inference_frame.shape[0]} "
                f"offset=({focus_offset_x},{focus_offset_y})",
                flush=True,
            )

        input_tensor, ratio, pad_x, pad_y = prepare_input(
            inference_frame
        )

        print(
            f"[EcoScan][{request_id}] "
            f"Tensor: {input_tensor.shape}",
            flush=True,
        )

        lock_start = time.perf_counter()

        # Timeout para evitar uma requisição ficar presa
        # indefinidamente esperando o detector.
        acquired = INFERENCE_LOCK.acquire(
            timeout=25
        )

        if not acquired:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Detector ocupado. "
                    "Tente novamente em alguns segundos."
                ),
            )

        lock_wait = time.perf_counter() - lock_start

        try:
            inference_start = time.perf_counter()

            print(
                f"[EcoScan][{request_id}] "
                ">>> INICIANDO INFERÊNCIA ONNX <<<",
                flush=True,
            )

            output = session.run(
                OUTPUT_NAMES,
                {
                    INPUT_NAME: input_tensor
                },
            )[0]

            inference_time = (
                time.perf_counter()
                - inference_start
            )

            print(
                f"[EcoScan][{request_id}] "
                f">>> INFERÊNCIA FINALIZADA: "
                f"{inference_time:.2f}s <<<",
                flush=True,
            )

        finally:
            INFERENCE_LOCK.release()

        results = decode_predictions(
            output,
            inference_frame.shape,
            ratio,
            pad_x,
            pad_y,
        )

        if FOCUS_MODE:
            for item in results:
                bbox = item.get("bbox")

                if bbox and len(bbox) >= 4:
                    bbox[0] += int(focus_offset_x)
                    bbox[1] += int(focus_offset_y)

            results = select_focus_predictions(
                results,
                frame.shape,
            )

        if DEBUG_PREDICTIONS:
            print(
                f"[EcoScan][{request_id}] "
                f"Predições finais: {results}",
                flush=True,
            )

        total_time = (
            time.perf_counter()
            - request_start
        )

        print(
            f"[EcoScan][{request_id}] "
            f"Detecções: {len(results)}",
            flush=True,
        )
        print(
            f"[EcoScan][{request_id}] "
            f"Tempo inferência: {inference_time:.3f}s",
            flush=True,
        )
        print(
            f"[EcoScan][{request_id}] "
            f"Tempo total: {total_time:.3f}s",
            flush=True,
        )
        print("=" * 60, flush=True)

        return {
            "predictions": results,
            "model": "GreenSorter ONNX",
            "image": {
                "width": original_w,
                "height": original_h,
            },
            "performance": {
                "inference_time_seconds": round(
                    inference_time,
                    3,
                ),
                "lock_wait_seconds": round(
                    lock_wait,
                    3,
                ),
                "total_time_seconds": round(
                    total_time,
                    3,
                ),
                "img_size": IMG_SIZE,
            },
        }

    except HTTPException:
        raise

    except MemoryError:
        print(
            f"[EcoScan][{request_id}] "
            "MEMORY ERROR durante a requisição.",
            flush=True,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Memória insuficiente para processar "
                "esta imagem."
            ),
        )

    except Exception as exc:
        print(
            f"[EcoScan][{request_id}] "
            f"ERRO: {type(exc).__name__}: {exc}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Erro interno ao processar a imagem."
            ),
        )

    finally:
        # Liberar referências temporárias SEM tocar na sessão ONNX.
        raw = None
        frame = None
        input_tensor = None
        output = None

        gc.collect()

        try:
            await file.close()
        except Exception:
            pass
