import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

YOLOV7_DIR = ROOT / "GreenSorter" / "yolov7"
MODEL_PATH = ROOT / "model.pt"

if not YOLOV7_DIR.exists():
    raise RuntimeError(
        f"Diretório YOLOv7 não encontrado: {YOLOV7_DIR}"
    )

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Modelo não encontrado: {MODEL_PATH}. "
        "O download_model.py precisa ser executado durante o build."
    )


# ============================================================
# YOLOV7 IMPORT PATH
# ============================================================

sys.path.insert(0, str(YOLOV7_DIR))


from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device


# ============================================================
# CONFIGURAÇÕES
# ============================================================

CONFIDENCE = float(
    os.getenv("CONFIDENCE", "0.35")
)

IOU = float(
    os.getenv("IOU", "0.45")
)

IMG_SIZE = int(
    os.getenv("IMG_SIZE", "512")
)

DEVICE_NAME = os.getenv(
    "DEVICE",
    "cpu"
)


# ============================================================
# CLASSES DO GREENSORTER
# ============================================================

# O modelo atual do GreenSorter possui estas classes.
#
# cardboard      -> papel
# metal          -> metal
# rigid_plastic  -> plástico
# soft_plastic   -> plástico
#
# Não vamos fingir que o modelo atual detecta vidro,
# orgânico ou rejeito enquanto não houver pesos treinados
# para essas classes.

CLASS_MAP = {
    "cardboard": "papel",
    "metal": "metal",
    "rigid_plastic": "plastico",
    "soft_plastic": "plastico",
}


# ============================================================
# REGRAS DA COLETA SELETIVA
# ============================================================

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


# ============================================================
# DEVICE
# ============================================================

device = select_device(DEVICE_NAME)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("[EcoScan] Inicializando modelo...")
print(f"[EcoScan] Modelo: {MODEL_PATH}")
print(f"[EcoScan] Device: {device}")
print(f"[EcoScan] Confidence: {CONFIDENCE}")
print(f"[EcoScan] IoU: {IOU}")
print(f"[EcoScan] Image size: {IMG_SIZE}")
print("=" * 60)


model = attempt_load(
    str(MODEL_PATH),
    map_location=device
)

model.eval()


if device.type != "cpu":
    model.half()


names = model.names


print("[EcoScan] Modelo carregado com sucesso.")
print(f"[EcoScan] Classes do modelo: {names}")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="EcoScan AI YOLO API",
    version="1.1.0",
    description=(
        "API de detecção de resíduos recicláveis "
        "usando YOLOv7 GreenSorter."
    ),
)


# ============================================================
# CORS
# ============================================================

allowed_origin = os.getenv(
    "ALLOWED_ORIGIN",
    "*"
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

    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],

    allow_headers=["*"],
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "EcoScan AI YOLO API",
        "status": "online",
        "model": "GreenSorter YOLOv7",
        "version": "1.1.0",
        "health": "/health",
        "predict": "/predict",
        "docs": "/docs",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "ok": True,
        "model_loaded": model is not None,
        "model": "GreenSorter YOLOv7",
        "device": str(device),
        "classes": list(CLASS_MAP.keys()),
        "confidence": CONFIDENCE,
        "iou": IOU,
        "img_size": IMG_SIZE,
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # VALIDAR MIME TYPE
    # --------------------------------------------------------

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Envie uma imagem."
        )


    # --------------------------------------------------------
    # READ FILE
    # --------------------------------------------------------

    raw = await file.read()


    if not raw:

        raise HTTPException(
            status_code=400,
            detail="Imagem vazia."
        )


    # --------------------------------------------------------
    # LIMITAR TAMANHO
    # --------------------------------------------------------

    MAX_FILE_SIZE = 2_500_000

    if len(raw) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail=(
                "Imagem muito grande. "
                "Envie uma imagem menor."
            )
        )


    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    frame = cv2.imdecode(
        np.frombuffer(
            raw,
            dtype=np.uint8
        ),
        cv2.IMREAD_COLOR
    )


    if frame is None:

        raise HTTPException(
            status_code=400,
            detail="Imagem inválida."
        )


    original_h, original_w = frame.shape[:2]


    # --------------------------------------------------------
    # PREPROCESSAMENTO
    # --------------------------------------------------------

    img = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    img = letterbox(
        img,
        IMG_SIZE,
        stride=32,
        auto=True
    )[0]


    img = img.transpose(
        (2, 0, 1)
    )


    img = np.ascontiguousarray(img)


    tensor = torch.from_numpy(
        img
    ).to(device)


    if device.type != "cpu":

        tensor = tensor.half()

    else:

        tensor = tensor.float()


    tensor /= 255.0


    if tensor.ndimension() == 3:

        tensor = tensor.unsqueeze(0)


    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        pred = model(
            tensor,
            augment=False
        )[0]


    # --------------------------------------------------------
    # NON-MAX SUPPRESSION
    # --------------------------------------------------------

    det = non_max_suppression(
        pred,
        CONFIDENCE,
        IOU,
        classes=None,
        agnostic=False,
    )[0]


    results = []


    # --------------------------------------------------------
    # PROCESSAR DETECÇÕES
    # --------------------------------------------------------

    if det is not None and len(det):

        det[:, :4] = scale_coords(
            tensor.shape[2:],
            det[:, :4],
            frame.shape,
        ).round()


        for *xyxy, conf, cls in det.tolist():

            class_id = int(cls)


            # Segurança contra índice inválido
            if class_id < 0 or class_id >= len(names):

                continue


            source_class = str(
                names[class_id]
            ).lower().strip()


            category_key = CLASS_MAP.get(
                source_class
            )


            # Ignorar classes que não pertencem
            # ao nosso mapa atual.
            if category_key is None:

                continue


            x1, y1, x2, y2 = map(
                int,
                xyxy
            )


            rule = RULES[
                category_key
            ]


            results.append({

                "source_class":
                    source_class,

                "category":
                    rule["category"],

                "category_key":
                    category_key,

                "bin":
                    rule["bin"],

                "destination":
                    rule["destination"],

                "decomposition":
                    rule["decomposition"],

                "score":
                    float(conf),

                "bbox": [

                    x1,
                    y1,

                    max(
                        0,
                        x2 - x1
                    ),

                    max(
                        0,
                        y2 - y1
                    ),
                ],
            })


    # --------------------------------------------------------
    # ORDENAR POR CONFIANÇA
    # --------------------------------------------------------

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )


    # --------------------------------------------------------
    # RESPOSTA
    # --------------------------------------------------------

    return {

        "predictions": results,

        "model":
            "GreenSorter YOLOv7",

        "image": {

            "width":
                original_w,

            "height":
                original_h,
        },
    }