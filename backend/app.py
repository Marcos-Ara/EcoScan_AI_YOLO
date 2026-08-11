import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent
YOLOV7_DIR = ROOT / "GreenSorter" / "yolov7"
MODEL_PATH = ROOT / "model.pt"

sys.path.insert(0, str(YOLOV7_DIR))

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Classes realmente aprendidas pelos pesos atuais do GreenSorter.
CLASS_MAP = {
    "cardboard": "papel",
    "metal": "metal",
    "rigid_plastic": "plastico",
    "soft_plastic": "plastico",
}

RULES = {
    "papel": {"category": "Papel", "bin": "Azul"},
    "plastico": {"category": "Plástico", "bin": "Vermelha"},
    "metal": {"category": "Metal", "bin": "Amarela"},
    # Reservadas para o futuro fine-tuning:
    "vidro": {"category": "Vidro", "bin": "Verde"},
    "organico": {"category": "Orgânico", "bin": "Marrom"},
    "rejeito": {"category": "Rejeito", "bin": "Cinza/Preta"},
}

CONFIDENCE = float(os.getenv("CONFIDENCE", "0.35"))
IOU = float(os.getenv("IOU", "0.45"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))

device = select_device(os.getenv("DEVICE", "cpu"))

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"model.pt não encontrado em {MODEL_PATH}. "
        "Execute download_model.py durante o build."
    )

model = attempt_load(str(MODEL_PATH), map_location=device)
model.eval()

if device.type != "cpu":
    model.half()

names = model.names

app = FastAPI(
    title="EcoScan AI YOLO API",
    version="1.0.0",
    description="Backend de detecção de materiais recicláveis usando GreenSorter YOLOv7.",
)

allowed_origin = os.getenv("ALLOWED_ORIGIN", "*")
origins = [item.strip() for item in allowed_origin.split(",") if item.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "EcoScan AI YOLO API",
        "status": "online",
        "model": "GreenSorter YOLOv7",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": model is not None,
        "model": "GreenSorter YOLOv7",
        "device": str(device),
        "classes": list(CLASS_MAP.keys()),
        "confidence": CONFIDENCE,
        "img_size": IMG_SIZE,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie uma imagem.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Imagem vazia.")
    if len(raw) > 2_500_000:
        raise HTTPException(status_code=413, detail="Imagem muito grande.")

    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Imagem inválida.")

    original_h, original_w = frame.shape[:2]

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = letterbox(img, IMG_SIZE, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)

    tensor = torch.from_numpy(img).to(device)
    tensor = tensor.half() if device.type != "cpu" else tensor.float()
    tensor /= 255.0

    if tensor.ndimension() == 3:
        tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        pred = model(tensor, augment=False)[0]

    det = non_max_suppression(
        pred,
        CONFIDENCE,
        IOU,
        classes=None,
        agnostic=False,
    )[0]

    results = []

    if det is not None and len(det):
        # scale_coords recebe as dimensões da imagem que entrou no modelo
        # e devolve as coordenadas na imagem enviada à API.
        det[:, :4] = scale_coords(
            tensor.shape[2:],
            det[:, :4],
            frame.shape,
        ).round()

        for *xyxy, conf, cls in det.tolist():
            class_id = int(cls)
            source_class = str(names[class_id]).lower().strip()
            category_key = CLASS_MAP.get(source_class)

            if category_key is None:
                continue

            x1, y1, x2, y2 = map(int, xyxy)
            rule = RULES[category_key]

            results.append({
                "source_class": source_class,
                "category": rule["category"],
                "category_key": category_key,
                "bin": rule["bin"],
                "score": float(conf),
                "bbox": [
                    x1,
                    y1,
                    max(0, x2 - x1),
                    max(0, y2 - y1),
                ],
            })

    results.sort(key=lambda item: item["score"], reverse=True)

    return {
        "predictions": results,
        "model": "GreenSorter YOLOv7",
        "image": {"width": original_w, "height": original_h},
    }
