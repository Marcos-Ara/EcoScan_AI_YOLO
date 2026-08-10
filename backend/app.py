from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent
GREEN_SORTER_DIR = ROOT / "GreenSorter"
MODEL_PATH = ROOT / "model" / "model.pt"

# GreenSorter is a YOLOv7 project. Its README declares:
# cardboard, metal, rigid_plastic, soft_plastic.
GREEN_SORTER_CLASSES = {
    "cardboard": "papel",
    "metal": "metal",
    "rigid_plastic": "plastico",
    "soft_plastic": "plastico",
}

WASTE_RULES = {
    "papel": {"category": "Papel", "bin": "Azul", "color": "#2f6ef3"},
    "plastico": {"category": "Plástico", "bin": "Vermelha", "color": "#df4b42"},
    "vidro": {"category": "Vidro", "bin": "Verde", "color": "#43a047"},
    "metal": {"category": "Metal", "bin": "Amarela", "color": "#d6a800"},
    "organico": {"category": "Orgânico", "bin": "Marrom", "color": "#8b5a2b"},
    "rejeito": {"category": "Rejeito", "bin": "Cinza/Preta", "color": "#5b6068"},
}

CONFIDENCE = 0.35
IOU = 0.45
IMG_SIZE = 640
MAX_DETECTIONS = 20

app = FastAPI(title="EcoScan AI YOLO API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Local development. Restrict this before public deployment.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def import_yolov7():
    if not GREEN_SORTER_DIR.exists():
        raise RuntimeError(
            "GreenSorter não encontrado. Execute setup_model.ps1 para clonar o repositório."
        )

    yolov7_path = GREEN_SORTER_DIR / "yolov7"
    if not yolov7_path.exists():
        raise RuntimeError("A pasta GreenSorter/yolov7 não existe.")

    sys.path.insert(0, str(GREEN_SORTER_DIR))
    sys.path.insert(0, str(yolov7_path))

    from models.experimental import attempt_load
    from utils.datasets import letterbox
    from utils.general import non_max_suppression, scale_coords
    from utils.torch_utils import select_device

    return attempt_load, letterbox, non_max_suppression, scale_coords, select_device


def load_model():
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Modelo não encontrado em {MODEL_PATH}. "
            "Coloque o model.pt do GreenSorter nesse caminho."
        )

    attempt_load, _, _, _, select_device = import_yolov7()
    selected_device = select_device(device)
    model = attempt_load(str(MODEL_PATH), map_location=selected_device)
    model.eval()

    # Half precision somente na GPU.
    if selected_device.type != "cpu":
        model.half()

    return model


@app.on_event("startup")
def startup():
    try:
        load_model()
        print(f"[EcoScan] Modelo carregado em {device}: {MODEL_PATH}")
    except Exception as exc:
        # O servidor continua subindo para que /health mostre o erro.
        print(f"[EcoScan] ERRO ao carregar modelo: {exc}")


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": model is not None,
        "device": device,
        "model": str(MODEL_PATH),
        "supported_green_sorter_classes": list(GREEN_SORTER_CLASSES.keys()),
        "future_classes": ["vidro", "organico", "rejeito"],
    }


def prepare_image(frame):
    _, letterbox, _, _, _ = import_yolov7()

    # BGR -> RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resized = letterbox(img, IMG_SIZE, stride=32, auto=True)[0]
    resized = resized.transpose((2, 0, 1))
    resized = np.ascontiguousarray(resized)

    tensor = torch.from_numpy(resized).to(device)
    tensor = tensor.half() if device != "cpu" else tensor.float()
    tensor /= 255.0

    if tensor.ndimension() == 3:
        tensor = tensor.unsqueeze(0)

    return tensor, img.shape[:2], resized.shape[1:]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Confira backend/model/model.pt e o GreenSorter.",
        )

    raw = await file.read()
    array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Imagem inválida.")

    tensor, original_shape, _ = prepare_image(frame)

    with torch.no_grad():
        prediction = model(tensor, augment=False)[0]

    _, _, non_max_suppression, scale_coords, _ = import_yolov7()
    detections = non_max_suppression(
        prediction,
        CONFIDENCE,
        IOU,
        classes=None,
        agnostic=False,
    )[0]

    results = []

    if detections is not None and len(detections):
        detections[:, :4] = scale_coords(
            tensor.shape[2:],
            detections[:, :4],
            frame.shape
        ).round()

        names = getattr(model, "names", {})

        for *xyxy, conf, cls in detections[:MAX_DETECTIONS].tolist():
            class_id = int(cls)
            if isinstance(names, dict):
                source_class = names.get(class_id, str(class_id))
            else:
                source_class = names[class_id]

            source_class = str(source_class).lower().strip()
            category_key = GREEN_SORTER_CLASSES.get(source_class)

            # Só mostramos categorias que o modelo realmente conhece.
            if category_key is None:
                continue

            x1, y1, x2, y2 = map(int, xyxy)
            rule = WASTE_RULES[category_key]

            results.append({
                "source_class": source_class,
                "category_key": category_key,
                "category": rule["category"],
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
        "supported_classes": list(GREEN_SORTER_CLASSES.keys()),
    }
