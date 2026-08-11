import os
import sys
import gc
import traceback
from pathlib import Path

# ============================================================
# OTIMIZAÇÃO DE MEMÓRIA / CPU
# ============================================================

# Render Free possui memória limitada.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
import torch

# Limitar threads do PyTorch
torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

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

# Render Free / proteção de memória
IMG_SIZE = int(
    os.getenv("IMG_SIZE", "320")
)

DEVICE_NAME = os.getenv(
    "DEVICE",
    "cpu"
)

# Limite máximo do arquivo recebido
MAX_FILE_SIZE = int(
    os.getenv("MAX_FILE_SIZE", "2500000")
)

# Limite de resolução da imagem.
# Imagens maiores serão reduzidas antes da inferência.
MAX_IMAGE_DIMENSION = int(
    os.getenv("MAX_IMAGE_DIMENSION", "1280")
)


# ============================================================
# CLASSES DO GREENSORTER
# ============================================================

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

device = select_device(
    DEVICE_NAME
)


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
print(f"[EcoScan] Max file size: {MAX_FILE_SIZE} bytes")
print(
    f"[EcoScan] Max image dimension: "
    f"{MAX_IMAGE_DIMENSION}px"
)
print("[EcoScan] PyTorch threads: 1")
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
    version="1.3.0",
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
        "version": "1.3.0",
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

        "classes": list(
            CLASS_MAP.keys()
        ),

        "confidence": CONFIDENCE,

        "iou": IOU,

        "img_size": IMG_SIZE,

        "max_file_size": MAX_FILE_SIZE,

        "max_image_dimension":
            MAX_IMAGE_DIMENSION,
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    raw = None
    frame = None
    img = None
    tensor = None
    pred = None
    det = None

    request_id = id(file)

    print("=" * 60)
    print(
        f"[EcoScan][{request_id}] "
        "NOVA REQUISIÇÃO /predict"
    )

    try:

        # ====================================================
        # 1. VALIDAR MIME TYPE
        # ====================================================

        print(
            f"[EcoScan][{request_id}] "
            f"Filename: {file.filename}"
        )

        print(
            f"[EcoScan][{request_id}] "
            f"Content-Type: {file.content_type}"
        )

        if (
            not file.content_type
            or not file.content_type.startswith("image/")
        ):

            print(
                f"[EcoScan][{request_id}] "
                "ERRO: MIME TYPE inválido"
            )

            raise HTTPException(
                status_code=400,
                detail="Envie uma imagem."
            )


        # ====================================================
        # 2. READ FILE
        # ====================================================

        print(
            f"[EcoScan][{request_id}] "
            "Lendo arquivo..."
        )

        raw = await file.read()

        print(
            f"[EcoScan][{request_id}] "
            f"Arquivo recebido: {len(raw)} bytes"
        )

        if not raw:

            raise HTTPException(
                status_code=400,
                detail="Imagem vazia."
            )


        # ====================================================
        # 3. LIMITAR TAMANHO DO ARQUIVO
        # ====================================================

        if len(raw) > MAX_FILE_SIZE:

            print(
                f"[EcoScan][{request_id}] "
                "ERRO: arquivo excede limite"
            )

            raise HTTPException(
                status_code=413,
                detail=(
                    "Imagem muito grande. "
                    "Envie uma imagem menor."
                )
            )


        # ====================================================
        # 4. DECODE
        # ====================================================

        print(
            f"[EcoScan][{request_id}] "
            "Decodificando imagem..."
        )

        frame = cv2.imdecode(
            np.frombuffer(
                raw,
                dtype=np.uint8
            ),
            cv2.IMREAD_COLOR
        )

        if frame is None:

            print(
                f"[EcoScan][{request_id}] "
                "ERRO: cv2.imdecode falhou"
            )

            raise HTTPException(
                status_code=400,
                detail="Imagem inválida."
            )


        original_h, original_w = frame.shape[:2]

        print(
            f"[EcoScan][{request_id}] "
            f"Resolução original: "
            f"{original_w}x{original_h}"
        )


        # ====================================================
        # 5. LIMITAR RESOLUÇÃO
        # ====================================================

        max_dimension = max(
            original_w,
            original_h
        )

        if max_dimension > MAX_IMAGE_DIMENSION:

            scale = (
                MAX_IMAGE_DIMENSION
                / max_dimension
            )

            new_w = max(
                1,
                int(original_w * scale)
            )

            new_h = max(
                1,
                int(original_h * scale)
            )

            print(
                f"[EcoScan][{request_id}] "
                f"Reduzindo imagem para "
                f"{new_w}x{new_h}"
            )

            frame = cv2.resize(
                frame,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

        else:

            print(
                f"[EcoScan][{request_id}] "
                "Resolução dentro do limite."
            )


        # ====================================================
        # 6. PREPROCESSAMENTO
        # ====================================================

        print(
            f"[EcoScan][{request_id}] "
            "Convertendo BGR -> RGB..."
        )

        img = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        print(
            f"[EcoScan][{request_id}] "
            "Executando letterbox..."
        )

        img = letterbox(
            img,
            IMG_SIZE,
            stride=32,
            auto=True
        )[0]


        print(
            f"[EcoScan][{request_id}] "
            f"Shape após letterbox: {img.shape}"
        )


        img = img.transpose(
            (2, 0, 1)
        )

        img = np.ascontiguousarray(
            img
        )


        # ====================================================
        # 7. CRIAR TENSOR
        # ====================================================

        print(
            f"[EcoScan][{request_id}] "
            "Criando tensor..."
        )

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


        print(
            f"[EcoScan][{request_id}] "
            f"Tensor criado: "
            f"{tuple(tensor.shape)}"
        )


        # ====================================================
        # 8. LIBERAR OBJETOS QUE NÃO SÃO MAIS NECESSÁRIOS
        # ====================================================

        del raw
        raw = None

        del img
        img = None

        gc.collect()


        # ====================================================
        # 9. INFERENCE
        # ====================================================

        print(
            f"[EcoScan][{request_id}] "
            ">>> INICIANDO INFERÊNCIA YOLO <<<"
        )

        with torch.inference_mode():

            pred = model(
                tensor,
                augment=False
            )[0]


        print(
            f"[EcoScan][{request_id}] "
            ">>> INFERÊNCIA CONCLUÍDA <<<"
        )

        print(
            f"[EcoScan][{request_id}] "
            f"Pred shape: {tuple(pred.shape)}"
        )


        # ====================================================
        # 10. NON-MAX SUPPRESSION
        # ====================================================

        print(
            f"[EcoScan][{request_id}] "
            "Executando NMS..."
        )

        det = non_max_suppression(
            pred,
            CONFIDENCE,
            IOU,
            classes=None,
            agnostic=False,
        )[0]


        print(
            f"[EcoScan][{request_id}] "
            "NMS concluído."
        )


        results = []


        # ====================================================
        # 11. PROCESSAR DETECÇÕES
        # ====================================================

        if det is not None and len(det):

            print(
                f"[EcoScan][{request_id}] "
                f"Detecções encontradas: {len(det)}"
            )

            det[:, :4] = scale_coords(
                tensor.shape[2:],
                det[:, :4],
                frame.shape,
            ).round()


            for *xyxy, conf, cls in det.tolist():

                class_id = int(cls)


                # Segurança contra índice inválido
                if (
                    class_id < 0
                    or class_id >= len(names)
                ):

                    print(
                        f"[EcoScan][{request_id}] "
                        f"Classe inválida: {class_id}"
                    )

                    continue


                source_class = str(
                    names[class_id]
                ).lower().strip()


                category_key = CLASS_MAP.get(
                    source_class
                )


                # Ignorar classes desconhecidas
                if category_key is None:

                    print(
                        f"[EcoScan][{request_id}] "
                        f"Classe ignorada: "
                        f"{source_class}"
                    )

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

        else:

            print(
                f"[EcoScan][{request_id}] "
                "Nenhuma detecção encontrada."
            )


        # ====================================================
        # 12. ORDENAR RESULTADOS
        # ====================================================

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )


        print(
            f"[EcoScan][{request_id}] "
            f"Resultados finais: {len(results)}"
        )


        # ====================================================
        # 13. RESPOSTA
        # ====================================================

        response = {

            "predictions":
                results,

            "model":
                "GreenSorter YOLOv7",

            "image": {

                "width":
                    original_w,

                "height":
                    original_h,
            },

            "processed_image": {

                "width":
                    frame.shape[1],

                "height":
                    frame.shape[0],
            },
        }


        print(
            f"[EcoScan][{request_id}] "
            ">>> REQUISIÇÃO CONCLUÍDA COM SUCESSO <<<"
        )

        print("=" * 60)

        return response


    except HTTPException:
        raise


    except Exception as exc:

        print("=" * 60)

        print(
            f"[EcoScan][{request_id}] "
            "!!! ERRO DURANTE /predict !!!"
        )

        print(
            f"[EcoScan][{request_id}] "
            f"Tipo: {type(exc).__name__}"
        )

        print(
            f"[EcoScan][{request_id}] "
            f"Mensagem: {str(exc)}"
        )

        print(
            f"[EcoScan][{request_id}] "
            "Traceback:"
        )

        traceback.print_exc()

        print("=" * 60)


        raise HTTPException(
            status_code=500,
            detail=(
                "Erro interno durante o processamento "
                f"da imagem: {type(exc).__name__}: {str(exc)}"
            )
        )


    finally:

        # ====================================================
        # LIMPEZA DE MEMÓRIA
        # ====================================================

        print(
            f"[EcoScan][{request_id}] "
            "Executando limpeza de memória..."
        )

        try:

            if pred is not None:
                del pred

            if det is not None:
                del det

            if tensor is not None:
                del tensor

            if img is not None:
                del img

            if frame is not None:
                del frame

            if raw is not None:
                del raw

        except Exception:
            pass


        gc.collect()


        # Limpeza CUDA apenas se estiver sendo usada
        if torch.cuda.is_available():

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


        print(
            f"[EcoScan][{request_id}] "
            "Limpeza concluída."
        )