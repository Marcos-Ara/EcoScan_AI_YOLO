import os
import sys
import gc
import time
import threading
from pathlib import Path

# ============================================================
# OTIMIZAÇÃO DE MEMÓRIA / CPU
# ============================================================

# Render Free possui memória e CPU limitadas.
# Mantemos o PyTorch com apenas uma thread.

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Evita alguns comportamentos de paralelismo
# desnecessários do OpenMP.

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ============================================================
# IMPORTS
# ============================================================

import cv2
import numpy as np
import torch

# ============================================================
# PYTORCH THREADS
# ============================================================

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # Pode acontecer se o PyTorch já tiver iniciado
    # alguma estrutura interna de paralelismo.
    pass

# ============================================================
# FASTAPI
# ============================================================

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

YOLOV7_DIR = ROOT / "GreenSorter" / "yolov7"
MODEL_PATH = ROOT / "model.pt"

# ============================================================
# VALIDAR ESTRUTURA
# ============================================================

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

sys.path.insert(
    0,
    str(YOLOV7_DIR)
)

from models.experimental import attempt_load

from utils.datasets import letterbox

from utils.general import (
    non_max_suppression,
    scale_coords,
)

from utils.torch_utils import select_device

# ============================================================
# CONFIGURAÇÕES
# ============================================================

CONFIDENCE = float(
    os.getenv(
        "CONFIDENCE",
        "0.35"
    )
)

IOU = float(
    os.getenv(
        "IOU",
        "0.45"
    )
)

# ============================================================
# IMPORTANTE
# ============================================================
#
# 224 é proposital.
#
# O modelo estava travando durante a inferência
# utilizando 320.
#
# 224 também é múltiplo de 32:
#
# 224 / 32 = 7
#
# Isso é adequado para YOLOv7.
#
# ============================================================

IMG_SIZE = int(
    os.getenv(
        "IMG_SIZE",
        "224"
    )
)

DEVICE_NAME = os.getenv(
    "DEVICE",
    "cpu"
)

# ============================================================
# LIMITES DE SEGURANÇA
# ============================================================

# Limite do arquivo recebido.
#
# 2,5 MB é suficiente para fotos comuns.
#
MAX_FILE_SIZE = int(
    os.getenv(
        "MAX_FILE_SIZE",
        "2500000"
    )
)

# Limite da resolução original.
#
# Imagens gigantes podem consumir muita memória
# durante o decode e preprocessing.

MAX_IMAGE_DIMENSION = int(
    os.getenv(
        "MAX_IMAGE_DIMENSION",
        "1280"
    )
)

# ============================================================
# PROTEÇÃO DE CONCORRÊNCIA
# ============================================================
#
# Render pode receber duas requisições ao mesmo tempo.
#
# Se duas inferências YOLO forem executadas simultaneamente,
# o consumo de memória pode aumentar bastante.
#
# Por isso permitimos apenas UMA inferência por vez.
#
# ============================================================

INFERENCE_LOCK = threading.Lock()

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

print("=" * 60, flush=True)

print(
    "[EcoScan] Selecionando dispositivo...",
    flush=True
)

device = select_device(
    DEVICE_NAME
)

print(
    f"[EcoScan] Device selecionado: {device}",
    flush=True
)

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60, flush=True)

print(
    "[EcoScan] Inicializando modelo...",
    flush=True
)

print(
    f"[EcoScan] Modelo: {MODEL_PATH}",
    flush=True
)

print(
    f"[EcoScan] Device: {device}",
    flush=True
)

print(
    f"[EcoScan] Confidence: {CONFIDENCE}",
    flush=True
)

print(
    f"[EcoScan] IoU: {IOU}",
    flush=True
)

print(
    f"[EcoScan] Image size: {IMG_SIZE}",
    flush=True
)

print(
    f"[EcoScan] Max file size: {MAX_FILE_SIZE} bytes",
    flush=True
)

print(
    f"[EcoScan] Max image dimension: "
    f"{MAX_IMAGE_DIMENSION}px",
    flush=True
)

print(
    "[EcoScan] PyTorch threads: 1",
    flush=True
)

print("=" * 60, flush=True)

# ============================================================
# CARREGAR PESOS
# ============================================================

model_load_start = time.time()

try:

    model = attempt_load(
        str(MODEL_PATH),
        map_location=device
    )

except Exception as e:

    print(
        "[EcoScan] ERRO AO CARREGAR MODELO:",
        repr(e),
        flush=True
    )

    raise

# ============================================================
# EVAL
# ============================================================

model.eval()

# ============================================================
# FUSE
# ============================================================
#
# Junta algumas operações Conv + BatchNorm.
#
# Isso pode reduzir o custo da inferência.
#
# ============================================================

try:

    print(
        "[EcoScan] Executando model.fuse()...",
        flush=True
    )

    model.fuse()

    print(
        "[EcoScan] Modelo fundido com sucesso.",
        flush=True
    )

except Exception as e:

    print(
        "[EcoScan] AVISO: model.fuse() falhou.",
        flush=True
    )

    print(
        f"[EcoScan] Motivo: {repr(e)}",
        flush=True
    )

# ============================================================
# PRECISÃO
# ============================================================

if device.type != "cpu":

    print(
        "[EcoScan] GPU detectada. "
        "Ativando FP16.",
        flush=True
    )

    model.half()

else:

    # Render Free normalmente utiliza CPU.
    #
    # Mantemos float32.
    #
    model.float()

# ============================================================
# CLASSES
# ============================================================

names = model.names

model_load_time = (
    time.time() - model_load_start
)

print(
    "[EcoScan] Modelo carregado com sucesso.",
    flush=True
)

print(
    f"[EcoScan] Classes do modelo: {names}",
    flush=True
)

print(
    f"[EcoScan] Tempo carregamento/fuse: "
    f"{model_load_time:.2f}s",
    flush=True
)

print("=" * 60, flush=True)

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

        "service":
            "EcoScan AI YOLO API",

        "status":
            "online",

        "model":
            "GreenSorter YOLOv7",

        "version":
            "1.3.0",

        "health":
            "/health",

        "predict":
            "/predict",

        "docs":
            "/docs",
    }

# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "ok":
            True,

        "model_loaded":
            model is not None,

        "model":
            "GreenSorter YOLOv7",

        "device":
            str(device),

        "classes":
            list(
                CLASS_MAP.keys()
            ),

        "confidence":
            CONFIDENCE,

        "iou":
            IOU,

        "img_size":
            IMG_SIZE,

        "max_file_size":
            MAX_FILE_SIZE,

        "max_image_dimension":
            MAX_IMAGE_DIMENSION,

        "inference_lock":
            "enabled",
    }

# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # ========================================================
    # ID DA REQUISIÇÃO
    # ========================================================

    request_id = id(file)

    print(
        "",
        flush=True
    )

    print(
        "=" * 60,
        flush=True
    )

    print(
        f"[EcoScan][{request_id}] "
        "NOVA REQUISIÇÃO /predict",
        flush=True
    )

    print(
        f"[EcoScan][{request_id}] "
        f"Filename: {file.filename}",
        flush=True
    )

    print(
        f"[EcoScan][{request_id}] "
        f"Content-Type: {file.content_type}",
        flush=True
    )

    print(
        "=" * 60,
        flush=True
    )

    request_start = time.time()

    raw = None
    frame = None
    img = None
    tensor = None
    pred = None
    det = None

    # ========================================================
    # VALIDAR MIME TYPE
    # ========================================================

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):

        print(
            f"[EcoScan][{request_id}] "
            "ERRO: arquivo não é imagem.",
            flush=True
        )

        raise HTTPException(
            status_code=400,
            detail="Envie uma imagem."
        )

    # ========================================================
    # READ FILE
    # ========================================================

    try:

        print(
            f"[EcoScan][{request_id}] "
            "Lendo arquivo...",
            flush=True
        )

        raw = await file.read()

    except Exception as e:

        print(
            f"[EcoScan][{request_id}] "
            f"ERRO ao ler arquivo: {repr(e)}",
            flush=True
        )

        raise HTTPException(
            status_code=400,
            detail="Não foi possível ler a imagem."
        )

    # ========================================================
    # VALIDAR ARQUIVO
    # ========================================================

    if not raw:

        print(
            f"[EcoScan][{request_id}] "
            "ERRO: arquivo vazio.",
            flush=True
        )

        raise HTTPException(
            status_code=400,
            detail="Imagem vazia."
        )

    print(
        f"[EcoScan][{request_id}] "
        f"Arquivo recebido: {len(raw)} bytes",
        flush=True
    )

    # ========================================================
    # LIMITE DE TAMANHO
    # ========================================================

    if len(raw) > MAX_FILE_SIZE:

        print(
            f"[EcoScan][{request_id}] "
            f"ERRO: arquivo excede {MAX_FILE_SIZE} bytes.",
            flush=True
        )

        raise HTTPException(
            status_code=413,
            detail=(
                "Imagem muito grande. "
                "Envie uma imagem menor."
            )
        )

    # ========================================================
    # DECODE
    # ========================================================

    try:

        print(
            f"[EcoScan][{request_id}] "
            "Decodificando imagem...",
            flush=True
        )

        frame = cv2.imdecode(
            np.frombuffer(
                raw,
                dtype=np.uint8
            ),
            cv2.IMREAD_COLOR
        )

    except Exception as e:

        print(
            f"[EcoScan][{request_id}] "
            f"ERRO no OpenCV: {repr(e)}",
            flush=True
        )

        raise HTTPException(
            status_code=400,
            detail="Não foi possível decodificar a imagem."
        )

    if frame is None:

        print(
            f"[EcoScan][{request_id}] "
            "ERRO: imagem inválida.",
            flush=True
        )

        raise HTTPException(
            status_code=400,
            detail="Imagem inválida."
        )

    # ========================================================
    # RESOLUÇÃO ORIGINAL
    # ========================================================

    original_h, original_w = frame.shape[:2]

    print(
        f"[EcoScan][{request_id}] "
        f"Resolução original: "
        f"{original_w}x{original_h}",
        flush=True
    )

    # ========================================================
    # LIMITE DE RESOLUÇÃO
    # ========================================================

    if (
        original_w > MAX_IMAGE_DIMENSION
        or original_h > MAX_IMAGE_DIMENSION
    ):

        print(
            f"[EcoScan][{request_id}] "
            "Imagem acima do limite. "
            "Redimensionando...",
            flush=True
        )

        scale = min(
            MAX_IMAGE_DIMENSION / original_w,
            MAX_IMAGE_DIMENSION / original_h
        )

        new_w = max(
            1,
            int(original_w * scale)
        )

        new_h = max(
            1,
            int(original_h * scale)
        )

        frame = cv2.resize(
            frame,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        print(
            f"[EcoScan][{request_id}] "
            f"Nova resolução: "
            f"{new_w}x{new_h}",
            flush=True
        )

    else:

        print(
            f"[EcoScan][{request_id}] "
            "Resolução dentro do limite.",
            flush=True
        )

    # ========================================================
    # PREPROCESSAMENTO
    # ========================================================

    try:

        print(
            f"[EcoScan][{request_id}] "
            "Convertendo BGR -> RGB...",
            flush=True
        )

        img = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        print(
            f"[EcoScan][{request_id}] "
            "Executando letterbox...",
            flush=True
        )

        # auto=False garante exatamente IMG_SIZE x IMG_SIZE.
        #
        # Isso evita formatos inesperados e mantém o tensor
        # pequeno e previsível.

        img = letterbox(
            img,
            IMG_SIZE,
            stride=32,
            auto=False
        )[0]

        print(
            f"[EcoScan][{request_id}] "
            f"Shape após letterbox: {img.shape}",
            flush=True
        )

        # ====================================================
        # HWC -> CHW
        # ====================================================

        img = img.transpose(
            (2, 0, 1)
        )

        img = np.ascontiguousarray(
            img
        )

        # ====================================================
        # TENSOR
        # ====================================================

        print(
            f"[EcoScan][{request_id}] "
            "Criando tensor...",
            flush=True
        )

        tensor = torch.from_numpy(
            img
        ).to(device)

        # ====================================================
        # FLOAT
        # ====================================================

        if device.type != "cpu":

            tensor = tensor.half()

        else:

            tensor = tensor.float()

        # ====================================================
        # NORMALIZAÇÃO
        # ====================================================

        tensor /= 255.0

        # ====================================================
        # BATCH
        # ====================================================

        if tensor.ndimension() == 3:

            tensor = tensor.unsqueeze(0)

        print(
            f"[EcoScan][{request_id}] "
            f"Tensor criado: {tuple(tensor.shape)}",
            flush=True
        )

    except Exception as e:

        print(
            f"[EcoScan][{request_id}] "
            f"ERRO NO PREPROCESSAMENTO: {repr(e)}",
            flush=True
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Erro ao preparar a imagem "
                f"para o modelo: {str(e)}"
            )
        )

    # ========================================================
    # INFERÊNCIA
    # ========================================================
    #
    # Apenas uma requisição pode executar YOLO por vez.
    #
    # Isso evita duas imagens consumirem CPU/memória
    # simultaneamente.
    #
    # ========================================================

    print(
        f"[EcoScan][{request_id}] "
        "Aguardando lock de inferência...",
        flush=True
    )

    lock_start = time.time()

    INFERENCE_LOCK.acquire()

    lock_wait_time = (
        time.time() - lock_start
    )

    print(
        f"[EcoScan][{request_id}] "
        f"Lock adquirido após "
        f"{lock_wait_time:.2f}s",
        flush=True
    )

    inference_start = time.time()

    try:

        print(
            f"[EcoScan][{request_id}] "
            ">>> INICIANDO INFERÊNCIA YOLO <<<",
            flush=True
        )

        # ====================================================
        # INFERENCE MODE
        # ====================================================

        with torch.inference_mode():

            pred = model(
                tensor,
                augment=False
            )[0]

        inference_time = (
            time.time() - inference_start
        )

        print(
            f"[EcoScan][{request_id}] "
            f">>> INFERÊNCIA FINALIZADA: "
            f"{inference_time:.2f}s <<<",
            flush=True
        )

    except Exception as e:

        inference_time = (
            time.time() - inference_start
        )

        print(
            f"[EcoScan][{request_id}] "
            "!!! ERRO DURANTE INFERÊNCIA !!!",
            flush=True
        )

        print(
            f"[EcoScan][{request_id}] "
            f"Tempo até erro: "
            f"{inference_time:.2f}s",
            flush=True
        )

        print(
            f"[EcoScan][{request_id}] "
            f"Tipo: {type(e).__name__}",
            flush=True
        )

        print(
            f"[EcoScan][{request_id}] "
            f"Detalhes: {repr(e)}",
            flush=True
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Erro durante a inferência YOLO: "
                f"{str(e)}"
            )
        )

    finally:

        INFERENCE_LOCK.release()

        print(
            f"[EcoScan][{request_id}] "
            "Lock de inferência liberado.",
            flush=True
        )

    # ========================================================
    # NON-MAX SUPPRESSION
    # ========================================================

    try:

        print(
            f"[EcoScan][{request_id}] "
            "Executando Non-Max Suppression...",
            flush=True
        )

        det = non_max_suppression(
            pred,
            CONFIDENCE,
            IOU,
            classes=None,
            agnostic=False
        )[0]

        print(
            f"[EcoScan][{request_id}] "
            f"NMS finalizado. "
            f"Detecções brutas: "
            f"{0 if det is None else len(det)}",
            flush=True
        )

    except Exception as e:

        print(
            f"[EcoScan][{request_id}] "
            f"ERRO no NMS: {repr(e)}",
            flush=True
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Erro durante o processamento "
                f"das detecções: {str(e)}"
            )
        )

    # ========================================================
    # RESULTADOS
    # ========================================================

    results = []

    # ========================================================
    # PROCESSAR DETECÇÕES
    # ========================================================

    try:

        if det is not None and len(det):

            print(
                f"[EcoScan][{request_id}] "
                "Ajustando bounding boxes...",
                flush=True
            )

            det[:, :4] = scale_coords(
                tensor.shape[2:],
                det[:, :4],
                frame.shape
            ).round()

            for *xyxy, conf, cls in det.tolist():

                class_id = int(cls)

                # ============================================
                # SEGURANÇA
                # ============================================

                if (
                    class_id < 0
                    or class_id >= len(names)
                ):

                    continue

                source_class = str(
                    names[class_id]
                ).lower().strip()

                category_key = CLASS_MAP.get(
                    source_class
                )

                # ============================================
                # IGNORAR CLASSES DESCONHECIDAS
                # ============================================

                if category_key is None:

                    print(
                        f"[EcoScan][{request_id}] "
                        f"Classe ignorada: "
                        f"{source_class}",
                        flush=True
                    )

                    continue

                # ============================================
                # BBOX
                # ============================================

                x1, y1, x2, y2 = map(
                    int,
                    xyxy
                )

                # ============================================
                # GARANTIR COORDENADAS VÁLIDAS
                # ============================================

                x1 = max(
                    0,
                    min(
                        x1,
                        frame.shape[1]
                    )
                )

                y1 = max(
                    0,
                    min(
                        y1,
                        frame.shape[0]
                    )
                )

                x2 = max(
                    0,
                    min(
                        x2,
                        frame.shape[1]
                    )
                )

                y2 = max(
                    0,
                    min(
                        y2,
                        frame.shape[0]
                    )
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

    except Exception as e:

        print(
            f"[EcoScan][{request_id}] "
            f"ERRO processando resultados: "
            f"{repr(e)}",
            flush=True
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Erro ao processar "
                f"as detecções: {str(e)}"
            )
        )

    # ========================================================
    # ORDENAR POR CONFIANÇA
    # ========================================================

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # ========================================================
    # TEMPO TOTAL
    # ========================================================

    total_time = (
        time.time() - request_start
    )

    print(
        f"[EcoScan][{request_id}] "
        f"Detecções válidas: {len(results)}",
        flush=True
    )

    print(
        f"[EcoScan][{request_id}] "
        f"Tempo de inferência: "
        f"{inference_time:.2f}s",
        flush=True
    )

    print(
        f"[EcoScan][{request_id}] "
        f"Tempo total: "
        f"{total_time:.2f}s",
        flush=True
    )

    print(
        f"[EcoScan][{request_id}] "
        ">>> REQUISIÇÃO FINALIZADA <<<",
        flush=True
    )

    print(
        "=" * 60,
        flush=True
    )

    # ========================================================
    # LIMPEZA
    # ========================================================

    # Não apagamos o modelo.
    #
    # Somente objetos temporários da requisição.

    del pred
    del det
    del tensor
    del img
    del frame
    del raw

    # Executar garbage collector para liberar objetos Python
    # temporários.

    gc.collect()

    # ========================================================
    # RESPOSTA
    # ========================================================

    return {

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

        "performance": {

            "inference_time_seconds":
                round(
                    inference_time,
                    3
                ),

            "lock_wait_seconds":
                round(
                    lock_wait_time,
                    3
                ),

            "total_time_seconds":
                round(
                    total_time,
                    3
                ),

            "img_size":
                IMG_SIZE,
        },
    }