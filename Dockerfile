# ============================================================
# EcoScan AI - Multi-stage Docker
#
# BUILDER
#   - Python
#   - PyTorch
#   - YOLOv7
#   - Dependências necessárias para conversão
#   - Converte model.pt -> model.onnx
#
# RUNTIME
#   - FastAPI
#   - ONNX Runtime
#   - OpenCV
#   - SEM PyTorch
#   - SEM YOLOv7
# ============================================================


# ============================================================
# BUILDER
# ============================================================

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

WORKDIR /build


# ============================================================
# DEPENDÊNCIAS DE BUILD
# ============================================================

COPY backend/requirements-build.txt ./requirements-build.txt

RUN python -m pip install \
    --no-cache-dir \
    --upgrade pip

RUN python -m pip install \
    --no-cache-dir \
    -r requirements-build.txt


# ============================================================
# GARANTIA DAS DEPENDÊNCIAS DO YOLOV7
# ============================================================

# O YOLOv7 importa tqdm durante o carregamento do modelo.
# Instalamos explicitamente para evitar erro durante a conversão.

RUN python -m pip install \
    --no-cache-dir \
    "tqdm==4.67.1"


# ============================================================
# VERIFICAÇÃO DO AMBIENTE DE BUILD
# ============================================================

RUN python -c "\
import torch; \
import torchvision; \
import tqdm; \
import onnx; \
import cv2; \
print('[EcoScan] PyTorch:', torch.__version__); \
print('[EcoScan] Torchvision:', torchvision.__version__); \
print('[EcoScan] tqdm:', tqdm.__version__); \
print('[EcoScan] ONNX:', onnx.__version__); \
print('[EcoScan] OpenCV:', cv2.__version__); \
print('[EcoScan] Ambiente de BUILD OK') \
"


# ============================================================
# CÓDIGO YOLOV7
# ============================================================

COPY backend/GreenSorter ./GreenSorter


# ============================================================
# SCRIPTS DO ECO SCAN
# ============================================================

COPY backend/download_model.py ./download_model.py
COPY backend/convert_model.py ./convert_model.py


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ENV IMG_SIZE=224

# Importante:
# O convert_model.py utiliza:
#
# from models.experimental import attempt_load
#
# Por isso o diretório yolov7 precisa estar no PYTHONPATH.

ENV PYTHONPATH=/build/GreenSorter/yolov7:/build


# ============================================================
# DOWNLOAD DO MODELO
# ============================================================

RUN python download_model.py


# ============================================================
# CONVERSÃO
#
# model.pt
#     ↓
# YOLOv7
#     ↓
# model.onnx
# ============================================================

RUN python convert_model.py


# ============================================================
# VERIFICAÇÃO DO MODELO ONNX
# ============================================================

RUN python -c "\
import os; \
import onnx; \
path='model.onnx'; \
assert os.path.exists(path), 'model.onnx não foi criado'; \
model=onnx.load(path); \
onnx.checker.check_model(model); \
print('[EcoScan] ======================================'); \
print('[EcoScan] ONNX VALIDADO COM SUCESSO'); \
print('[EcoScan] Tamanho:', round(os.path.getsize(path)/1024/1024, 2), 'MB'); \
print('[EcoScan] ======================================') \
"


# ============================================================
# RUNTIME
# ============================================================

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1


# ============================================================
# CONFIGURAÇÃO ECO SCAN
# ============================================================

ENV DEVICE=cpu
ENV CONFIDENCE=0.35
ENV IOU=0.45
ENV IMG_SIZE=224

ENV MAX_FILE_SIZE=1500000
ENV MAX_IMAGE_DIMENSION=960
ENV MAX_DETECTIONS=20


# ============================================================
# DIRETÓRIO
# ============================================================

WORKDIR /app


# ============================================================
# DEPENDÊNCIAS DO RUNTIME
# ============================================================

COPY backend/requirements.txt ./requirements.txt

RUN python -m pip install \
    --no-cache-dir \
    -r requirements.txt \
    && rm -rf /root/.cache/pip


# ============================================================
# API
# ============================================================

COPY backend/app.py ./app.py


# ============================================================
# MODELO ONNX
#
# Apenas o modelo convertido entra no runtime.
#
# NÃO copiamos:
#
# - model.pt
# - GreenSorter
# - PyTorch
# - YOLOv7
# - convert_model.py
# ============================================================

COPY --from=builder /build/model.onnx ./model.onnx


# ============================================================
# VERIFICAÇÃO DO RUNTIME
# ============================================================

RUN python -c "\
import os; \
import onnxruntime; \
path='model.onnx'; \
assert os.path.exists(path), 'model.onnx não encontrado no runtime'; \
print('[EcoScan] ======================================'); \
print('[EcoScan] RUNTIME OK'); \
print('[EcoScan] ONNX Runtime:', onnxruntime.__version__); \
print('[EcoScan] Modelo:', round(os.path.getsize(path)/1024/1024, 2), 'MB'); \
print('[EcoScan] ======================================') \
"


# ============================================================
# PORTA
# ============================================================

EXPOSE 10000


# ============================================================
# START
# ============================================================

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]