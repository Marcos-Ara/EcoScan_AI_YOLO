# ============================================================
# EcoScan AI - Runtime Docker
#
# Modelo YOLO11:
#   waste-sorting-smyr8/2
#
# O modelo é executado pelo Roboflow Serverless.
# O Render NÃO precisa instalar:
#   - PyTorch
#   - YOLOv7
#   - ONNX Runtime
#
# Isso reduz bastante o tamanho do container.
# ============================================================

FROM python:3.11-slim


# ============================================================
# AMBIENTE
# ============================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1


# ============================================================
# DIRETÓRIO
# ============================================================

WORKDIR /app


# ============================================================
# DEPENDÊNCIAS DO SISTEMA
#
# Algumas versões/dependências do OpenCV procuram bibliotecas
# gráficas no Linux. Instalamos as bibliotecas necessárias como
# proteção para o ambiente slim do Render.
# ============================================================

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# DEPENDÊNCIAS PYTHON
# ============================================================

COPY backend/requirements.txt ./requirements.txt


RUN python -m pip install \
    --no-cache-dir \
    --upgrade pip


RUN python -m pip install \
    --no-cache-dir \
    -r requirements.txt


# ============================================================
# FORÇAR OPENCV HEADLESS
#
# Evita conflitos caso alguma dependência instale
# opencv-python normal.
# ============================================================

RUN python -m pip uninstall -y \
    opencv-python \
    opencv-contrib-python \
    opencv-python-headless \
    || true


RUN python -m pip install \
    --no-cache-dir \
    opencv-python-headless==4.10.0.84


# ============================================================
# LIMPEZA
# ============================================================

RUN rm -rf /root/.cache/pip


# ============================================================
# TESTE DO OPENCV
#
# O build só continua se o cv2 realmente carregar.
# ============================================================

RUN python -c "\
import cv2; \
print('[EcoScan] OpenCV OK:', cv2.__version__); \
print('[EcoScan] OpenCV carregado corretamente no Render') \
"


# ============================================================
# API
# ============================================================

COPY backend/app.py ./app.py


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ENV ROBOFLOW_API_URL=https://serverless.roboflow.com

ENV ROBOFLOW_MODEL_ID=waste-sorting-smyr8/2

ENV CONFIDENCE=0.40

ENV IOU=0.45

ENV MAX_FILE_SIZE=1500000

ENV MAX_IMAGE_DIMENSION=960

ENV MAX_DETECTIONS=1

ENV FOCUS_MODE=true

ENV FOCUS_CROP_RATIO=0.90

ENV FOCUS_MAX_DETECTIONS=1

ENV DEBUG_PREDICTIONS=false

ENV ROBOFLOW_TIMEOUT_SECONDS=30

ENV ALLOWED_ORIGIN=https://marcos-ara.github.io


# ============================================================
# PORTA
# ============================================================

EXPOSE 10000


# ============================================================
# START
# ============================================================

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]