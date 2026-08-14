# ============================================================
# EcoScan AI - Runtime Docker
#
# Modelo YOLO11:
#   waste-sorting-smyr8/2
#
# O modelo é executado pelo Roboflow Serverless.
# O Render NÃO precisa instalar PyTorch, YOLOv7 ou ONNX.
# Isso reduz muito o tamanho e o tempo de deploy.
# ============================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt

RUN python -m pip install \
    --no-cache-dir \
    --upgrade pip \
    && python -m pip install \
    --no-cache-dir \
    -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY backend/app.py ./app.py

# Configuração padrão; valores sensíveis ficam no Render.
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

EXPOSE 10000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
