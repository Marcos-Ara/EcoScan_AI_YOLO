# ============================================================
# EcoScan AI - Multi-stage Docker
#
# Builder: usa PyTorch apenas para converter model.pt -> ONNX.
# Runtime: NÃO instala PyTorch e NÃO carrega YOLOv7.
# Isso reduz RAM e tamanho da imagem final.
# ============================================================

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

WORKDIR /build

COPY backend/requirements-build.txt ./requirements-build.txt

RUN pip install \
    --no-cache-dir \
    --upgrade pip \
    && pip install \
    --no-cache-dir \
    -r requirements-build.txt

COPY backend/GreenSorter ./GreenSorter
COPY backend/download_model.py ./download_model.py
COPY backend/convert_model.py ./convert_model.py

ENV IMG_SIZE=224

RUN python download_model.py
RUN python convert_model.py

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

ENV DEVICE=cpu
ENV CONFIDENCE=0.35
ENV IOU=0.45
ENV IMG_SIZE=224
ENV MAX_FILE_SIZE=1500000
ENV MAX_IMAGE_DIMENSION=960
ENV MAX_DETECTIONS=20

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt

RUN pip install \
    --no-cache-dir \
    -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY backend/app.py ./app.py

# Somente o modelo ONNX entra na imagem final.
# model.pt e o código YOLOv7 ficam exclusivamente no builder.
COPY --from=builder /build/model.onnx ./model.onnx

EXPOSE 10000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
