FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/GreenSorter ./GreenSorter
COPY backend/app.py backend/download_model.py ./

RUN python download_model.py

ENV DEVICE=cpu
ENV CONFIDENCE=0.35
ENV IOU=0.45
ENV IMG_SIZE=512

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1
