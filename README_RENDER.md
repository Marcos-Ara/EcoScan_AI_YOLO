# EcoScan AI — Deploy com Roboflow YOLO11

A arquitetura de produção agora é:

```text
GitHub Pages
    ↓ HTTPS
Frontend EcoScan
    ↓ HTTPS /predict
Render
    ↓
FastAPI + OpenCV + inference-sdk
    ↓ HTTPS
Roboflow Serverless
    ↓
YOLO11 — waste-sorting-smyr8/2
```

## Modelo

O backend usa o modelo:

```text
waste-sorting-smyr8/2
```

O modelo é hospedado pelo Roboflow. O Render não instala PyTorch, YOLOv7 ou ONNX Runtime.

## Variável obrigatória no Render

Crie uma variável **Secret**:

```text
ROBOFLOW_API_KEY
```

Também são configuradas:

```text
ROBOFLOW_API_URL=https://serverless.roboflow.com
ROBOFLOW_MODEL_ID=waste-sorting-smyr8/2
CONFIDENCE=0.40
IOU=0.45
MAX_DETECTIONS=1
FOCUS_MODE=true
FOCUS_CROP_RATIO=0.90
FOCUS_MAX_DETECTIONS=1
ROBOFLOW_TIMEOUT_SECONDS=30
ALLOWED_ORIGIN=https://marcos-ara.github.io
```

## Segurança

A chave do Roboflow não pode ficar em `config.js`, `script.js`, GitHub ou qualquer arquivo público.
Se uma chave real já foi exposta, revogue-a e gere outra antes de usar o ambiente de produção.

## Endpoint

O frontend continua chamando:

```text
POST /predict
```

com o campo multipart:

```text
file
```

A resposta mantém a estrutura esperada pelo frontend:

```json
{
  "predictions": [
    {
      "source_class": "metal",
      "class_name": "Metal",
      "category": "Metal",
      "category_key": "metal",
      "bin": "🟡 Amarela",
      "destination": "Reciclagem",
      "decomposition": "Varia por material",
      "score": 0.91,
      "bbox": [10, 20, 300, 400]
    }
  ]
}
```

## Modo foco

Por padrão o backend recorta a região central da imagem e retorna apenas a melhor detecção. Isso preserva a melhoria já obtida no EcoScan de trabalhar com um objeto por vez.

## Local

No PowerShell:

```powershell
cd backend
$env:ROBOFLOW_API_KEY="SUA_CHAVE_NOVA"
$env:ROBOFLOW_MODEL_ID="waste-sorting-smyr8/2"
.\start_backend.ps1
```

Teste:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

## Observação

Os arquivos antigos `convert_model.py`, `download_model.py` e `GreenSorter/yolov7` permanecem na base para histórico/local, mas não participam mais do Docker de produção.
