# EcoScan AI — YOLO11 via Roboflow

## Arquitetura atual

```text
Câmera
  ↓
Frontend GitHub Pages
  ↓
POST /predict
  ↓
Render / FastAPI
  ↓
FOCUS crop central
  ↓
Roboflow Serverless
  ↓
waste-sorting-smyr8/2
  ↓
Normalização das classes
  ↓
Card EcoScan
```

## Classes do modelo

O frontend/backend entende as classes principais do modelo como:

```text
paper      -> Papel      -> 🔵 Azul
cardboard  -> Papel      -> 🔵 Azul
plastic    -> Plástico   -> 🔴 Vermelha
glass      -> Vidro      -> 🟢 Verde
metal      -> Metal      -> 🟡 Amarela
```

## Por que o frontend não chama o Roboflow diretamente?

A chave do Roboflow é secreta e não deve ser colocada no JavaScript publicado no GitHub Pages. Por isso o browser chama o Render e o Render chama o Roboflow.

## Modelo de resposta

O backend preserva `predictions`, `category`, `category_key`, `score` e `bbox` para não quebrar o frontend existente.

## Modelo antigo

O GreenSorter YOLOv7 não é mais usado em produção. Os arquivos antigos continuam na base apenas para histórico e testes locais.
