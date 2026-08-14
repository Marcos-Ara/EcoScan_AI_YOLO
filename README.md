# EcoScan AI

Aplicação web de classificação/detecção de resíduos com câmera, autenticação Firebase e API FastAPI.

## Produção atual

O EcoScan usa o modelo Roboflow:

`waste-sorting-smyr8/2`

O frontend está no GitHub Pages e o backend no Render. O backend chama o Roboflow Serverless usando `inference-sdk` e mantém o segredo da API somente no ambiente do Render.

## Fluxo

```text
GitHub Pages
   ↓
FastAPI no Render
   ↓
Roboflow Serverless
   ↓
YOLO11 / waste-sorting-smyr8/2
```

## Segurança

A variável `ROBOFLOW_API_KEY` nunca deve ir para `config.js` ou para o repositório.

Se uma chave real foi exposta em algum lugar, revogue-a no Roboflow e gere uma nova.

## Deploy

Consulte `README_RENDER.md` e `README_ROBOFLOW_DEPLOY.md`.
