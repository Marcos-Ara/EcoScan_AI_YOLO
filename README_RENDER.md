# EcoScan AI — Backend Render + GreenSorter

Este pacote está preparado para colocar a API YOLOv7 online no Render.

## Deploy pelo Render

1. Suba `backend/` e `render.yaml` para o seu repositório.
2. No Render, escolha **New → Blueprint**.
3. Selecione o repositório `Marcos-Ara/EcoScan_AI_YOLO_GreenSorter`.
4. O `render.yaml` criará `ecoscan-yolo-api`.
5. Aguarde o build.
6. Teste:
   `https://SEU-SERVICO.onrender.com/health`

Deve retornar `model_loaded: true`.

## Frontend

No `config.js` do EcoScan Pages:

```js
window.ECOSCAN_API_BASE = 'https://ecoscan-yolo-api.onrender.com';
```

Use a URL real mostrada pelo Render.

## Classes atuais

O GreenSorter original tem:

- cardboard → Papel 🔵
- metal → Metal 🟡
- rigid_plastic → Plástico 🔴
- soft_plastic → Plástico 🔴

Vidro, orgânico e rejeito estão preparados no código, mas NÃO são detectados por estes pesos. Para isso será necessário fine-tuning.

## Importante sobre desempenho

O Render Free é adequado para validar o funcionamento, mas YOLOv7 + PyTorch em CPU pode ser lento para câmera em tempo real. Depois do teste podemos otimizar o modelo ou usar uma máquina com CPU/GPU mais adequada.

## Fonte do modelo

O modelo é baixado automaticamente no build do Render a partir do release indicado pelo README do GreenSorter.
