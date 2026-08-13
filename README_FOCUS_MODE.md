# EcoScan AI — Modo Foco / Um Objeto

Esta versão foi preparada para testar o GreenSorter em um cenário de **um objeto por vez**.

## O que mudou

- O backend usa uma área central da imagem por padrão (`FOCUS_MODE=true`).
- O backend retorna no máximo uma detecção (`FOCUS_MAX_DETECTIONS=1`).
- O frontend desenha no máximo uma caixa (`MAX_BOXES=1`).
- O mapeamento de classes continua sendo `cardboard -> Papel`, `metal -> Metal`, `rigid_plastic -> Plástico` e `soft_plastic -> Plástico`.
- O build continua convertendo `model.pt` para ONNX no Render; os arquivos binários gerados não precisam ficar versionados no pacote-fonte.

## Limitação importante

O modo foco reduz falsos positivos do fundo, mas **não treina novamente o modelo**. O GreenSorter original possui quatro classes: `cardboard`, `metal`, `rigid_plastic` e `soft_plastic`. Se um objeto metálico continuar sendo classificado como `cardboard` mesmo isolado e centralizado, a limitação estará no peso do modelo/dados de treinamento e não no Docker ou no frontend.

## Render

Não é necessário adicionar PyTorch ao runtime. O `Dockerfile` continua com dois estágios: build com YOLOv7/PyTorch e runtime com FastAPI + ONNX Runtime.
