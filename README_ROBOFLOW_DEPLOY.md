# EcoScan AI + Roboflow YOLO11

## Modelo

```text
waste-sorting-smyr8/2
```

## Instalação

O runtime usa o SDK oficial HTTP do Roboflow:

```bash
pip install inference-sdk
```

A documentação oficial do Roboflow mostra `InferenceHTTPClient` com `https://serverless.roboflow.com` e recomenda carregar a chave por variável de ambiente. ([https://inference.roboflow.com/inference_helpers/inference_sdk/](https://inference.roboflow.com/inference_helpers/inference_sdk/))

## Render

Crie uma variável Secret:

```text
ROBOFLOW_API_KEY
```

E mantenha:

```text
ROBOFLOW_API_URL=https://serverless.roboflow.com
ROBOFLOW_MODEL_ID=waste-sorting-smyr8/2
```

## Nunca

Não coloque uma chave real em:

- `config.js`
- `script.js`
- GitHub
- ZIP público
- HTML

## API

O frontend chama `POST /predict`.

O backend recebe a imagem, usa o modo foco, executa o modelo remoto e devolve as previsões no formato que o frontend já entende.
