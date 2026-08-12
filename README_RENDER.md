# EcoScan AI — Deploy com ONNX

A arquitetura de produção é:

```text
GitHub Pages
    ↓ HTTPS
Frontend EcoScan
    ↓ HTTPS /predict
Render
    ↓
FastAPI + ONNX Runtime
    ↓
GreenSorter YOLOv7 convertido para ONNX
```

## O que acontece no build

O Docker usa duas etapas:

1. **Builder**: instala PyTorch + YOLOv7, baixa `model.pt` e executa `convert_model.py`.
2. **Runtime**: instala apenas FastAPI, OpenCV, NumPy e ONNX Runtime e recebe somente `model.onnx`.

O runtime não carrega PyTorch nem o código YOLOv7.

## Conversão local

Dentro de `backend`:

```powershell
python download_model.py
python convert_model.py
```

Se o ambiente virtual estiver na raiz do projeto e você estiver dentro de `backend`, também pode usar:

```powershell
& "..\.venv\Scripts\python.exe" download_model.py
& "..\.venv\Scripts\python.exe" convert_model.py
```

A conversão deve terminar com:

```text
[EcoScan] Dry-run OK: (1, N, 9)
[EcoScan] ONNX Runtime OK: inferência de teste concluída.
[EcoScan] CONVERSÃO CONCLUÍDA COM SUCESSO
```

O `N` depende do tamanho da entrada. Com `IMG_SIZE=224`, o YOLOv7 GreenSorter normalmente produz 3087 candidatos.

## Erro `Import "models..."` no VS Code

O YOLOv7 original usa imports absolutos como `models.*` e `utils.*`. O projeto mantém esse comportamento porque o checkpoint também depende desses nomes.

O arquivo `pyrightconfig.json` já informa ao Pylance que:

```text
backend/GreenSorter/yolov7
```

é um caminho de importação.

Se o aviso continuar no VS Code, faça:

1. `Ctrl + Shift + P`
2. `Python: Restart Language Server`
3. ou `Developer: Reload Window`

O import correto continua sendo:

```python
from models.experimental import attempt_load
```

Não troque para um import relativo do pacote, porque o código original do YOLOv7 usa `models` e `utils` como módulos de topo.

## API

Depois de iniciar:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

`POST /predict` recebe uma imagem no campo `file`.

## Render

O `render.yaml` aponta para o `Dockerfile`. A conversão acontece durante o build, portanto `model.pt` não precisa estar no estágio final do container.

Antes do deploy, confira a URL em `config.js`:

```js
window.ECOSCAN_API_BASE = 'https://SEU-ENDERECO.onrender.com';
```

Não use `localhost` nessa configuração quando o frontend estiver no GitHub Pages.
