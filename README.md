# EcoScan AI — GreenSorter YOLOv7 + ONNX

Base do EcoScan AI com câmera no navegador e detecção de resíduos usando os pesos oficiais do GreenSorter.

## Classes reais do modelo

O modelo usado neste projeto possui quatro classes:

| Classe GreenSorter | Categoria EcoScan | Lixeira |
|---|---|---|
| `cardboard` | Papel | 🔵 Azul |
| `metal` | Metal | 🟡 Amarela |
| `rigid_plastic` | Plástico | 🔴 Vermelha |
| `soft_plastic` | Plástico | 🔴 Vermelha |

Vidro, orgânico e rejeito aparecem apenas como regras preparadas no frontend/backend; eles **não são classes detectadas pelos pesos atuais**.

## Estrutura

```text
EcoScan_AI_YOLO_GreenSorter/
├── index.html
├── script.js
├── styles.css
├── config.js
├── Dockerfile
├── render.yaml
├── pyrightconfig.json
└── backend/
    ├── app.py
    ├── convert_model.py
    ├── download_model.py
    ├── requirements.txt
    ├── requirements-build.txt
    ├── requirements-local.txt
    ├── start_backend.ps1
    ├── model.pt
    ├── model.onnx
    └── GreenSorter/
        └── yolov7/
            ├── models/
            └── utils/
```

## 1. Ambiente local

O projeto usa Python 3.11.

Se o `.venv` estiver na raiz do projeto:

```powershell
.\.venv\Scripts\Activate.ps1
```

Dentro de `backend`, com o ambiente ativado:

```powershell
python -m pip install -r requirements-local.txt
python download_model.py
python convert_model.py
```

> Se o PowerShell estiver em `backend` e o `.venv` estiver na pasta pai, o caminho explícito é `..\.venv\Scripts\python.exe`.

### Dependências

- `requirements-local.txt`: ambiente local completo, incluindo PyTorch/YOLOv7 para conversão.
- `requirements-build.txt`: dependências usadas no estágio de build do Docker.
- `requirements.txt`: somente runtime da API; não instala PyTorch.

## 2. Verificar o modelo

O `model.pt` oficial tem aproximadamente 71,3 MiB / 74.768.207 bytes.

Para validar:

```powershell
python download_model.py
```

Depois:

```powershell
python -c "import sys; sys.path.insert(0, 'GreenSorter/yolov7'); import torch; m=torch.load('model.pt', map_location='cpu', weights_only=False); print('MODELO OK'); print(type(m))"
```

O resultado esperado começa com:

```text
MODELO OK
<class 'dict'>
```

## 3. Converter para ONNX

Execute:

```powershell
python convert_model.py
```

A conversão agora verifica:

- existência e assinatura do `model.pt`;
- import do YOLOv7;
- carregamento do checkpoint;
- configuração correta da camada Detect;
- dry-run com saída `[1, N, 9]`;
- exportação ONNX para arquivo temporário;
- validação estrutural com `onnx`;
- carregamento pelo ONNX Runtime;
- inferência real de teste;
- publicação de `model.onnx` somente depois das validações.

### Importante sobre `models.experimental`

O código correto é:

```python
from models.experimental import attempt_load
```

O YOLOv7 deste projeto usa imports absolutos (`models.*` e `utils.*`). O `convert_model.py` coloca `backend/GreenSorter/yolov7` no `sys.path` antes do import e o `pyrightconfig.json` informa esse caminho ao Pylance.

O aviso amarelo do Pylance é diferente de um erro de execução. Depois de recarregar o Language Server, o import deve deixar de aparecer como `reportMissingImports`.

## 4. Rodar a API

A forma mais simples é:

```powershell
cd backend
.\start_backend.ps1
```

O script:

1. cria o `.venv` do backend se necessário;
2. instala as dependências;
3. valida/baixa `model.pt`;
4. converte para `model.onnx` se necessário;
5. inicia o FastAPI em `http://127.0.0.1:8000`.

Teste:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

## 5. Frontend

Em outro terminal:

```powershell
.\start_frontend.ps1
```

Abra:

```text
http://127.0.0.1:5500
```

Para produção, configure em `config.js` a URL HTTPS real da API do Render.

## 6. Docker / Render

O `Dockerfile` faz a conversão no estágio builder e deixa somente o ONNX no runtime.

```text
builder
  ├── model.pt
  ├── PyTorch
  └── YOLOv7
       ↓
   model.onnx
       ↓
runtime
  ├── FastAPI
  ├── OpenCV
  ├── NumPy
  └── ONNX Runtime
```

Isso evita carregar PyTorch/YOLOv7 no servidor final.

## 7. Diagnóstico rápido

### `Import "models.experimental" could not be resolved`

No VS Code:

```text
Ctrl + Shift + P
→ Python: Restart Language Server
```

O projeto já contém `pyrightconfig.json` com o caminho correto.

### `Modelo não encontrado: ...\backend\model.pt`

Execute:

```powershell
python download_model.py
```

### `_pickle.UnpicklingError: could not find MARK`

Isso significa que `model.pt` não era um checkpoint PyTorch válido. Se o arquivo começar com texto como `import os`, ele está errado. Execute novamente:

```powershell
python download_model.py
```

### `ModuleNotFoundError: No module named 'models'`

Não execute `torch.load('model.pt')` puro fora da raiz do YOLOv7. Para teste manual:

```powershell
python -c "import sys; sys.path.insert(0, 'GreenSorter/yolov7'); import torch; m=torch.load('model.pt', map_location='cpu', weights_only=False); print('MODELO OK')"
```

O `convert_model.py` já faz isso automaticamente.
