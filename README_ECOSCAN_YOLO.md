# EcoScan AI — YOLO + GreenSorter

## O que foi alterado

O projeto deixou de usar COCO-SSD para classificação de resíduos. Agora:

1. A câmera continua sendo aberta pelo navegador.
2. O navegador reduz cada frame para até 320 px e envia JPEG para a API local.
3. O build converte o `model.pt` do GreenSorter/YOLOv7 para ONNX. A API carrega `model.onnx` uma única vez.
4. O YOLO detecta `cardboard`, `metal`, `rigid_plastic` e `soft_plastic`.
5. O backend converte essas classes para:
   - cardboard -> Papel -> 🔵 Azul
   - metal -> Metal -> 🟡 Amarela
   - rigid_plastic -> Plástico -> 🔴 Vermelha
   - soft_plastic -> Plástico -> 🔴 Vermelha
6. O frontend desenha as caixas e mostra confiança, material e lixeira.
7. Histórico, estatísticas, tema e câmera continuam no frontend.

## IMPORTANTE sobre as 6 categorias

O GreenSorter original possui 4 classes. O próprio README do projeto informa:
`cardboard`, `metal`, `rigid_plastic`, `soft_plastic`.

Portanto este pacote NÃO inventa detecção de vidro, orgânico ou rejeito.

As regras dessas três categorias já estão preparadas no código para a próxima fase de fine-tuning.

## Instalação

### 1. Modelo

Entre na pasta `backend` e execute:

```powershell
python download_model.py
```

O projeto baixa automaticamente o `model.pt` oficial do release `v0.1` e valida se o arquivo recebido é um checkpoint PyTorch válido.

O arquivo fica em:

```text
backend/model.pt
```

### 2. Backend

```powershell
cd backend
.\start_backend.ps1
```

Teste:

```text
http://127.0.0.1:8000/health
```

Deve aparecer:

```json
{
  "ok": true,
  "model_loaded": true
}
```

### 3. Frontend

Em outro PowerShell:

```powershell
cd ..
.\start_frontend.ps1
```

Abra:

```text
http://127.0.0.1:5500
```

Não abra o `index.html` diretamente com `file://`, porque o acesso à câmera e a API local ficam mais problemáticos.

## Erros corrigidos

### 1. COCO-SSD classificava objetos, não materiais
O código antigo tinha regras como `bottle -> Vidro`, mas uma garrafa pode ser de plástico, vidro ou outro material. Isso foi removido.

### 2. `can` e outras classes não eram confiáveis no COCO-SSD
O mapeamento antigo dependia de nomes de objetos que não representavam corretamente o material.

### 3. Câmera e IA estavam acopladas ao COCO-SSD
Agora o detector é uma API ONNX local/remota, usando os pesos treinados do GreenSorter.

### 4. Loop de detecção sem controle de rede
Agora o frontend controla o intervalo e envia frames reduzidos, evitando mandar 1080p/30 FPS para o backend.

### 5. O modelo não é recarregado a cada frame
O backend carrega `model.onnx` uma única vez no startup.

### 6. Escala das caixas
O backend recebe o frame reduzido, mas devolve as caixas e o frontend converte para a resolução do vídeo antes de desenhar.

## Próxima fase: completar as 6 classes

Quando o teste das quatro classes estiver funcionando, a próxima etapa é fine-tuning do GreenSorter para:

```text
Papel
Plástico
Vidro
Metal
Orgânico
Rejeito
```

Não basta criar um `if` no JavaScript para dizer que determinada imagem é vidro/orgânico/rejeito. Essas classes precisam existir nos pesos treinados.

O dataset original do GreenSorter também está no repositório em `train`, `valid` e `test`, o que pode servir como base para essa etapa.
