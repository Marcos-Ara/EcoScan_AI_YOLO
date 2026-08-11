# EcoScan AI — Base completa auditada

Esta é a base preparada para o fluxo:

```text
Celular
  ↓ HTTPS
GitHub Pages
  ↓ HTTPS
Render
  ↓
FastAPI
  ↓
GreenSorter YOLOv7
  ↓
Papel / Plástico / Metal
```

## Auditoria feita

### Frontend
- câmera independente do backend;
- funciona em HTTPS/GitHub Pages;
- API configurável por `config.js`;
- detector não fica disparando requisições em alta frequência quando a API falha;
- frames enviados em largura reduzida;
- tratamento de câmera/permissão;
- canvas de detecção;
- histórico, estatísticas e interface preservados.

### Backend
- FastAPI;
- `/`;
- `/health`;
- `/predict`;
- CORS;
- validação de imagem;
- limite de tamanho do upload;
- YOLOv7 GreenSorter;
- `model.pt` baixado automaticamente;
- CPU;
- um único worker para não carregar o modelo várias vezes.

### Deploy
- `render.yaml`;
- `Dockerfile` como alternativa;
- Python 3.11.11 fixado;
- `.gitignore` para não enviar `model.pt`;
- modelo oficial baixado no build.

## Classes dos pesos atuais

Os pesos oficiais do GreenSorter são para quatro classes:

| Classe | EcoScan | Lixeira |
|---|---|---|
| cardboard | Papel | 🔵 Azul |
| rigid_plastic | Plástico | 🔴 Vermelha |
| soft_plastic | Plástico | 🔴 Vermelha |
| metal | Metal | 🟡 Amarela |

Ainda NÃO existe detecção real para:

- vidro 🟢;
- orgânico 🟤;
- rejeito ⚫.

Essas três serão adicionadas por fine-tuning depois. Não serão falsamente mapeadas.

## Modelo

O release oficial do GreenSorter `v0.1` contém `model.pt`, com 74,768,207 bytes. O build do Render baixa esse arquivo automaticamente.

## IMPORTANTE: Render

O plano Free do Render tem 512 MB RAM e 0,1 CPU. YOLOv7 + PyTorch é pesado demais para tratar o Free como configuração confiável.

Por isso o `render.yaml` usa:

```yaml
plan: standard
```

A configuração Standard tem 2 GB RAM e 1 CPU.

O Free pode ser usado apenas como tentativa de teste, mas não é a configuração que devemos considerar "pronta para funcionar".

## Passo 1 — GitHub

Envie TODOS os arquivos desta pasta para:

`Marcos-Ara/EcoScan_AI_YOLO_GreenSorter`

Não envie um arquivo ZIP dentro do repositório. O conteúdo do ZIP deve ser a raiz do projeto.

Depois confira se o GitHub mostra:

```text
index.html
script.js
styles.css
config.js
render.yaml
Dockerfile
backend/
```

## Passo 2 — Render

No Render:

1. New → Blueprint.
2. Conecte o GitHub.
3. Escolha `Marcos-Ara/EcoScan_AI_YOLO_GreenSorter`.
4. O Render deve ler `render.yaml`.
5. Confirme o serviço `ecoscan-yolo-api`.
6. Faça o deploy.
7. Aguarde o build baixar `model.pt`.
8. Abra:

`https://SEU-ENDERECO.onrender.com/health`

O resultado esperado é:

```json
{
  "ok": true,
  "model_loaded": true,
  "model": "GreenSorter YOLOv7"
}
```

## Passo 3 — Frontend

Depois de saber a URL REAL do Render, altere somente:

```js
window.ECOSCAN_API_BASE = 'https://SEU-ENDERECO.onrender.com';
```

em `config.js`.

Faça commit/push.

## Passo 4 — GitHub Pages

Abra o GitHub Pages.

O navegador precisa estar em HTTPS para usar a câmera.

Ao abrir a câmera:

```text
📷 Câmera ativa • 🤖 YOLO conectado
```

Se aparecer:

```text
📷 Câmera ativa • ⚠️ detector offline
```

a câmera está correta; o problema passa a ser exclusivamente a API/URL/Render.

## Passo 5 — Teste do backend

Antes de testar a câmera, sempre teste:

```text
GET /health
```

Depois:

```text
GET /docs
```

E por último:

```text
POST /predict
```

## Ordem que devemos seguir

Não pule etapas:

1. colocar a base no GitHub;
2. conferir arquivos no GitHub;
3. criar o serviço no Render;
4. aguardar build;
5. testar `/health`;
6. testar `/docs`;
7. configurar `config.js`;
8. atualizar GitHub Pages;
9. abrir no celular;
10. testar câmera;
11. testar uma caixa/papel;
12. testar plástico;
13. testar metal;
14. medir velocidade;
15. otimizar;
16. só depois fazer fine-tuning para vidro, orgânico e rejeito.

## Observação sobre velocidade

O backend está preparado para validar o funcionamento primeiro. YOLOv7 em CPU e hospedagem comum não é uma solução ideal para 7 frames por segundo.

Por isso o frontend começa em aproximadamente 1 detecção por segundo e envia frames de até 512 px. Depois que a cadeia inteira estiver funcionando, podemos otimizar a inferência sem quebrar a câmera.

## Segurança

Durante o primeiro teste:

```text
ALLOWED_ORIGIN=*
```

Depois que o GitHub Pages estiver confirmado, podemos trocar para o domínio exato do Pages e restringir o CORS.

## O que NÃO deve entrar no GitHub

```text
backend/model.pt
```

O `.gitignore` já impede isso.

O modelo é baixado pelo Render durante o build.
