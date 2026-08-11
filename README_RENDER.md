# EcoScan AI — versão leve para Render Free

Esta versão mantém o GreenSorter, mas muda a arquitetura de produção:

- `model.pt` é baixado somente no estágio de build.
- PyTorch/YOLOv7 são usados somente para converter o modelo.
- O estágio final contém apenas `model.onnx`, FastAPI, OpenCV e ONNX Runtime.
- `IMG_SIZE=224`.
- Uma única inferência por vez.
- Limite de arquivo e resolução.
- Timeout de lock para evitar requisições presas.
- Limpeza de objetos temporários com `gc`.
- Frontend envia frames menores e mais espaçados.
- O frontend também possui **Selecionar imagem** para testar uma foto diretamente.

## Deploy

O `render.yaml` já está configurado para `plan: free`.

Faça push de todos os arquivos:

```bash
git add .
git commit -m "feat: runtime ONNX leve para Render Free"
git push
```

No Render, confirme que não existe uma variável externa `IMG_SIZE=320` ou `IMG_SIZE=512`. O projeto usa `224`.

## O que esperar do primeiro build

O build instala PyTorch apenas no estágio temporário e converte:

`model.pt` → `model.onnx`

O container final NÃO instala PyTorch.

No log do serviço, o runtime deve mostrar:

```text
[EcoScan] Runtime: CPU / 1 thread
[EcoScan] ONNX carregado
```

E `/health` deve retornar:

```json
{
  "ok": true,
  "model_loaded": true,
  "model": "GreenSorter ONNX",
  "runtime": "onnxruntime",
  "device": "cpu",
  "img_size": 224
}
```

## Teste

Abra `/docs` e use `POST /predict` com uma imagem.

Depois teste o frontend. Na tela de Scan, além da câmera, existe o botão **Selecionar imagem**.
