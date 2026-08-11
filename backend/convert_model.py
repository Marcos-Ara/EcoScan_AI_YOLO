import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Limitar paralelismo durante a conversão.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

ROOT = Path(__file__).resolve().parent
YOLOV7_DIR = ROOT / "GreenSorter" / "yolov7"
MODEL_PATH = ROOT / "model.pt"
ONNX_PATH = ROOT / "model.onnx"

IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))

sys.path.insert(0, str(YOLOV7_DIR))

from models.experimental import attempt_load


def main():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modelo não encontrado: {MODEL_PATH}")

    print("=" * 60, flush=True)
    print("[EcoScan] Convertendo GreenSorter YOLOv7 -> ONNX", flush=True)
    print(f"[EcoScan] Pesos: {MODEL_PATH}", flush=True)
    print(f"[EcoScan] Saída: {ONNX_PATH}", flush=True)
    print(f"[EcoScan] IMG_SIZE: {IMG_SIZE}", flush=True)
    print("=" * 60, flush=True)

    device = torch.device("cpu")

    print("[EcoScan] Carregando pesos para conversão...", flush=True)
    model = attempt_load(
        str(MODEL_PATH),
        map_location=device,
    )

    model.eval()

    # Fusão é feita SOMENTE no build. O runtime não carrega PyTorch.
    try:
        model.fuse()
        print("[EcoScan] Fuse concluído.", flush=True)
    except Exception as exc:
        print(
            f"[EcoScan] Aviso: fuse não foi aplicado: {exc!r}",
            flush=True,
        )

    # Export do Detect em modo de grade + concatenação.
    detect = model.model[-1]
    detect.export = False
    detect.include_nms = False
    detect.end2end = False
    detect.concat = True

    # Ativa modo compatível com export ONNX para SiLU/Hardswish.
    dummy = torch.zeros(
        1,
        3,
        IMG_SIZE,
        IMG_SIZE,
        dtype=torch.float32,
        device=device,
    )

    with torch.inference_mode():
        dry = model(dummy)

    if isinstance(dry, (tuple, list)):
        dry_shape = [
            tuple(x.shape) for x in dry
            if hasattr(x, "shape")
        ]
    else:
        dry_shape = tuple(dry.shape)

    print(
        f"[EcoScan] Saída do dry-run: {dry_shape}",
        flush=True,
    )

    print("[EcoScan] Exportando ONNX...", flush=True)

    # O GreenSorter usa 4 classes -> saída [1, N, 9].
    torch.onnx.export(
        model,
        dummy,
        str(ONNX_PATH),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes=None,
    )

    del dry
    del dummy
    del model
    gc.collect()

    if not ONNX_PATH.exists():
        raise RuntimeError("O arquivo model.onnx não foi criado.")

    size_mb = ONNX_PATH.stat().st_size / 1024 / 1024

    if size_mb < 1:
        raise RuntimeError(
            "model.onnx foi criado, mas parece inválido."
        )

    # Validação estrutural.
    import onnx

    print("[EcoScan] Validando ONNX...", flush=True)

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    print(
        f"[EcoScan] ONNX válido. Tamanho: {size_mb:.1f} MB",
        flush=True,
    )

    # Teste real com ONNX Runtime no build.
    import onnxruntime as ort

    print("[EcoScan] Testando ONNX Runtime...", flush=True)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

    session = ort.InferenceSession(
        str(ONNX_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    dummy_np = np.zeros(
        (1, 3, IMG_SIZE, IMG_SIZE),
        dtype=np.float32,
    )

    output = session.run(
        None,
        {input_name: dummy_np},
    )[0]

    print(
        f"[EcoScan] Teste ONNX Runtime OK: {output.shape}",
        flush=True,
    )

    expected_columns = 5 + 4

    if output.ndim != 3:
        raise RuntimeError(
            f"Saída ONNX inesperada: {output.shape}"
        )

    if (
        output.shape[-1] != expected_columns
        and output.shape[1] != expected_columns
    ):
        raise RuntimeError(
            "Saída ONNX não corresponde ao GreenSorter de 4 classes: "
            f"{output.shape}"
        )

    print("[EcoScan] Conversão concluída com sucesso.", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
