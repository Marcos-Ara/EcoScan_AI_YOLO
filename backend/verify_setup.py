import os
import sys
from pathlib import Path

# ============================================================
# EcoScan AI - Verificação completa do ambiente
# ============================================================

ROOT = Path(__file__).resolve().parent
YOLOV7_DIR = ROOT / "GreenSorter" / "yolov7"
MODEL_PATH = ROOT / "model.pt"
ONNX_PATH = ROOT / "model.onnx"


def ok(message: str):
    print(f"[OK] {message}")


def fail(message: str):
    print(f"[ERRO] {message}")
    raise SystemExit(1)


def main():
    print("=" * 60)
    print("EcoScan AI - VERIFICAÇÃO COMPLETA")
    print("=" * 60)
    print(f"Python: {sys.executable}")
    print(f"Versão: {sys.version.split()[0]}")

    if not YOLOV7_DIR.is_dir():
        fail(f"YOLOv7 não encontrado: {YOLOV7_DIR}")
    ok("Pasta YOLOv7 encontrada")

    yolo_path = str(YOLOV7_DIR)
    if yolo_path not in sys.path:
        sys.path.insert(0, yolo_path)

    try:
        import numpy as np
        ok(f"NumPy {np.__version__}")
    except Exception as exc:
        fail(f"NumPy: {exc}")

    try:
        import torch
        ok(f"PyTorch {torch.__version__}")
    except Exception as exc:
        fail(f"PyTorch: {exc}")

    try:
        import onnx
        ok(f"ONNX {onnx.__version__}")
    except Exception as exc:
        fail(f"ONNX: {exc}")

    try:
        import onnxruntime as ort
        ok(f"ONNX Runtime {ort.__version__}")
    except Exception as exc:
        fail(f"ONNX Runtime: {exc}")

    try:
        from models.experimental import attempt_load
        ok("Import models.experimental.attempt_load")
    except Exception as exc:
        fail(f"Import models.experimental: {exc}")

    if not MODEL_PATH.is_file():
        fail(f"model.pt não encontrado: {MODEL_PATH}")

    if MODEL_PATH.stat().st_size < 1_000_000:
        fail(f"model.pt parece inválido: {MODEL_PATH.stat().st_size} bytes")

    with MODEL_PATH.open("rb") as handle:
        signature = handle.read(4)

    if signature != b"PK\x03\x04":
        fail("model.pt não possui a assinatura esperada de checkpoint ZIP")
    ok(f"model.pt válido ({MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB)")

    try:
        checkpoint = torch.load(
            str(MODEL_PATH),
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            fail("model.pt foi lido, mas não contém a chave 'model'")
        del checkpoint
        ok("torch.load(model.pt) OK")
    except Exception as exc:
        fail(f"torch.load(model.pt): {type(exc).__name__}: {exc}")

    if not ONNX_PATH.is_file():
        fail(f"model.onnx não encontrado: {ONNX_PATH}")

    try:
        onnx_model = onnx.load(str(ONNX_PATH), load_external_data=False)
        onnx.checker.check_model(onnx_model, full_check=False)
        ok("model.onnx passou no onnx.checker")
    except Exception as exc:
        fail(f"Validação ONNX: {type(exc).__name__}: {exc}")

    try:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.enable_mem_pattern = False
        opts.enable_cpu_mem_arena = False

        session = ort.InferenceSession(
            str(ONNX_PATH),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        inputs = session.get_inputs()
        outputs = session.get_outputs()

        if not inputs:
            fail("model.onnx não possui entrada")
        if not outputs:
            fail("model.onnx não possui saída")

        input_meta = inputs[0]
        output_meta = outputs[0]

        print(f"[INFO] Input: {input_meta.name} {input_meta.shape}")
        print(f"[INFO] Output: {output_meta.name} {output_meta.shape}")

        img_size = int(os.getenv("IMG_SIZE", "224"))
        dummy = np.zeros(
            (1, 3, img_size, img_size),
            dtype=np.float32,
        )

        result = session.run(
            [output_meta.name],
            {input_meta.name: dummy},
        )[0]

        if result.ndim != 3 or result.shape[0] != 1 or result.shape[-1] != 9:
            fail(f"Saída de inferência inesperada: {result.shape}; esperado [1,N,9]")

        ok(f"ONNX Runtime inferiu corretamente: {result.shape}")

    except SystemExit:
        raise
    except Exception as exc:
        fail(f"ONNX Runtime: {type(exc).__name__}: {exc}")

    print("=" * 60)
    print("[SUCESSO] Ambiente EcoScan AI está pronto.")
    print("=" * 60)


if __name__ == "__main__":
    main()
