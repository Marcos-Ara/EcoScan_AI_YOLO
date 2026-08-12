import gc
import os
import sys
from pathlib import Path

# ============================================================
# EcoScan AI - YOLOv7 -> ONNX
# Conversor robusto para CPU / Render / ONNX Runtime
# ============================================================

# Limitar threads antes de importar NumPy/PyTorch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    # O PyTorch pode já ter inicializado o pool de threads.
    pass


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
YOLOV7_DIR = ROOT / "GreenSorter" / "yolov7"
MODEL_PATH = ROOT / "model.pt"
ONNX_PATH = ROOT / "model.onnx"
ONNX_TEMP_PATH = ROOT / "model.onnx.tmp"


# ============================================================
# CONFIGURAÇÃO
# ============================================================

try:
    IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
except ValueError:
    IMG_SIZE = 224

# O modelo GreenSorter foi preparado para trabalhar nesta faixa.
IMG_SIZE = max(128, min(IMG_SIZE, 224))


# ============================================================
# IMPORTAR YOLOV7
# ============================================================


def configure_yolov7_imports() -> None:
    """Coloca a raiz do YOLOv7 no sys.path antes do import.

    O código original do YOLOv7 usa imports absolutos como
    `models.*` e `utils.*`. Por isso não basta importar como
    `GreenSorter.yolov7.models.*`: a raiz `yolov7` também precisa
    estar disponível no caminho de módulos.
    """

    if not YOLOV7_DIR.is_dir():
        raise RuntimeError(
            "Diretório YOLOv7 não encontrado: "
            f"{YOLOV7_DIR}"
        )

    yolo_path = str(YOLOV7_DIR)
    if yolo_path not in sys.path:
        sys.path.insert(0, yolo_path)


configure_yolov7_imports()

# IMPORTANTE:
# O Pylance encontra este módulo através do pyrightconfig.json
# existente na raiz do projeto. Em execução, o sys.path acima
# resolve os imports absolutos usados pelo YOLOv7.
from models.experimental import attempt_load


# ============================================================
# UTILITÁRIOS
# ============================================================


def force_cleanup() -> None:
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def fail(message: str) -> None:
    raise RuntimeError(f"[EcoScan] {message}")


def validate_checkpoint_file() -> None:
    """Impede tentar carregar um model.pt que na verdade seja texto.

    O problema anterior aconteceu exatamente porque model.pt continha
    código Python em vez do checkpoint binário do PyTorch.
    """

    if not MODEL_PATH.is_file():
        fail(f"Modelo não encontrado: {MODEL_PATH}")

    size = MODEL_PATH.stat().st_size
    if size < 1_000_000:
        fail(
            "model.pt é pequeno demais para ser o checkpoint oficial "
            f"({size} bytes). Execute download_model.py novamente."
        )

    try:
        with MODEL_PATH.open("rb") as handle:
            signature = handle.read(4)
    except OSError as exc:
        fail(f"Não foi possível ler model.pt: {exc}")

    # Checkpoints modernos do PyTorch normalmente são ZIPs.
    # O model.pt oficial do GreenSorter usado neste projeto é um ZIP.
    if signature != b"PK\x03\x04":
        fail(
            "model.pt não parece ser um checkpoint PyTorch válido. "
            "Os primeiros bytes não correspondem a um arquivo ZIP do "
            "checkpoint. Execute download_model.py novamente."
        )


def extract_prediction_tensor(value):
    """Obtém o tensor de predição de uma saída do YOLOv7."""

    if isinstance(value, torch.Tensor):
        return value

    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor) and item.ndim >= 2:
                return item

    raise RuntimeError(
        "A saída do YOLOv7 não contém um tensor de predição utilizável. "
        f"Tipo recebido: {type(value).__name__}"
    )


def describe_prediction(value) -> tuple:
    tensor = extract_prediction_tensor(value)
    return tuple(tensor.shape)


# ============================================================
# CONVERSÃO
# ============================================================


def main() -> None:
    print("=" * 60, flush=True)
    print("[EcoScan] CONVERSÃO YOLOv7 -> ONNX", flush=True)
    print("=" * 60, flush=True)

    validate_checkpoint_file()

    size_mb = MODEL_PATH.stat().st_size / 1024 / 1024

    print(f"[EcoScan] Modelo: {MODEL_PATH}", flush=True)
    print(f"[EcoScan] Pesos: {size_mb:.1f} MB", flush=True)
    print(f"[EcoScan] IMG_SIZE: {IMG_SIZE}", flush=True)
    print("[EcoScan] Device: CPU", flush=True)

    device = torch.device("cpu")

    # --------------------------------------------------------
    # CARREGAR MODELO
    # --------------------------------------------------------

    print("[EcoScan] Carregando modelo...", flush=True)

    try:
        model = attempt_load(
            str(MODEL_PATH),
            map_location=device,
        )
    except Exception as exc:
        force_cleanup()
        fail(
            "Falha ao carregar model.pt. "
            "Verifique se o arquivo é o checkpoint oficial do GreenSorter. "
            f"Detalhe: {type(exc).__name__}: {exc}"
        )

    model = model.float().eval()
    print("[EcoScan] Modelo carregado.", flush=True)

    # --------------------------------------------------------
    # FUSE
    # --------------------------------------------------------

    print("[EcoScan] Executando fuse()...", flush=True)
    try:
        model.fuse()
        print("[EcoScan] Fuse concluído.", flush=True)
    except Exception as exc:
        # Fuse é otimização. A conversão continua mesmo sem ela.
        print(
            f"[EcoScan] Aviso: fuse() não aplicado: {exc!r}",
            flush=True,
        )

    # --------------------------------------------------------
    # CONFIGURAR DETECT
    # --------------------------------------------------------

    try:
        detect = model.model[-1]
    except Exception as exc:
        del model
        force_cleanup()
        fail(f"Não foi possível localizar a camada Detect: {exc}")

    # ATENÇÃO:
    # export=True no código YOLOv7 original força `self.training` e pode
    # fazer o dry-run/export gerar os mapas de features em vez de uma
    # saída [1, N, 5+C]. Para o nosso export, precisamos manter eval()
    # e usar concat=True. Durante torch.onnx.export(), o YOLOv7 detecta
    # automaticamente torch.onnx.is_in_onnx_export().
    detect.export = False
    detect.include_nms = False
    detect.end2end = False
    detect.concat = True

    model.eval()

    print(
        "[EcoScan] Detect configurado: export=False, "
        "include_nms=False, end2end=False, concat=True.",
        flush=True,
    )

    # --------------------------------------------------------
    # ENTRADA
    # --------------------------------------------------------

    dummy = torch.zeros(
        1,
        3,
        IMG_SIZE,
        IMG_SIZE,
        dtype=torch.float32,
        device=device,
    )

    print(
        f"[EcoScan] Entrada: {tuple(dummy.shape)}",
        flush=True,
    )

    # --------------------------------------------------------
    # DRY RUN
    # --------------------------------------------------------

    print("[EcoScan] Executando dry-run...", flush=True)

    try:
        with torch.inference_mode():
            dry = model(dummy)

        prediction = extract_prediction_tensor(dry)
        prediction_shape = tuple(prediction.shape)

        if prediction.ndim != 3 or prediction.shape[0] != 1:
            fail(
                "O dry-run não produziu o formato esperado [1, N, 5+C]. "
                f"Recebido: {prediction_shape}"
            )

        # GreenSorter possui 4 classes => 9 valores por detecção.
        expected_values = 5 + 4
        if prediction.shape[-1] != expected_values:
            fail(
                "O número de valores por detecção não bate com as 4 classes "
                f"do GreenSorter. Esperado: {expected_values}; "
                f"recebido: {prediction.shape[-1]}."
            )

        print(
            f"[EcoScan] Dry-run OK: {prediction_shape}",
            flush=True,
        )

    except Exception:
        del dummy
        del model
        force_cleanup()
        raise

    del dry
    force_cleanup()

    # --------------------------------------------------------
    # EXPORTAR ONNX PARA ARQUIVO TEMPORÁRIO
    # --------------------------------------------------------

    print("[EcoScan] Iniciando exportação ONNX...", flush=True)

    ONNX_TEMP_PATH.unlink(missing_ok=True)

    try:
        torch.onnx.export(
            model,
            dummy,
            str(ONNX_TEMP_PATH),
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes=None,
            verbose=False,
        )
    except Exception as exc:
        ONNX_TEMP_PATH.unlink(missing_ok=True)
        del dummy
        del model
        force_cleanup()
        fail(
            "Falha durante a exportação ONNX. "
            f"Detalhe: {type(exc).__name__}: {exc}"
        )

    del dummy
    del model
    force_cleanup()

    if not ONNX_TEMP_PATH.is_file():
        fail("O arquivo temporário model.onnx.tmp não foi criado.")

    temp_size_mb = ONNX_TEMP_PATH.stat().st_size / 1024 / 1024
    print(
        f"[EcoScan] ONNX temporário: {temp_size_mb:.1f} MB",
        flush=True,
    )

    if ONNX_TEMP_PATH.stat().st_size < 1_000_000:
        ONNX_TEMP_PATH.unlink(missing_ok=True)
        fail("O model.onnx gerado parece inválido ou incompleto.")

    # --------------------------------------------------------
    # VALIDAR ONNX + ONNX RUNTIME
    # --------------------------------------------------------

    print("[EcoScan] Validando ONNX...", flush=True)

    try:
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(
            str(ONNX_TEMP_PATH),
            load_external_data=False,
        )
        onnx.checker.check_model(
            onnx_model,
            full_check=False,
        )

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
        session_options.enable_mem_pattern = False
        session_options.enable_cpu_mem_arena = False

        session = ort.InferenceSession(
            str(ONNX_TEMP_PATH),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()

        if input_meta.name != "images":
            fail(
                f"Nome da entrada ONNX inesperado: {input_meta.name!r}"
            )

        if not output_meta:
            fail("O ONNX não possui nenhuma saída.")

        runtime_output = session.run(
            [output_meta[0].name],
            {
                input_meta.name: (
                    __import__("numpy").zeros(
                        (1, 3, IMG_SIZE, IMG_SIZE),
                        dtype="float32",
                    )
                )
            },
        )[0]

        runtime_shape = tuple(runtime_output.shape)

        if runtime_output.ndim != 3 or runtime_output.shape[0] != 1:
            fail(
                "A saída ONNX Runtime não está no formato [1, N, 9]. "
                f"Recebido: {runtime_shape}"
            )

        if runtime_output.shape[-1] != 9:
            fail(
                "A saída ONNX Runtime não possui 9 valores por detecção. "
                f"Recebido: {runtime_shape}"
            )

        print(
            f"[EcoScan] ONNX estruturalmente válido: {runtime_shape}",
            flush=True,
        )
        print(
            "[EcoScan] ONNX Runtime OK: inferência de teste concluída.",
            flush=True,
        )

        del runtime_output
        del session
        del onnx_model
        force_cleanup()

    except ImportError as exc:
        # O build deve ter onnx/onnxruntime. Localmente, não impedimos
        # a conversão se o usuário tiver instalado somente PyTorch.
        print(
            "[EcoScan] AVISO: onnx/onnxruntime não estão disponíveis. "
            f"Validação de runtime ignorada: {exc}",
            flush=True,
        )

    except Exception as exc:
        ONNX_TEMP_PATH.unlink(missing_ok=True)
        force_cleanup()
        fail(
            "A validação do ONNX falhou. O arquivo antigo não será "
            "substituído. "
            f"Detalhe: {type(exc).__name__}: {exc}"
        )

    # --------------------------------------------------------
    # PUBLICAR MODEL.ONNX
    # --------------------------------------------------------

    os.replace(ONNX_TEMP_PATH, ONNX_PATH)

    final_size_mb = ONNX_PATH.stat().st_size / 1024 / 1024

    print("[EcoScan] model.onnx publicado com sucesso.", flush=True)
    print(f"[EcoScan] Tamanho: {final_size_mb:.1f} MB", flush=True)

    print("=" * 60, flush=True)
    print("[EcoScan] CONVERSÃO CONCLUÍDA COM SUCESSO", flush=True)
    print(f"[EcoScan] Entrada: {IMG_SIZE}x{IMG_SIZE}", flush=True)
    print(f"[EcoScan] Saída esperada: [1, N, 9]", flush=True)
    print(f"[EcoScan] ONNX: {final_size_mb:.1f} MB", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
