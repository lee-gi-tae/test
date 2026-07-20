from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs_gpu"
LOG_DIR = OUTPUT_DIR / "logs"
RESULTS_PATH = ROOT / "results_gpu.csv"
BATCHES = (1, 2, 4)
OPSET = 18
RTOL = 1e-3
ATOL = 1e-4

# 서로 다른 CUDA 커널의 TF32 선택 차이가 ONNX 변환 오차로 오인되지 않도록
# 양쪽 모두 IEEE FP32 경로로 고정합니다.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class EnvironmentErrorForValidation(RuntimeError):
    pass


class OutputMismatch(AssertionError):
    def __init__(self, message: str, metrics: dict[str, str | float]):
        super().__init__(message)
        self.metrics = metrics


def driver_version() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def environment() -> dict[str, str]:
    return {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": str(torch.backends.cudnn.version()),
        "onnx_version": onnx.__version__,
        "onnxruntime_gpu_version": ort.__version__,
        "timm_version": timm.__version__,
        "driver_version": driver_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }


def strict_cuda_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return options


def strict_cuda_providers() -> list[tuple[str, dict[str, str]]]:
    return [("CUDAExecutionProvider", {"use_tf32": "0"})]


def preflight() -> None:
    if not torch.cuda.is_available():
        raise EnvironmentErrorForValidation("PyTorch CUDA를 사용할 수 없습니다")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise EnvironmentErrorForValidation("CUDAExecutionProvider가 없습니다")

    value = (torch.ones(4, device="cuda") * 2).sum().item()
    if value != 8:
        raise EnvironmentErrorForValidation("PyTorch CUDA 기준 연산이 잘못됐습니다")

    # 모델 호환성과 환경 문제를 분리하기 위한 최소 CUDA Add 그래프입니다.
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    one = helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "one"], ["y"])],
        "cuda_preflight",
        [x_info],
        [y_info],
        [one],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        ir_version=10,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cuda_preflight.onnx"
        onnx.save(model, path)
        try:
            session = ort.InferenceSession(
                str(path),
                sess_options=strict_cuda_options(),
                providers=strict_cuda_providers(),
            )
            result = session.run(None, {"x": np.zeros((1, 4), np.float32)})[0]
        except Exception as exc:
            raise EnvironmentErrorForValidation(
                f"ORT CUDA 기준 모델 실행 실패: {type(exc).__name__}: {exc}"
            ) from exc
        if not np.array_equal(result, np.ones((1, 4), np.float32)):
            raise EnvironmentErrorForValidation("ORT CUDA 기준 모델 출력이 잘못됐습니다")


def export_dynamo(model: torch.nn.Module, sample: torch.Tensor, path: Path) -> None:
    batch = torch.export.Dim("batch", min=1, max=max(BATCHES))
    torch.onnx.export(
        model,
        (sample,),
        path,
        dynamo=True,
        dynamic_shapes=({0: batch},),
        input_names=["input"],
        output_names=["output"],
        opset_version=OPSET,
        external_data=False,
    )


def validate_outputs(
    model: torch.nn.Module,
    session: ort.InferenceSession,
    input_shape: tuple[int, int, int],
) -> dict[str, str | float]:
    max_abs = 0.0
    mean_abs: list[float] = []
    top1_matches: list[float] = []
    passed: list[int] = []

    for batch in BATCHES:
        torch.manual_seed(batch)
        cpu_input = torch.randn(batch, *input_shape, dtype=torch.float32)
        gpu_input = cpu_input.cuda()
        with torch.inference_mode():
            torch_output = model(gpu_input)
        if not isinstance(torch_output, torch.Tensor):
            raise TypeError(f"지원하지 않는 출력 형식: {type(torch_output).__name__}")
        expected = torch_output.detach().cpu().numpy()
        actual = session.run(
            None,
            {session.get_inputs()[0].name: cpu_input.numpy()},
        )[0]

        if expected.shape != actual.shape:
            raise AssertionError(f"출력 shape 불일치: PyTorch={expected.shape}, ONNX={actual.shape}")
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise AssertionError("출력에 NaN 또는 Inf가 있습니다")

        difference = np.abs(expected - actual)
        max_abs = max(max_abs, float(difference.max(initial=0)))
        mean_abs.append(float(difference.mean()))
        close = np.allclose(expected, actual, rtol=RTOL, atol=ATOL)
        match = 1.0
        if expected.ndim == 2:
            match = np.mean(expected.argmax(axis=1) == actual.argmax(axis=1))
            top1_matches.append(float(match))
        passed.append(batch)
        current_metrics: dict[str, str | float] = {
            "batches": ";".join(map(str, passed)),
            "max_abs_error": max_abs,
            "mean_abs_error": float(np.mean(mean_abs)),
            "top1_match": float(np.mean(top1_matches)) if top1_matches else "",
        }
        if not close:
            raise OutputMismatch(
                f"수치 허용오차 초과: batch={batch}, rtol={RTOL}, atol={ATOL}",
                current_metrics,
            )
        if match != 1.0:
            raise OutputMismatch(
                f"Top-1 불일치: batch={batch}, match={match}",
                current_metrics,
            )
        del cpu_input, gpu_input, torch_output

    return {
        "batches": ";".join(map(str, passed)),
        "max_abs_error": max_abs,
        "mean_abs_error": float(np.mean(mean_abs)),
        "top1_match": float(np.mean(top1_matches)) if top1_matches else "",
    }


def validate_model(spec: dict[str, str]) -> dict[str, object]:
    name = spec["model"]
    checkpoint = spec["checkpoint"]
    safe_name = checkpoint.replace("/", "_").replace("\\", "_")
    onnx_path = OUTPUT_DIR / f"{safe_name}.dynamo.onnx"
    log_path = LOG_DIR / f"{safe_name}.dynamo.log"
    base: dict[str, object] = {
        "model": name,
        "checkpoint": checkpoint,
        "verdict": "불가능",
        "exporter": "Dynamo",
        "opset": OPSET,
        "cpu_fallback_disabled": True,
    }
    model = None
    session = None
    started = time.perf_counter()
    stage = "MODEL_LOAD"
    try:
        torch.manual_seed(0)
        model = timm.create_model(checkpoint, pretrained=True).eval().cpu()
        config = timm.data.resolve_model_data_config(model)
        shape = tuple(config["input_size"])
        base["input_shape"] = "x".join(map(str, shape))
        # torch.export는 크기 0/1을 특수화하므로 동적 batch 예제는 2를 사용합니다.
        # 생성된 ONNX는 아래 비교 단계에서 batch 1, 2, 4를 모두 검증합니다.
        sample = torch.randn(2, *shape, dtype=torch.float32)

        stage = "EXPORT"
        onnx_path.unlink(missing_ok=True)
        export_dynamo(model, sample, onnx_path)

        stage = "CHECKER"
        checked = onnx.load(str(onnx_path), load_external_data=False)
        onnx.checker.check_model(str(onnx_path))
        base["actual_opset"] = ";".join(
            f"{item.domain or 'ai.onnx'}:{item.version}" for item in checked.opset_import
        )

        stage = "CUDA_SESSION"
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=strict_cuda_options(),
            providers=strict_cuda_providers(),
        )

        stage = "CUDA_COMPARE"
        model.cuda()
        metrics = validate_outputs(model, session, shape)
        base.update(metrics)
        base.update({"verdict": "가능", "failure_stage": "", "error": ""})
        log_path.write_text("PASS\n", encoding="utf-8")
    except Exception as exc:
        if isinstance(exc, OutputMismatch):
            base.update(exc.metrics)
        base.update(
            {
                "failure_stage": stage,
                "error": f"{type(exc).__name__}: {exc}"[:2000],
            }
        )
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        base["seconds"] = round(time.perf_counter() - started, 3)
        session = None
        model = None
        gc.collect()
        torch.cuda.empty_cache()
    return base


def save_results(rows: list[dict[str, object]]) -> None:
    env = environment()
    merged = [{**env, **row} for row in rows]
    fields = sorted({key for row in merged for key in row})
    with RESULTS_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dynamo-only ONNX CUDA validation")
    parser.add_argument("--models", nargs="*", help="모델명 또는 checkpoint 일부")
    args = parser.parse_args()

    specs = json.loads((ROOT / "models_gpu.json").read_text(encoding="utf-8"))
    if args.models:
        terms = [term.lower() for term in args.models]
        specs = [
            spec for spec in specs
            if any(term in spec["model"].lower() or term in spec["checkpoint"].lower() for term in terms)
        ]
    if not specs:
        parser.error("선택된 모델이 없습니다")

    OUTPUT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    try:
        preflight()
    except Exception as exc:
        rows = [
            {
                "model": spec["model"],
                "checkpoint": spec["checkpoint"],
                "verdict": "미판정",
                "failure_stage": "ENVIRONMENT_ERROR",
                "error": f"{type(exc).__name__}: {exc}"[:2000],
            }
            for spec in specs
        ]
        save_results(rows)
        print(f"GPU 환경 검사 실패: {exc}", file=sys.stderr)
        return 2

    print("GPU 환경 검사 통과", flush=True)
    rows: list[dict[str, object]] = []
    for index, spec in enumerate(specs, 1):
        print(f"[{index}/{len(specs)}] {spec['model']} ({spec['checkpoint']})", flush=True)
        row = validate_model(spec)
        rows.append(row)
        save_results(rows)
        print(f"  => {row['verdict']} ({row.get('failure_stage', 'PASS')})", flush=True)
    return 0 if all(row["verdict"] == "가능" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
