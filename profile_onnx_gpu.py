from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch

import check_onnx_gpu as base


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs_gpu"
PROFILE_DIR = OUTPUT_DIR / "profiles"
LOG_DIR = OUTPUT_DIR / "profile_logs"
SUMMARY_CSV = ROOT / "results_gpu_profile.csv"
NODES_CSV = ROOT / "cpu_nodes_gpu_profile.csv"

# Shape 결과나 상수만 입력으로 받는 아래 연산은 shape/인덱스 계산으로 분류합니다.
SHAPE_FLOW_OPS = {
    "Add", "Cast", "Ceil", "Concat", "ConstantOfShape", "Div", "Floor",
    "Gather", "GatherElements", "Max", "Min", "Mod", "Mul", "Range",
    "Reshape", "Shape", "Size", "Slice", "Squeeze", "Sub", "Unsqueeze",
}
HEAVY_COMPUTE_OPS = {
    "Attention", "AveragePool", "BatchNormalization", "Conv", "ConvTranspose",
    "Gemm", "GlobalAveragePool", "LayerNormalization", "MatMul", "MaxPool",
    "MultiHeadAttention", "NonMaxSuppression", "ReduceMean", "Resize", "Softmax",
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["model"]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metadata_nodes(model_path: Path) -> dict[str, bool]:
    model = onnx.load(str(model_path), load_external_data=False)
    constant_values = {item.name for item in model.graph.initializer}
    metadata_values: set[str] = set()
    classification: dict[str, bool] = {}

    for node in model.graph.node:
        inputs_are_metadata = all(
            not name or name in constant_values or name in metadata_values
            for name in node.input
        )
        is_metadata = node.op_type in {"Shape", "Size"} or (
            node.op_type in SHAPE_FLOW_OPS and inputs_are_metadata
        )
        classification[node.name] = is_metadata
        if node.op_type == "Constant":
            constant_values.update(node.output)
        elif is_metadata:
            metadata_values.update(node.output)
    return classification


def profile_events(profile_path: Path, model_path: Path, model_name: str) -> list[dict[str, object]]:
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    metadata = metadata_nodes(model_path)
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}

    for event in events:
        if event.get("cat") != "Node":
            continue
        args = event.get("args", {})
        provider = str(args.get("provider", ""))
        if provider != "CPUExecutionProvider":
            continue
        raw_name = str(event.get("name", ""))
        node_name = raw_name.removesuffix("_kernel_time")
        op_type = str(args.get("op_name", "UNKNOWN"))
        if op_type in {"MemcpyFromHost", "MemcpyToHost"}:
            category = "DATA_TRANSFER"
        elif metadata.get(node_name, False) or op_type in {"Shape", "Size"}:
            category = "SHAPE_METADATA"
        elif op_type in HEAVY_COMPUTE_OPS:
            category = "HEAVY_COMPUTE"
        else:
            category = "REVIEW_NEEDED"
        key = (node_name, op_type, category)
        if key not in grouped:
            grouped[key] = {
                "model": model_name,
                "node_name": node_name,
                "op_type": op_type,
                "provider": provider,
                "category": category,
                "invocations": 0,
                "total_duration_us": 0,
            }
        grouped[key]["invocations"] = int(grouped[key]["invocations"]) + 1
        grouped[key]["total_duration_us"] = int(grouped[key]["total_duration_us"]) + int(event.get("dur", 0))
    return list(grouped.values())


def compare_batches(model, session, shape: tuple[int, int, int]) -> dict[str, object]:
    max_abs = 0.0
    mean_abs: list[float] = []
    all_close = True
    top1_match = True
    batches: list[int] = []
    for batch in base.BATCHES:
        torch.manual_seed(batch)
        cpu_input = torch.randn(batch, *shape, dtype=torch.float32)
        with torch.inference_mode():
            expected = model(cpu_input.cuda()).detach().cpu().numpy()
        actual = session.run(None, {session.get_inputs()[0].name: cpu_input.numpy()})[0]
        difference = np.abs(expected - actual)
        max_abs = max(max_abs, float(difference.max(initial=0)))
        mean_abs.append(float(difference.mean()))
        all_close = all_close and bool(np.allclose(expected, actual, rtol=base.RTOL, atol=base.ATOL))
        if expected.ndim == 2:
            top1_match = top1_match and bool(np.array_equal(expected.argmax(1), actual.argmax(1)))
        batches.append(batch)
    return {
        "batches": ";".join(map(str, batches)),
        "max_abs_error": max_abs,
        "mean_abs_error": float(np.mean(mean_abs)),
        "allclose": all_close,
        "top1_match": top1_match,
    }


def ensure_onnx(spec: dict[str, str]) -> tuple[Path, tuple[int, int, int]]:
    checkpoint = spec["checkpoint"]
    safe_name = checkpoint.replace("/", "_").replace("\\", "_")
    path = OUTPUT_DIR / f"{safe_name}.dynamo.onnx"
    model = timm.create_model(checkpoint, pretrained=True).eval().cpu()
    shape = tuple(timm.data.resolve_model_data_config(model)["input_size"])
    if not path.exists():
        torch.manual_seed(0)
        base.export_dynamo(model, torch.randn(2, *shape), path)
        onnx.checker.check_model(str(path))
    del model
    return path, shape


def run_one(spec: dict[str, str]) -> tuple[dict[str, object], list[dict[str, object]]]:
    name, checkpoint = spec["model"], spec["checkpoint"]
    result: dict[str, object] = {"model": name, "checkpoint": checkpoint}
    model = None
    session = None
    profile_path = None
    started = time.perf_counter()
    try:
        model_path, shape = ensure_onnx(spec)
        result["input_shape"] = "x".join(map(str, shape))
        model = timm.create_model(checkpoint, pretrained=True).eval().cuda()

        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(PROFILE_DIR / checkpoint.replace("/", "_"))
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=[
                ("CUDAExecutionProvider", {"use_tf32": "0"}),
                "CPUExecutionProvider",
            ],
        )
        result.update(compare_batches(model, session, shape))
        profile_path = Path(session.end_profiling())
        session = None
        node_rows = profile_events(profile_path, model_path, name)
        counts = Counter(str(row["category"]) for row in node_rows)
        ops = sorted({str(row["op_type"]) for row in node_rows})
        heavy_ops = sorted({str(row["op_type"]) for row in node_rows if row["category"] == "HEAVY_COMPUTE"})
        review_ops = sorted({str(row["op_type"]) for row in node_rows if row["category"] == "REVIEW_NEEDED"})
        result.update(
            {
                "cpu_unique_nodes": len(node_rows),
                "cpu_shape_metadata_nodes": counts["SHAPE_METADATA"],
                "cpu_data_transfer_nodes": counts["DATA_TRANSFER"],
                "cpu_heavy_compute_nodes": counts["HEAVY_COMPUTE"],
                "cpu_review_needed_nodes": counts["REVIEW_NEEDED"],
                "cpu_op_types": ";".join(ops),
                "cpu_heavy_ops": ";".join(heavy_ops),
                "cpu_review_ops": ";".join(review_ops),
            }
        )
        if counts["HEAVY_COMPUTE"]:
            result["cpu_work_verdict"] = "주요연산_CPU"
        elif counts["REVIEW_NEEDED"]:
            result["cpu_work_verdict"] = "검토필요"
        else:
            result["cpu_work_verdict"] = "보조연산만_CPU"
        result["error"] = ""
    except Exception as exc:
        result.update(
            {
                "cpu_work_verdict": "실행실패",
                "error": f"{type(exc).__name__}: {exc}"[:2000],
            }
        )
        (LOG_DIR / f"{checkpoint.replace('/', '_')}.log").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        node_rows = []
    finally:
        if session is not None:
            try:
                profile_path = Path(session.end_profiling())
            except Exception:
                pass
        result["seconds"] = round(time.perf_counter() - started, 3)
        session = None
        model = None
        gc.collect()
        torch.cuda.empty_cache()
    return result, node_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="ORT CPU node profiling on CUDA")
    parser.add_argument("--models", nargs="*", help="모델명 또는 checkpoint 일부")
    args = parser.parse_args()
    specs = json.loads((ROOT / "models_gpu.json").read_text(encoding="utf-8"))
    if args.models:
        terms = [item.lower() for item in args.models]
        specs = [
            spec for spec in specs
            if any(term in spec["model"].lower() or term in spec["checkpoint"].lower() for term in terms)
        ]
    if not specs:
        parser.error("선택된 모델이 없습니다")

    OUTPUT_DIR.mkdir(exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        base.preflight()
    except Exception as exc:
        print(f"GPU 환경 검사 실패: {exc}", file=sys.stderr)
        return 2

    summaries: list[dict[str, object]] = []
    nodes: list[dict[str, object]] = []
    for index, spec in enumerate(specs, 1):
        print(f"[{index}/{len(specs)}] {spec['model']}", flush=True)
        summary, node_rows = run_one(spec)
        summaries.append({**base.environment(), **summary})
        nodes.extend(node_rows)
        write_csv(SUMMARY_CSV, summaries)
        write_csv(NODES_CSV, nodes)
        print(f"  => {summary['cpu_work_verdict']}", flush=True)
    return 0 if all(not row.get("error") for row in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
