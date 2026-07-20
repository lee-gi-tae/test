# ONNX model compatibility check

## 가장 간단한 실행 방법

Windows에서 `run_gpu_check.bat`을 실행하면 가상환경 생성, 고정 GPU 패키지 설치,
전체 모델 검사와 CSV 생성까지 자동으로 진행됩니다.

```powershell
.\run_gpu_check.bat
```

검사 목록은 `models_gpu.json`에서 관리하며 `category` 값으로 구조를 구분합니다.

- `CNN`: ConvNeXt, RepViT, MobileNet, MobileOne, RDNet, FasterNet, EfficientNet
- `HYBRID`: FastViT, EfficientViT, EdgeNeXt
- `ATTENTION`: SwiftFormer, TinyViT

Python 3.12 프로젝트 전용 환경에서 timm 분류 모델을 ONNX로 변환하고 검증합니다.

검증 순서:

1. 권장 Dynamo exporter와 동적 batch로 변환
2. ONNX checker
3. ONNX Runtime CPU 로드 및 실행
4. batch 1, 2, 4에서 PyTorch 출력과 수치 비교
5. Dynamo 경로 실패 시 legacy exporter로 전체 절차 재시도

현재 고정 환경(PyTorch 2.13)의 Dynamo exporter 구현에 맞춰 기본 opset은 18입니다.

실행:

```powershell
.\.venv\Scripts\python.exe check_onnx.py --fresh
```

중단된 실행을 이어갈 때:

```powershell
.\.venv\Scripts\python.exe check_onnx.py --resume
```

일부 모델만 실행:

```powershell
.\.venv\Scripts\python.exe check_onnx.py --models SwiftFormer MaxViT
```

결과는 `results.csv`, 상세 오류는 `outputs/logs/`에 저장됩니다. 기본 모델과 정확한
timm 가중치 ID는 `models.json`에서 변경할 수 있습니다.

## GPU Dynamo-only 검증

RTX 5070에서 Legacy fallback 없이 pretrained 모델을 검증합니다.

```powershell
.\.venv-gpu\Scripts\python.exe check_onnx_gpu.py
```

GPU 환경은 `requirements-gpu.txt`에 고정되어 있으며, 결과는 `results_gpu.csv`,
ONNX와 상세 로그는 `outputs_gpu/`에 저장됩니다. CPU fallback을 금지한
`CUDAExecutionProvider` 세션에서 batch 1, 2, 4를 모두 통과해야 `가능`입니다.

CPU가 담당한 노드를 실제 profiling으로 확인하려면 다음을 실행합니다.

```powershell
.\.venv-gpu\Scripts\python.exe profile_onnx_gpu.py
```

모델별 요약은 `results_gpu_profile.csv`, CPU 노드 상세 목록은
`cpu_nodes_gpu_profile.csv`에 즉시 누적 저장됩니다.
