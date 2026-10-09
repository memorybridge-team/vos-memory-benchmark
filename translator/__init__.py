"""본 모델: Small 기억 칸을 팀 translator 로 바꿔 Base+ 에 넣는다 (다시 보기 없음). docs/PROTOCOL.md "본 모델".

팀 평가 코드 (vos_memory_inspector.lvos_evaluation.no_replay_case) 와 같은 길:
  칸마다 maskmem_features (1,64,64,64) bf16 · obj_ptr (1,256) fp32 를 translator 에 넣고,
  autocast 를 끈 채 (fp32 계산) 바꾼 뒤, 원래 dtype 그대로 Base+ 에 넣는다.
  나머지 칸은 Direct State Copy 처럼 Small 것 그대로 (SAM2 가 s+1 부터 읽지 않거나, 두 모델이 같은 값).

    load()                  전달받은 가중치인지 확인하고 GPU 에 올린다 (2_evaluate.py 가 시작할 때 한 번 → 시간 열에 안 들어감)
    prepare(session, pkg)   비교군과 같은 규칙: Base+ 세션 준비 → 추적 시작 프레임
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

import torch

import settings
from baseline import Baseline

_translator = None


def load() -> None:
    global _translator
    _translator = None
    root = Path(settings.TRANSLATOR_DIR)
    weights = root / settings.TRANSLATOR_WEIGHTS
    digest = hashlib.sha256()
    with weights.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != settings.TRANSLATOR_SHA256:
        raise ValueError(f"전달받은 translator 가중치가 아닙니다 (SHA256 다름): {weights}")
    source = (root / settings.TRANSLATOR_SOURCE).resolve()
    # 프로젝트 모듈의 우선순위를 유지하며 전달본 패키지 경로만 import 때 사용한다.
    original_path = sys.path[:]
    try:
        sys.path.append(str(source))
        module = importlib.import_module("vos_memory_inspector.transformer_translator")
    finally:
        sys.path[:] = original_path
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise ValueError(f"다른 translator 코드가 이미 import되어 있습니다: {module.__file__}")
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    _translator = module.TransformerStateTranslator.from_payload(payload).eval().to(settings.DEVICE)


def translate(entries: dict) -> dict:
    """{프레임: 기억 칸} → 같은 모양, 칸마다 maskmem_features · obj_ptr 만 바뀐 것."""
    frames = sorted(entries)
    if _translator is None or not frames:
        raise ValueError("translator를 먼저 load하고 비어 있지 않은 기억을 전달하세요.")
    spatial = torch.stack([entries[f]["maskmem_features"][0] for f in frames])[None, None]   # [1, 1, 칸, 64, 64, 64]
    pointer = torch.stack([entries[f]["obj_ptr"][0] for f in frames])[None, None]            # [1, 1, 칸, 256]
    spatial_dtype, pointer_dtype = spatial.dtype, pointer.dtype
    spatial_shape, pointer_shape = spatial.shape, pointer.shape
    with torch.inference_mode(), torch.autocast(torch.device(settings.DEVICE).type, enabled=False):
        # translator.translate(CanonicalState) 안에서 하는 계산 그대로 (칸이 모두 유효할 때)
        spatial, pointer = _translator.translate_handoff_tensors(spatial, pointer)
    if spatial.shape != spatial_shape or pointer.shape != pointer_shape:
        raise ValueError("translator 출력의 기억 모양이 입력과 다릅니다.")
    spatial = spatial.to(dtype=spatial_dtype)
    pointer = pointer.to(dtype=pointer_dtype)
    return {f: {**entries[f], "maskmem_features": spatial[0, 0, k:k + 1], "obj_ptr": pointer[0, 0, k:k + 1]}
            for k, f in enumerate(frames)}


def prepare(session, pkg):
    session.load_memory(translate(pkg.small_memory))
    return pkg.switch_frame + 1


MODEL = Baseline("translator", "본 모델 (translator)", "model", prepare, revision=2)
