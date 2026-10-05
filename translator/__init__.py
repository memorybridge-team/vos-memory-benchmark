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
import sys
from pathlib import Path

import torch

import settings
from baseline import Baseline

_translator = None


def load() -> None:
    global _translator
    root = Path(settings.TRANSLATOR_DIR)
    weights = root / settings.TRANSLATOR_WEIGHTS
    if hashlib.sha256(weights.read_bytes()).hexdigest() != settings.TRANSLATOR_SHA256:
        raise ValueError(f"전달받은 translator 가중치가 아닙니다 (SHA256 다름): {weights}")
    sys.path.insert(0, str(root / settings.TRANSLATOR_SOURCE))
    from vos_memory_inspector.transformer_translator import TransformerStateTranslator
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    _translator = TransformerStateTranslator.from_payload(payload).eval().to(settings.DEVICE)


def translate(entries: dict) -> dict:
    """{프레임: 기억 칸} → 같은 모양, 칸마다 maskmem_features · obj_ptr 만 바뀐 것."""
    frames = sorted(entries)
    spatial = torch.stack([entries[f]["maskmem_features"][0] for f in frames])[None, None]   # [1, 1, 칸, 64, 64, 64]
    pointer = torch.stack([entries[f]["obj_ptr"][0] for f in frames])[None, None]            # [1, 1, 칸, 256]
    with torch.inference_mode(), torch.autocast(torch.device(settings.DEVICE).type, enabled=False):
        # translator.translate(CanonicalState) 안에서 하는 계산 그대로 (칸이 모두 유효할 때)
        spatial, pointer = _translator.translate_handoff_tensors(spatial, pointer)
    return {f: {**entries[f], "maskmem_features": spatial[0, 0, k:k + 1], "obj_ptr": pointer[0, 0, k:k + 1]}
            for k, f in enumerate(frames)}


def prepare(session, pkg):
    session.load_memory(translate(pkg.small_memory))
    return pkg.switch_frame + 1


MODEL = Baseline("translator", "본 모델 (translator)", "model", prepare)
