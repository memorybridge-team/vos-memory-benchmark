"""설치된 SAM2 구조 확인.

본 실험 전에 확인할 세 가지:
  1. Small / Base+ 기억 칸의 모양이 같은가 → 같아야 Direct State Copy 가 된다.
  2. 기억 위치 정보(maskmem_pos_enc)가 두 모델에서 같은 값인가 → 같으면 복사해도 문제 없음.
  3. 같은 모델에서 기억을 꺼냈다 새 세션에 넣고 이어가도, 끊지 않은 결과와 프레임마다 같은가
     → 같아야 "꺼내기/넣기" 코드가 기억을 빠짐없이 옮긴다는 뜻.
"""

from __future__ import annotations

import numpy as np
import torch

import settings

REQUIRED_STATE_KEYS = ("output_dict_per_obj", "temp_output_dict_per_obj",
                       "frames_tracked_per_obj", "obj_id_to_idx", "device", "storage_device")


def _shape(x):
    if isinstance(x, (list, tuple)):
        return [_shape(v) for v in x]
    return f"{tuple(x.shape)} {str(x.dtype).replace('torch.', '')}"


def describe(runner, video, obj_id: int, start: int, n_frames: int = 8) -> dict:
    """모델 설정값 + 기억 칸 모양. 몇 프레임만 추적해 본다."""
    p = runner.predictor
    session = runner.start(video)
    missing = [k for k in REQUIRED_STATE_KEYS if k not in session.state]
    session.add_prompt(start, video.object_mask(start, obj_id))
    last = min(start + n_frames, video.num_frames - 1)
    for _ in session.track(start, last):
        pass
    entry = session.memory_of(last)
    info = {
        "model": runner.name,
        "image_size": p.image_size,
        "num_maskmem": p.num_maskmem,
        "max_obj_ptrs_in_encoder": p.max_obj_ptrs_in_encoder,
        "hidden_dim": p.hidden_dim,
        "mem_dim": p.mem_dim,
        "missing_state_keys": missing,
        "fields": {name: _shape(entry[name]) for name in entry},
        "entry": entry,
    }
    session.close()
    return info


def shape_mismatches(a: dict, b: dict) -> list[str]:
    """두 모델의 기억 칸 모양이 다른 곳."""
    return [f"{name}: {a['fields'][name]} vs {b['fields'].get(name)}"
            for name in a["fields"] if a["fields"][name] != b["fields"].get(name)]


def same_pos_enc(a: dict, b: dict) -> bool:
    pa = [t.float().cpu() for t in a["entry"]["maskmem_pos_enc"]]
    pb = [t.float().cpu() for t in b["entry"]["maskmem_pos_enc"]]
    return len(pa) == len(pb) and all(torch.equal(x, y) for x, y in zip(pa, pb))


def window_is_enough(info: dict) -> bool:
    """MEMORY_WINDOW 가 SAM2가 실제로 읽는 범위를 덮는가."""
    need = max(info["num_maskmem"] - 1, info["max_obj_ptrs_in_encoder"] - 1)
    return settings.MEMORY_WINDOW >= need


def roundtrip(runner, video, obj_id: int, start: int, cut: int, last: int) -> dict:
    """끊지 않고 쭉 간 결과 vs cut 에서 기억을 꺼내 새 세션에 넣고 이어간 결과."""
    prompt = video.object_mask(start, obj_id)

    whole = runner.start(video)
    whole.add_prompt(start, prompt)
    reference = {out.frame: out.mask for out in whole.track(start, last)}
    whole.close()

    first = runner.start(video)
    first.add_prompt(start, prompt)
    for _ in first.track(start, cut):
        pass
    memory = first.export_memory()
    first.close()

    second = runner.start(video)
    second.load_memory(memory)
    resumed = {out.frame: out.mask for out in second.track(cut + 1, last)}
    second.close()

    frames = sorted(resumed)
    same = [f for f in frames if np.array_equal(resumed[f], reference[f])]
    diff_pixels = max((int((resumed[f] != reference[f]).sum()) for f in frames), default=0)
    return {"frames": len(frames), "identical_frames": len(same), "max_diff_pixels": diff_pixels}
