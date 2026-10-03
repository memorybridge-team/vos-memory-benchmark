"""가짜 SAM2: 진짜 SAM2 세션과 같은 사용법 (start / add_prompt / load_memory / track / export_memory).

프롬프트 마스크 아래의 색을 기억해 두고, 다음 프레임마다 그 색인 픽셀을 칠한다.
기억 칸에 색이 obj_ptr 로 들어 있어서, 기억을 넘기면 Base+ 도 같은 색을 따라간다.
Small 은 다섯 줄마다 한 줄을 빼먹어서 Base+ 보다 점수가 조금 낮다.
기억 칸이 없으면(아무 프롬프트도 못 받으면) 빈 마스크.
"""

import numpy as np
import torch
from PIL import Image

import settings
from model.sam2_runner import FrameOut


def _entry(color, mask: np.ndarray) -> dict:
    c = color if color is not None else (-1, -1, -1)
    return {
        "maskmem_features": torch.full((1, 4, 2, 2), float(mask.mean())),
        "maskmem_pos_enc": [torch.zeros(1, 4, 2, 2)],
        "pred_masks": torch.from_numpy(mask.astype(np.float32))[None, None],
        "obj_ptr": torch.tensor([[*c, 1.0, 0.0]], dtype=torch.float32),
        "object_score_logits": torch.tensor([[10.0 if mask.any() else -10.0]]),
    }


def _color_of(entry):
    c = tuple(int(round(v)) for v in entry["obj_ptr"][0, :3].tolist())
    return None if min(c) < 0 else c


class FakeRunner:
    def __init__(self, name: str):
        self.name = name

    def start(self, video):
        return FakeSession(self, video)


class FakeSession:
    def __init__(self, runner, video):
        self.runner, self.video = runner, video
        self.cond, self.non_cond = {}, {}

    def _image(self, frame):
        return np.array(Image.open(self.video.frame_paths[frame]).convert("RGB"))

    def add_prompt(self, frame, mask):
        pixels = self._image(frame)[mask]
        color = None
        if len(pixels):
            values, counts = np.unique(pixels, axis=0, return_counts=True)
            color = tuple(int(v) for v in values[counts.argmax()])
        self.cond[frame] = _entry(color, mask)

    def load_memory(self, entries):
        for frame, e in entries.items():
            target = self.cond if e["is_cond"] else self.non_cond
            target[frame] = {k: v for k, v in e.items() if k != "is_cond"}

    def encode_prompts(self):
        pass

    def export_memory(self):
        out = {}
        for store, is_cond in ((self.cond, True), (self.non_cond, False)):
            for frame, e in store.items():
                out[frame] = {**{k: (v.clone() if torch.is_tensor(v) else [x.clone() for x in v])
                                 for k, v in e.items()}, "is_cond": is_cond}
        return out

    def memory_of(self, frame):
        return self.cond.get(frame) or self.non_cond.get(frame)

    def track(self, first, last):
        for f in range(first, last + 1):
            if f in self.cond:
                mask = self.cond[f]["pred_masks"][0, 0].numpy() > 0.5
            else:
                earlier = [t for t in list(self.cond) + list(self.non_cond) if t < f]
                color = _color_of(self.memory_of(max(earlier))) if earlier else None
                if color is None:
                    mask = np.zeros(self.video.size, dtype=bool)
                else:
                    mask = (self._image(f) == color).all(-1)
                    if self.runner.name == "small":
                        mask[::5] = False
                self.non_cond[f] = _entry(color, mask)
                for t in [t for t in self.non_cond if t < f - settings.MEMORY_WINDOW]:
                    del self.non_cond[t]
            yield FrameOut(f, mask, bool(mask.any()))

    def close(self):
        pass
