"""MOSEv2 valid: 여기서는 채점을 못 한다 (정답이 첫 프레임뿐). 제출용 PNG를 만들고 서버 점수를 받는다.

1. 평가 중: 객체마다 마스크를 따로 저장 (MaskSaver)
     masks/small/<영상>/<객체>/<프레임>.png                Small 이 처음~끝 (전환 전 구간 + Source-only)
     masks/full_replay/<영상>/<객체>/<프레임>.png          Full Replay 전체
     masks/<비교군>__<전환>/<영상>/<객체>/<프레임>.png     그 비교군의 전환 뒤 구간
   빈 마스크는 저장하지 않는다 (파일 없음 = 빈 마스크).
2. 평가 끝: 제출 하나 = 비교군 × 전환 시점. 객체들을 한 장으로 합쳐 zip (build_submissions)
     Source-only 는 전환 시점과 무관 → 1번. 나머지 9개 × 전환 3개 = 27번. 합 28번.
3. 서버 점수 CSV → 결과 줄 (scripts/4_mosev2_scores.py)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import settings
from benchmark.baselines import MAIN, EXTRA


def mosev2_root() -> Path:
    return Path(settings.OUTPUT_ROOT) / "mosev2"


class MaskSaver:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or mosev2_root()) / "masks"

    def _path(self, folder, video, obj_id, frame_name) -> Path:
        return self.root / folder / video / str(obj_id) / f"{frame_name}.png"

    def save(self, folder, video, obj_id, frame_name, mask) -> None:
        if not mask.any():
            return
        path = self._path(folder, video, obj_id, frame_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(bool)).save(path)

    def load(self, folder, video, obj_id, frame_name):
        path = self._path(folder, video, obj_id, frame_name)
        if not path.exists():
            return None
        with Image.open(path) as im:
            return np.array(im).astype(bool)


def switch_names() -> list[str]:
    return [f"A{round(f * 100)}" for f in settings.SWITCH_A_FRACTIONS]


def submission_list() -> list[tuple[str, str, str | None]]:
    """[(제출 이름, 비교군, 전환 이름)] — Source-only 는 전환 이름 None."""
    baselines = MAIN + (EXTRA if settings.MOSEV2_SUBMIT_EXTRAS else [])
    subs = []
    for m in baselines:
        if m.name == "source_only":
            subs.append(("source_only", m.name, None))
        else:
            subs += [(f"{m.name}__{sw}", m.name, sw) for sw in switch_names()]
    return subs


def _folder_for(baseline: str, switch_name: str | None, obj: dict, frame: int) -> str:
    """이 객체의 이 프레임 마스크를 어느 폴더에서 가져올지."""
    if baseline == "source_only":
        return "small"
    s = next(sw["frame"] for sw in obj["switches"] if sw["name"] == switch_name)
    if frame <= s:
        return "small"
    if baseline == "full_replay":
        return "full_replay"
    return f"{baseline}__{switch_name}"


def _default_palette() -> list[int]:
    rng = np.random.default_rng(0)
    colors = [[0, 0, 0]] + rng.integers(0, 256, size=(255, 3)).tolist()
    return [c for rgb in colors for c in rgb]


def build_submission(name, baseline, switch_name, video_list, videos, saver, out_root) -> Path:
    """객체들을 한 장으로 합친다. 겹치면 번호 작은 객체가 이긴다 (SAM2 공식 평가 코드와 같은 규칙)."""
    folder = Path(out_root) / name
    inner = folder / settings.MOSEV2_ZIP_INNER_FOLDER if settings.MOSEV2_ZIP_INNER_FOLDER else folder
    for entry in video_list["videos"]:
        video = videos[entry["video"]]
        palette = video.palette() or _default_palette()
        objects = sorted(entry["objects"], key=lambda o: o["object"], reverse=True)
        height, width = video.size
        (inner / video.name).mkdir(parents=True, exist_ok=True)
        for frame in range(video.num_frames):
            canvas = np.zeros((height, width), dtype=np.uint8)
            for obj in objects:
                if frame < obj["start"]:
                    continue
                src = _folder_for(baseline, switch_name, obj, frame)
                mask = saver.load(src, video.name, obj["object"], video.frame_name(frame))
                if mask is not None:
                    canvas[mask] = obj["object"]
            png = Image.fromarray(canvas, mode="P")
            png.putpalette(palette)
            png.save(inner / video.name / f"{video.frame_name(frame)}.png")
    zip_path = shutil.make_archive(str(folder), "zip", root_dir=folder)
    return Path(zip_path)


def build_submissions(video_list, videos, saver) -> list[Path]:
    out_root = mosev2_root() / "submissions"
    zips = []
    for name, baseline, switch_name in submission_list():
        zips.append(build_submission(name, baseline, switch_name, video_list, videos, saver, out_root))
        print(f"  제출 파일: {zips[-1]}")
    return zips


def parse_submission(name: str) -> list[tuple[str, str]]:
    """제출 이름 → [(비교군, 전환 이름)]. Source-only 는 모든 전환 시점에 같은 점수."""
    if "__" in name:
        baseline, sw = name.split("__", 1)
        return [(baseline, sw)]
    return [(name, sw) for sw in switch_names()]
