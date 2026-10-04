"""영상 하나의 객체 하나 × 전환 시점 전부 × 비교군 전부 → 결과 줄 목록.

  ① Full Replay: Base+ 가 처음 ~ 끝을 한 번 추적한다 (전환 시점과 무관).
       마스크를 압축해 들고 있는다 → 다른 비교군의 [추가] 출력 일치도 기준.
  ② Small 이 처음 ~ 끝을 한 번 추적한다.
       = Source-only 결과이자, 모든 비교군이 같이 쓰는 전환 전 구간.
       전환 프레임마다 기억 상자(HandoffPackage)를 챙겨 둔다.
  ③ 전환 시점 × 나머지 비교군: 새 Base+ 세션에 비교군마다 다른 것을 넘기고 전환 뒤 ~ 끝 추적.
  ④ Base+ 실행마다 시간·GPU 메모리 기록 (cost.py). Source-only 는 전환이 없어 재지 않는다.
  ⑤ 채점: 전환 뒤 J·J&F (주) + [추가] 실패 비율·출력 일치도.
  ⑥ 줄 목록을 돌려준다 (저장·이어하기는 records.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import settings
from evaluation import cost
from baseline import BASELINES, no_handoff
from baseline.handoff import HandoffPackage
from evaluation.scoring import extra_metrics, jf, main_metrics


@dataclass
class Run:
    """추적 한 번의 결과: 프레임별 점수·출력 일치도·시간, GPU 메모리."""
    scores: dict = field(default_factory=dict)     # 프레임 → FrameScore (정답 있는 프레임만)
    agreement: dict = field(default_factory=dict)  # 프레임 → Full Replay 마스크와의 IoU
    times: dict = field(default_factory=dict)      # 프레임 → 초 (Base+ 만)
    setup_seconds: float = 0.0
    gpu_before_mb: float | None = None             # Base+ 세션을 연 직후 GPU 사용량
    gpu_peaks: dict = field(default_factory=dict)  # 프레임 → 그 프레임까지(준비 포함) GPU 최고치


class PackedMasks:
    """마스크를 프레임마다 압축해 둔다 (참/거짓 한 칸을 1비트로 → 8분의 1 크기)."""

    def __init__(self, size):
        self.size = size
        self.packed = {}

    def add(self, frame: int, mask: np.ndarray) -> None:
        self.packed[frame] = np.packbits(mask)

    def get(self, frame: int) -> np.ndarray:
        height, width = self.size
        return np.unpackbits(self.packed[frame], count=height * width).reshape(self.size).astype(bool)


class FrameScorer:
    """정답이 있는 프레임에서 예측 마스크 하나를 채점한다."""

    def __init__(self, video, obj_id: int):
        self.video, self.obj_id = video, obj_id

    def score(self, frame: int, pred: np.ndarray):
        gt = self.video.read_labels(frame)
        if gt is None:
            return None
        labels, ignore = gt
        own = labels == self.obj_id
        return jf.FrameScore(
            j=jf.j_score(pred, own, ignore),
            f=jf.f_score(pred, own, ignore),
            gt_visible=bool(own.any()),
        )


@dataclass
class Keeper:
    """한 프레임 결과를 채점하고 Full Replay 와 비교한다."""
    scorer: FrameScorer
    replay: PackedMasks | None = None       # ① Full Replay 가 끝난 뒤 채워짐

    def keep(self, run: Run, frame: int, mask: np.ndarray) -> None:
        score = self.scorer.score(frame, mask)
        if score is not None:
            run.scores[frame] = score
        if self.replay is not None and frame in self.replay.packed:
            run.agreement[frame] = extra_metrics.mask_iou(mask, self.replay.get(frame))


def _run_small(small, video, obj, prompt, keeper):
    """② Small 이 처음 ~ 끝. 전환 프레임마다 기억 상자를 챙긴다."""
    start, end = obj["start"], obj["end"]
    switch_frames = {sw["frame"] for sw in obj["switches"]}
    keep_recent = settings.EXTRA_RECENT_K + 1
    run, packages, recent, last_visible = Run(), {}, {}, None

    session = small.start(video)
    session.add_prompt(start, prompt)
    for out in session.track(start, end):
        f = out.frame
        keeper.keep(run, f, out.mask)
        recent[f] = out.mask
        recent.pop(f - keep_recent, None)
        if out.visible and out.mask.any():
            last_visible = (f, out.mask)
        if f in switch_frames:
            packages[f] = HandoffPackage(
                switch_frame=f, prompt_frame=start, prompt_mask=prompt,
                small_memory=session.export_memory(),
                last_visible=last_visible, recent_masks=dict(recent))
    session.close()
    return run, packages


def _run_base(base, video, obj, prepare, keep_after, keeper, collect=None):
    """새 Base+ 세션을 prepare 로 준비하고 끝까지 추적. keep_after 보다 뒤 프레임만 결과로 남긴다.

    GPU: 세션을 연 직후 사용량을 적고, 준비가 끝난 때(track_from − 1 칸)와 프레임마다
    그때까지의 최고치를 적는다 → gpu_peaks[s] = Base+ 가 s+1 처리 준비를 마칠 때까지의 최고치.
    collect 가 있으면 남긴 마스크를 거기에 압축해 모은다 (① Full Replay 만).
    """
    end = obj["end"]
    run = Run()
    session = base.start(video)
    run.gpu_before_mb = cost.gpu_mb()
    cost.reset_gpu_peak()
    t0 = cost.now()
    track_from = prepare(session)
    if track_from is not None:
        session.encode_prompts()    # SAM2 는 원래 track 첫 프레임 때 함 → 준비 시간에 넣는다
    run.setup_seconds = cost.now() - t0
    if track_from is None:      # 아무것도 못 받음 → 전환 뒤 전부 빈 마스크
        empty = np.zeros(video.size, dtype=bool)
        for f in range(keep_after + 1, end + 1):
            keeper.keep(run, f, empty)
    else:
        run.gpu_peaks[track_from - 1] = cost.gpu_peak_mb()
        for out, seconds in cost.timed(session.track(track_from, end)):
            run.times[out.frame] = seconds
            run.gpu_peaks[out.frame] = cost.gpu_peak_mb()
            if out.frame > keep_after:
                keeper.keep(run, out.frame, out.mask)
                if collect is not None:
                    collect.add(out.frame, out.mask)
    session.close()
    return run


def _row(video, obj, sw, baseline, run, cost_info) -> dict:
    s = sw["frame"]
    row = {"dataset": video.dataset, "video": video.name, "object": obj["object"],
           "switch_name": sw["name"], "switch_frame": s,
           "start": obj["start"], "end": obj["end"], "baseline": baseline.name, "role": baseline.role}
    # 전환 뒤, 정답에 객체가 보이는 프레임만 채점한다.
    visible = [sc for f, sc in sorted(run.scores.items()) if f > s and sc.gt_visible]
    row.update(main_metrics.score_columns(visible))
    row["extra_failure_rate"] = extra_metrics.failure_rate(visible)
    row["extra_agreement"] = extra_metrics.agreement(run.agreement, s)
    row.update(cost_info)
    return row


def evaluate_object(video, obj, small, base) -> list[dict]:
    obj_id, start = obj["object"], obj["start"]
    prompt = video.object_mask(start, obj_id)
    keeper = Keeper(FrameScorer(video, obj_id))

    # ① Full Replay: Base+ 처음 ~ 끝, 한 번. 마스크는 출력 일치도 기준으로 들고 있는다.
    replay_masks = PackedMasks(video.size)
    replay_run = _run_base(base, video, obj,
                           lambda session: no_handoff.full_replay(session, start, prompt),
                           keep_after=start, keeper=keeper, collect=replay_masks)
    keeper.replay = replay_masks
    replay_run.agreement = dict.fromkeys(replay_masks.packed, 1.0)   # 자기 자신과는 항상 같음

    # ② Small 처음 ~ 끝
    small_run, packages = _run_small(small, video, obj, prompt, keeper)

    # ③ 전환 시점 × 비교군
    rows = []
    for sw in obj["switches"]:
        s = sw["frame"]
        pkg = packages[s]
        for baseline in BASELINES:
            if baseline.name == "source_only":
                run = small_run
            elif baseline.name == "full_replay":
                run = replay_run
            else:
                run = _run_base(base, video, obj,
                                lambda session: baseline.prepare(session, pkg),
                                keep_after=s, keeper=keeper)
            cost_info = cost.NO_COST if run is small_run else cost.cost_columns(run, s)
            rows.append(_row(video, obj, sw, baseline, run, cost_info))
    return rows
