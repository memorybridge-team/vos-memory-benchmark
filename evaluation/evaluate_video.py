"""영상 하나의 객체 하나 × 전환 시점 전부 × 고른 방법 (비교군 · 본 모델) → 결과 줄 목록.

  ① Full Replay: Base+ 가 처음 ~ 끝을 한 번 추적한다 (전환 시점과 무관).
  ② Small 이 처음 ~ 끝을 한 번 추적한다.
       = Source-only 결과이자, 모든 방법이 같이 쓰는 전환 전 구간.
       전환 프레임마다 기억 상자(HandoffPackage)를 챙겨 둔다.
  ③ 전환 시점 × 나머지 방법: 새 Base+ 세션에 방법마다 다른 것을 넘기고 전환 뒤 ~ 끝 추적.
       ①② 는 고른 방법과 상관없이 늘 돌린다 (③ 이 쓰므로). 줄은 고른 방법 것만 낸다.
  ④ 전환 준비와 replay 구간만 시간·GPU 메모리 기록 (cost.py). Source-only 는 전환 없음.
  ⑤ 채점: 전환 뒤 J·J&F·실패 비율과 프레임별 점수.
  ⑥ 줄 목록을 돌려준다 (저장·이어하기는 records.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import settings
from evaluation import cost
from baseline import no_handoff
from baseline.handoff import HandoffPackage
from evaluation.scoring import jf, main_metrics


@dataclass
class Run:
    """추적 한 번의 결과: 프레임별 점수·시간, GPU 메모리."""
    scores: dict = field(default_factory=dict)     # 프레임 → FrameScore (정답 있는 프레임만)
    times: dict = field(default_factory=dict)      # 프레임 → 초 (Base+ 만)
    setup_seconds: float = 0.0
    gpu_before_mb: float | None = None             # Base+ 세션을 연 직후 GPU 사용량
    gpu_peaks: dict = field(default_factory=dict)  # 프레임 → 그 프레임까지(준비 포함) GPU 최고치


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
    """정답이 있는 프레임의 점수를 보관한다."""
    scorer: FrameScorer

    def keep(self, run: Run, frame: int, mask: np.ndarray) -> None:
        score = self.scorer.score(frame, mask)
        if score is not None:
            run.scores[frame] = score


def _run_small(small, video, obj, prompt, keeper):
    """② Small 이 처음 ~ 끝. 전환 프레임마다 기억 상자를 챙긴다."""
    start, end = obj["start"], obj["end"]
    switch_frames = {sw["frame"] for sw in obj["switches"]}
    run, packages, last_visible = Run(), {}, None

    session = small.start(video)
    session.add_prompt(start, prompt)
    for out in session.track(start, end):
        f = out.frame
        keeper.keep(run, f, out.mask)
        if out.mask.any():
            last_visible = (f, out.mask)
        if f in switch_frames:
            packages[f] = HandoffPackage(
                switch_frame=f, prompt_frame=start, prompt_mask=prompt,
                small_memory=session.export_memory(),
                last_visible=last_visible)
    session.close()
    return run, packages


def _run_base(base, video, obj, prepare, keep_after, keeper):
    """새 Base+ 세션을 prepare 로 준비하고 끝까지 추적. keep_after 보다 뒤 프레임만 결과로 남긴다.

    GPU: 세션을 연 직후 사용량을 적고, 준비가 끝난 때(track_from − 1 칸)와 프레임마다
    그때까지의 최고치를 적는다 → gpu_peaks[s] = Base+ 가 s+1 처리 준비를 마칠 때까지의 최고치.
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
    session.close()
    return run


def _row(video, obj, sw, baseline, run, cost_info) -> dict:
    s = sw["frame"]
    row = {"dataset": video.dataset, "video": video.name, "object": obj["object"],
           "switch_name": sw["name"], "switch_frame": s,
           "start": obj["start"], "end": obj["end"], "baseline": baseline.name, "role": baseline.role,
           "baseline_revision": baseline.revision,
           "evaluation_revision": settings.EVALUATION_REVISION}
    # 전환 뒤, 정답에 객체가 보이는 프레임만 채점한다.
    scored = [(f, sc) for f, sc in sorted(run.scores.items()) if f > s and sc.gt_visible]
    visible = [sc for _, sc in scored]
    row.update(main_metrics.score_columns(visible))
    row["failure_rate"] = main_metrics.failure_rate(visible)
    row["frame_scores"] = [{"frame": f, "frames_after_switch": f - s,
                            "j": sc.j, "f": sc.f, "jf": sc.jf} for f, sc in scored]
    row.update(cost_info)
    return row


def evaluate_object(video, obj, small, base, methods) -> list[dict]:
    """methods = 결과 줄을 낼 방법들 (evaluation/methods.py)."""
    obj_id, start = obj["object"], obj["start"]
    prompt = video.object_mask(start, obj_id)
    keeper = Keeper(FrameScorer(video, obj_id))

    # ① Base+ Native 결과는 객체마다 한 번 실행하고 전환 뒤 구간을 잘라 쓴다.
    replay_run = _run_base(base, video, obj,
                           lambda session: no_handoff.full_replay(session, start, prompt),
                           keep_after=start, keeper=keeper)

    # ② Small 처음 ~ 끝
    small_run, packages = _run_small(small, video, obj, prompt, keeper)

    # ③ 전환 시점 × 방법
    rows = []
    for sw in obj["switches"]:
        s = sw["frame"]
        pkg = packages[s]
        for baseline in methods:
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
