"""영상 하나의 객체 하나 × 전환 시점 전부 × 고른 방법 (비교군 · 본 모델) → 결과 줄 목록.

  ① Full Replay: Base+ 가 처음 ~ 끝을 한 번 추적한다 (전환 시점과 무관).
  ② Small 이 처음 ~ 끝을 한 번 추적한다.
       = Source-only 결과이자, 모든 방법이 같이 쓰는 전환 전 구간.
       전환 프레임마다 기억 상자(HandoffPackage)를 챙겨 둔다.
  ③ 전환 시점 × 나머지 방법: 새 Base+ 세션에 방법마다 다른 것을 넘기고 전환 뒤 ~ 끝 추적.
       ① Native는 회차별 저장된 기준이 있으면 재사용한다. Small은 필요한 객체마다 실행한다.
  ④ 전환 준비와 replay 구간만 시간·GPU 메모리 기록 (cost.py). Source-only 는 전환 없음.
  ⑤ 채점: J·J&F·실패 비율·프레임별 회복률과 전환 전후 곡선.
  ⑥ 줄 목록을 돌려준다 (저장·이어하기는 records.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import settings
from evaluation import cost, native, repeats
from baseline import no_handoff
from baseline.handoff import HandoffPackage
from evaluation.scoring import jf, main_metrics, recovery


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


def _row(video, obj, sw, baseline, run, cost_info, reference, pre_run, run_id, seed) -> dict:
    s = sw["frame"]
    row = {"dataset": video.dataset, "video": video.name, "object": obj["object"],
           "switch_name": sw["name"], "switch_frame": s,
           "start": obj["start"], "end": obj["end"], "baseline": baseline.name, "role": baseline.role,
           "baseline_revision": baseline.revision,
           "evaluation_revision": settings.EVALUATION_REVISION,
           "run_id": run_id, "seed": seed,
           "native_reference_id": reference["native_reference_id"]}
    # 전환 뒤, 정답에 객체가 보이는 프레임만 채점한다.
    scored = [(f, sc) for f, sc in sorted(run.scores.items()) if f > s and sc.gt_visible]
    visible = [sc for _, sc in scored]
    row.update(main_metrics.score_columns(visible))
    row["failure_rate"] = main_metrics.failure_rate(visible)
    reference_scores = native.scores(reference)
    row["frame_scores"] = [recovery.raw_point(f, sc, reference_scores.get(f), s) for f, sc in scored]
    row.update(recovery.columns(row["frame_scores"]))
    row["pre_switch_frame_scores"] = [
        recovery.raw_point(f, sc, reference_scores.get(f), s)
        for f, sc in sorted(pre_run.scores.items()) if f <= s and sc.gt_visible]
    row['recovery_reference'] = 'pending'
    row['pending_native_n_frames'] = len(row['frame_scores'])
    row.update(cost_info)
    return row


def evaluate_object(video, obj, small, base, methods, run_id=1, seed=None,
                    native_reference=None, save_native=None) -> list[dict]:
    """각 회차 Native를 한 번만 실행/저장하고 모든 방법·50/75%에 공유한다."""
    seed = settings.EVALUATION_SEED if seed is None else seed
    obj_id, start = obj["object"], obj["start"]
    prompt = video.object_mask(start, obj_id)
    keeper = Keeper(FrameScorer(video, obj_id))

    reference = native_reference
    if reference is None:
        repeats.seed_condition(seed, run_id, video.dataset, video.name, obj_id, "full_replay")
        replay_run = _run_base(base, video, obj,
                              lambda session: no_handoff.full_replay(session, start, prompt),
                              keep_after=start - 1, keeper=keeper)
        reference = native.pack(video, obj, replay_run, run_id, seed)
        if save_native is not None:
            save_native(reference)
    elif native.reference_key(reference) != native.case_key(video, obj, run_id, seed):
        raise ValueError("Native의 영상·객체·전환·회차·seed가 평가 조건과 다릅니다.")
    replay_run = Run(scores=native.scores(reference))

    repeats.seed_condition(seed, run_id, video.dataset, video.name, obj_id, "source_only")
    small_run, packages = _run_small(small, video, obj, prompt, keeper)

    rows = []
    for sw in obj["switches"]:
        s = sw["frame"]
        pkg = packages[s]
        for baseline in methods:
            if baseline.name == "source_only":
                run, cost_info = small_run, cost.NO_COST
            elif baseline.name == "full_replay":
                run, cost_info = replay_run, reference["costs"][str(s)]
            else:
                repeats.seed_condition(seed, run_id, video.dataset, video.name, obj_id,
                                       f"{baseline.name}/{sw['name']}/{s}")
                run = _run_base(base, video, obj,
                                lambda session: baseline.prepare(session, pkg),
                                keep_after=s, keeper=keeper)
                cost_info = cost.cost_columns(run, s)
            pre_run = replay_run if baseline.name == "full_replay" else small_run
            rows.append(_row(video, obj, sw, baseline, run, cost_info,
                             reference, pre_run, run_id, seed))
    return rows
