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
from evaluation.switches import object_exclusion
from baseline import no_handoff
from baseline.handoff import HandoffPackage
from evaluation.scoring import jf, main_metrics, recovery, restoration


@dataclass
class Run:
    """추적 한 번의 결과: 프레임별 점수·시간, GPU 메모리."""
    feature_snapshots: dict = field(default_factory=dict)  # 전환 시점 Native CPU 기억
    restoration: dict | None = None                     # 추론 단계의 프레임별 계산값
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


def _run_small(small, video, obj, prompt, keeper, *,
               switch_frames=None, memory_frames=None, end_frame=None):
    """Small을 필요한 마지막 프레임까지 처리하고 요청한 전환의 기억만 꺼낸다."""
    start = obj["start"]
    end = obj["end"] if end_frame is None else end_frame
    switch_frames = ({sw["frame"] for sw in obj["switches"]}
                     if switch_frames is None else set(switch_frames))
    memory_frames = switch_frames if memory_frames is None else set(memory_frames)
    run, packages, last_visible = Run(), {}, None

    session = small.start(video)
    try:
        session.add_prompt(start, prompt)
        for out in session.track(start, end):
            f = out.frame
            keeper.keep(run, f, out.mask)
            if out.mask.any():
                last_visible = (f, out.mask)
            if f in switch_frames:
                packages[f] = HandoffPackage(
                    switch_frame=f, prompt_frame=start, prompt_mask=prompt,
                    small_memory=session.export_memory() if f in memory_frames else {},
                    last_visible=last_visible)
    finally:
        session.close()
    return run, packages


def _run_base(base, video, obj, prepare, keep_after, keeper,
              snapshot_frames=(), restoration_reference=None):
    """새 Base+ 세션을 prepare 로 준비하고 끝까지 추적. keep_after 보다 뒤 프레임만 결과로 남긴다.

    GPU: 세션을 연 직후 사용량을 적고, 준비가 끝난 때(track_from − 1 칸)와 replay 프레임마다
    그때까지의 최고치를 적는다 → gpu_peaks[s] = Base+ 가 s+1 처리 준비를 마칠 때까지의 최고치.
    """
    end = obj["end"]
    run = Run()
    session = base.start(video)
    try:
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
            if restoration_reference is not None and track_from == keep_after + 1:
                # 직접 복사/translator/anchor: s+1 추적이 시작되기 직전에 측정.
                run.restoration = restoration.result(session.export_features(), restoration_reference, keep_after)
            # Native는 마지막 전환까지, 다른 방법은 replay의 s까지 측정한다.
            # s+1 이후 추론/채점은 진행하되 전환 비용용 동기화와 통계는 만들지 않는다.
            measure_through = max(snapshot_frames, default=keep_after)
            measure_frames = max(0, measure_through - track_from + 1)
            for out, seconds in cost.timed(session.track(track_from, end), measure_frames):
                if seconds is not None:
                    run.times[out.frame] = seconds
                    run.gpu_peaks[out.frame] = cost.gpu_peak_mb()
                if out.frame in snapshot_frames:
                    run.feature_snapshots[out.frame] = session.export_features()
                if restoration_reference is not None and out.frame == keep_after:
                    # Replay는 s까지 다시 처리한 뒤, s+1을 처리하기 전에 측정.
                    run.restoration = restoration.result(session.export_features(), restoration_reference, keep_after)
                if out.frame > keep_after:
                    keeper.keep(run, out.frame, out.mask)
    finally:
        session.close()
    return run


def _row(video, obj, sw, baseline, run, cost_info, reference, pre_run, run_id, seed,
         reference_scores=None) -> dict:
    s = sw["frame"]
    row = {"dataset": video.dataset, "video": video.name, "object": obj["object"],
           "switch_name": sw["name"], "switch_frame": s,
           "start": obj["start"], "end": obj["end"], "baseline": baseline.name, "role": baseline.role,
           "baseline_revision": baseline.revision,
           "evaluation_revision": settings.EVALUATION_REVISION,
           "runtime_revision": settings.EVALUATION_RUNTIME_REVISION,
           "cost_runtime_revision": (None if baseline.name == 'source_only'
                                     else reference.get('runtime_revision', 1) if baseline.name == 'full_replay'
                                     else settings.EVALUATION_RUNTIME_REVISION),
           "frame_prefetch": settings.FRAME_PREFETCH,
           "run_id": run_id, "seed": seed,
           "native_reference_id": reference["native_reference_id"]}
    # 전환 뒤, 정답에 객체가 보이는 프레임만 채점한다.
    scored = [(f, sc) for f, sc in sorted(run.scores.items()) if f > s and sc.gt_visible]
    visible = [sc for _, sc in scored]
    row.update(main_metrics.score_columns(visible))
    row["failure_rate"] = main_metrics.failure_rate(visible)
    if reference_scores is None:
        reference_scores = native.scores(reference)
    row["frame_scores"] = [recovery.raw_point(f, sc, reference_scores.get(f), s) for f, sc in scored]
    row.update(recovery.columns(row["frame_scores"]))
    row["pre_switch_frame_scores"] = [
        recovery.raw_point(f, sc, reference_scores.get(f), s)
        for f, sc in sorted(pre_run.scores.items()) if f <= s and sc.gt_visible]
    row.update(recovery.phase_columns(row['pre_switch_frame_scores'], 'pre'))
    row.update(recovery.phase_columns(row['frame_scores'], 'post'))
    row['recovery_statistic'] = 'ratio_of_means'
    row['recovery_reference'] = 'pending'
    row['pending_native_n_frames'] = len(row['frame_scores'])
    row.update(cost_info)
    return row


def evaluate_object(video, obj, small, base, methods, run_id=1, seed=None,
                    native_reference=None, save_native=None, pending_conditions=None) -> list[dict]:
    """각 회차 Native를 한 번만 실행/저장하고 모든 방법·25/50/75%에 공유한다."""
    # 직접 호출도 짧은 객체의 모든 전환을 제외한다. 모델/프롬프트/Native 실행 전 검사.
    if object_exclusion(obj) is not None:
        return []
    seed = settings.EVALUATION_SEED if seed is None else seed
    conditions = {(sw['name'], sw['frame'], m.name) for sw in obj['switches'] for m in methods}
    if pending_conditions is not None:
        conditions &= set(pending_conditions)
    if not conditions:
        return []
    obj_id, start = obj["object"], obj["start"]
    prompt = video.object_mask(start, obj_id)
    keeper = Keeper(FrameScorer(video, obj_id))

    reference = native_reference
    if reference is None:
        repeats.seed_condition(seed, run_id, video.dataset, video.name, obj_id, "full_replay")
        replay_run = _run_base(base, video, obj,
                              lambda session: no_handoff.full_replay(session, start, prompt),
                              keep_after=start - 1, keeper=keeper,
                              snapshot_frames={sw["frame"] for sw in obj["switches"]})
        reference = native.pack(video, obj, replay_run, run_id, seed)
        native_memories = replay_run.feature_snapshots
        if save_native is not None:
            save_native(reference, native_memories)
    elif native.reference_key(reference) != native.case_key(video, obj, run_id, seed):
        raise ValueError("Native의 영상·객체·전환·회차·seed가 평가 조건과 다릅니다.")
    if native_reference is not None:
        native_memories = (native.load_memories(reference)
                           if any(name != 'source_only' for _, _, name in conditions) else {})
    reference_scores = native.scores(reference)
    replay_run = Run(scores=reference_scores)

    # 완료한 전환을 재실행하지 않는다. Source-only가 필요할 때만 Small을 끝까지 돌린다.
    small_conditions = {(name, f, method) for name, f, method in conditions if method != 'full_replay'}
    small_run, packages = Run(), {}
    if small_conditions:
        repeats.seed_condition(seed, run_id, video.dataset, video.name, obj_id, "source_only")
        handoff_frames = {f for _, f, method in small_conditions if method != 'source_only'}
        memory_frames = {f for _, f, method in small_conditions
                         if method in ('direct_state_copy', 'translator')}
        end_frame = (obj['end'] if any(method == 'source_only' for _, _, method in small_conditions)
                     else max(f for _, f, _ in small_conditions))
        small_run, packages = _run_small(small, video, obj, prompt, keeper,
                                        switch_frames=handoff_frames, memory_frames=memory_frames,
                                        end_frame=end_frame)

    rows = []
    for sw in obj["switches"]:
        s = sw["frame"]
        for baseline in methods:
            if (sw['name'], s, baseline.name) not in conditions:
                continue
            if baseline.name == "source_only":
                run, cost_info = small_run, cost.NO_COST
            elif baseline.name == "full_replay":
                run, cost_info = replay_run, reference["costs"][str(s)]
            else:
                repeats.seed_condition(seed, run_id, video.dataset, video.name, obj_id,
                                       f"{baseline.name}/{sw['name']}/{s}")
                pkg = packages[s]
                run = _run_base(base, video, obj,
                                lambda session: baseline.prepare(session, pkg),
                                keep_after=s, keeper=keeper, restoration_reference=native_memories[s])
                cost_info = cost.cost_columns(run, s)
            pre_run = replay_run if baseline.name == "full_replay" else small_run
            row = _row(video, obj, sw, baseline, run, cost_info,
                       reference, pre_run, run_id, seed, reference_scores=reference_scores)
            if baseline.name == "source_only":
                row.update(restoration.not_applicable(s))
            elif baseline.name == "full_replay":
                row.update(restoration.result(native_memories[s], native_memories[s], s))
            elif run.restoration is not None:
                row.update(run.restoration)
            else:
                raise ValueError(f'전환 시점의 복원율 측정이 누락되었습니다: {baseline.name}, {s}')
            # 원본에는 칸별 R²/SSE/SST만 저장한다.
            # 전체 기억 R²과 객체/영상/회차 요약은 3_make_tables.py에서 재계산한다.
            rows.append(row)
    return rows
