"""영상 하나의 객체 하나 × 전환 시점 전부 × 비교군 전부 → 결과 줄 목록.

  ① Small 이 처음 ~ 끝을 한 번 추적한다.
       = Source-only 결과이자, 모든 비교군이 같이 쓰는 전환 전 구간.
       전환 프레임마다 기억 상자(HandoffPackage)를 챙겨 둔다.
  ② Full Replay: Base+ 가 처음 ~ 끝을 한 번 추적한다 (전환 시점과 무관).
  ③ 전환 시점 × 나머지 비교군: 새 Base+ 세션에 비교군마다 다른 것을 넘기고 전환 뒤 ~ 끝 추적.
  ④ 시간·VRAM·다시 본 프레임 수 기록.
  ⑤ 채점: 정답이 모든 프레임에 있으면 J&F·J (영상 전체 / 전환 뒤), [추가] 지표·조건 분류.
          MOSEv2 valid 는 채점 대신 마스크 PNG 저장.
  ⑥ 줄 목록을 돌려준다 (저장·이어하기는 records.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import settings
from benchmark import cost
from benchmark.methods import METHODS, no_handoff
from benchmark.model.memory import HandoffPackage
from benchmark.scoring import extra_metrics, extra_strata, jf

SCORE_KEYS = ("j_whole", "f_whole", "jf_whole", "n_frames_whole",
              "j_post", "f_post", "jf_post", "n_frames_post")


@dataclass
class Run:
    """추적 한 번의 결과: 정답 프레임별 점수, 프레임별 시간, 비용."""
    scores: dict = field(default_factory=dict)   # 프레임 → FrameScore (정답 있는 프레임만)
    times: dict = field(default_factory=dict)    # 프레임 → 초
    setup_seconds: float = 0.0
    peak_vram_mb: float | None = None
    reseen: int = 0


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
        kept = pred if ignore is None else pred & ~ignore
        return jf.FrameScore(
            j=jf.j_score(pred, own, ignore),
            f=jf.f_score(pred, own, ignore),
            gt_visible=bool(own.any()),
            pred_any=bool(kept.any()),
            other_iou=extra_metrics.other_object_iou(pred, labels, ignore, self.obj_id),
        )


def _keep(run, frame, mask, scorer, saver, folder, video, obj_id):
    """한 프레임 결과를 채점하거나(정답 있음) 제출용으로 저장(MOSEv2 valid)."""
    if scorer is not None:
        score = scorer.score(frame, mask)
        if score is not None:
            run.scores[frame] = score
    if saver is not None:
        saver.save(folder, video.name, obj_id, video.frame_name(frame), mask)


def _run_small(small, video, obj, prompt, scorer, saver):
    """① Small 이 처음 ~ 끝. 전환 프레임마다 기억 상자를 챙긴다."""
    obj_id, start, end = obj["object"], obj["start"], obj["end"]
    switch_frames = {sw["frame"] for sw in obj["switches"]}
    keep_recent = settings.EXTRA_RECENT_K + 1
    run, packages, recent, last_visible = Run(), {}, {}, None

    session = small.start(video)
    with cost.peak_vram() as vram:
        t0 = cost.now()
        session.add_prompt(start, prompt)
        run.setup_seconds = cost.now() - t0
        for out, seconds in cost.timed(session.track(start, end)):
            f = out.frame
            run.times[f] = seconds
            _keep(run, f, out.mask, scorer, saver, "small", video, obj_id)
            recent[f] = out.mask
            recent.pop(f - keep_recent, None)
            if out.visible and out.mask.any():
                last_visible = (f, out.mask)
            if f in switch_frames:
                packages[f] = HandoffPackage(
                    switch_frame=f, prompt_frame=start, prompt_mask=prompt,
                    small_memory=session.export_memory(),
                    last_visible=last_visible, recent_masks=dict(recent))
    run.peak_vram_mb = vram.mb
    session.close()
    return run, packages


def _run_base(base, video, obj, prepare, keep_after, scorer, saver, folder):
    """새 Base+ 세션을 prepare 로 준비하고 끝까지 추적. keep_after 보다 뒤 프레임만 결과로 남긴다."""
    obj_id, end = obj["object"], obj["end"]
    run = Run()
    session = base.start(video)
    with cost.peak_vram() as vram:
        t0 = cost.now()
        track_from, run.reseen = prepare(session)
        run.setup_seconds = cost.now() - t0
        if track_from is None:      # 아무것도 못 받음 → 전환 뒤 전부 빈 마스크
            empty = np.zeros(video.size, dtype=bool)
            for f in range(keep_after + 1, end + 1):
                _keep(run, f, empty, scorer, saver, folder, video, obj_id)
        else:
            for out, seconds in cost.timed(session.track(track_from, end)):
                run.times[out.frame] = seconds
                if out.frame > keep_after:
                    _keep(run, out.frame, out.mask, scorer, saver, folder, video, obj_id)
    run.peak_vram_mb = vram.mb
    session.close()
    return run


def _jf_columns(scores: list, suffix: str) -> dict:
    if not scores:
        return {f"j_{suffix}": None, f"f_{suffix}": None, f"jf_{suffix}": None,
                f"n_frames_{suffix}": 0}
    j = float(np.mean([s.j for s in scores]))
    f = float(np.mean([s.f for s in scores]))
    return {f"j_{suffix}": j, f"f_{suffix}": f, f"jf_{suffix}": (j + f) / 2,
            f"n_frames_{suffix}": len(scores)}


def _row(video, obj, sw, method, part, small_run, post_run, cost_info, strata) -> dict:
    s = sw["frame"]
    row = {"dataset": video.dataset, "part": part, "video": video.name, "object": obj["object"],
           "switch_set": sw["set"], "switch_name": sw["name"], "switch_frame": s,
           "start": obj["start"], "end": obj["end"], "method": method.name, "role": method.role}
    if video.has_full_gt:
        # 영상 전체 기준: 프롬프트 프레임 다음 ~ 전환 프레임은 Small, 그 뒤는 방법의 결과.
        pre = {f: sc for f, sc in small_run.scores.items() if obj["start"] < f <= s}
        post = {f: sc for f, sc in post_run.scores.items() if f > s}
        row.update(_jf_columns(list(pre.values()) + list(post.values()), "whole"))
        # 전환 뒤 기준: 전환 뒤, 정답에 객체가 보이는 프레임만.
        row.update(_jf_columns([sc for sc in post.values() if sc.gt_visible], "post"))
        row.update(extra_metrics.compute(pre, post))
        row.update(strata)
    else:
        row.update({k: None for k in SCORE_KEYS})
    row.update(cost_info)
    return row


def evaluate_object(video, obj, small, base, stats, part=None, saver=None) -> list[dict]:
    obj_id, start = obj["object"], obj["start"]
    prompt = video.object_mask(start, obj_id)
    scorer = FrameScorer(video, obj_id) if video.has_full_gt else None
    profile = extra_strata.gt_profile(video, obj_id, start, obj["end"]) if video.has_full_gt else None

    # ① Small 처음 ~ 끝
    small_run, packages = _run_small(small, video, obj, prompt, scorer, saver)

    # ② Full Replay: Base+ 처음 ~ 끝, 한 번
    replay_run = _run_base(base, video, obj,
                           lambda session: (no_handoff.full_replay(session, start, prompt), 0),
                           keep_after=start, scorer=scorer, saver=saver, folder="full_replay")

    # ③ 전환 시점 × 비교군
    rows = []
    for sw in obj["switches"]:
        s = sw["frame"]
        pkg = packages[s]
        strata = extra_strata.classify(profile, video, s) if profile is not None else {}
        for method in METHODS:
            if method.name == "source_only":
                run = small_run
                cost_info = cost.cost_columns(run.times, run.setup_seconds, s, 0,
                                              run.peak_vram_mb, keeps_running=True)
            elif method.name == "full_replay":
                run = replay_run
                cost_info = cost.cost_columns(run.times, run.setup_seconds, s,
                                              no_handoff.full_replay_reseen(s, start),
                                              run.peak_vram_mb)
            else:
                run = _run_base(base, video, obj,
                                lambda session: method.prepare(session, pkg, stats),
                                keep_after=s, scorer=scorer, saver=saver,
                                folder=f"{method.name}__{sw['name']}")
                cost_info = cost.cost_columns(run.times, run.setup_seconds, s, run.reseen,
                                              run.peak_vram_mb)
            rows.append(_row(video, obj, sw, method, part, small_run, run, cost_info, strata))
    return rows
