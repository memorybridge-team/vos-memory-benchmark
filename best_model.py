"""학습에 쓸 Best model 고르기.

기준: 전환 뒤 J&F 회복률을 25/50/75% 전환에서 각각 구한 뒤 평균한 값이 가장 큰 모델.
회복률(%) = 방법의 점수 ÷ Full Replay의 점수 × 100  (영상마다 비율을 구한 뒤 평균)
Full Replay = Base+ 로 처음부터 끝까지 돌린 결과 (상한선).

점수 줄 하나: {"video": "abc", "fraction": 0.25, "jf": 0.81}

학습 루프에서:
    best = BestModel("outputs/train_run1")
    for epoch in range(n_epochs):
        train_one_epoch(model)
        best.update(epoch, model, evaluate_on_dev(model), replay_rows)
"""

import json
from collections import defaultdict
from pathlib import Path

import torch

FRACTIONS = (0.25, 0.50, 0.75)


def video_scores(rows, fraction):
    """해당 전환 시점의 줄만 골라 영상별 J&F 평균 (객체가 여러 개면 평균)."""
    scores = defaultdict(list)
    for r in rows:
        if r["fraction"] == fraction:
            scores[r["video"]].append(r["jf"])
    return {video: sum(v) / len(v) for video, v in scores.items()}


def retention(method_rows, replay_rows, fraction):
    """영상마다 방법 ÷ Full Replay × 100 → 영상들의 평균. Full Replay가 0인 영상은 뺌."""
    method = video_scores(method_rows, fraction)
    replay = video_scores(replay_rows, fraction)
    if method.keys() != replay.keys():
        raise ValueError(f"{fraction} 전환: 방법과 Full Replay의 영상 목록이 다름")
    ratios = [method[v] / replay[v] * 100 for v in method if replay[v] > 0]
    return sum(ratios) / len(ratios)


def selection_score(method_rows, replay_rows):
    """시점별 회복률과 그 평균. "score"가 클수록 좋은 모델."""
    result = {f"r{round(f * 100)}": retention(method_rows, replay_rows, f) for f in FRACTIONS}
    result["score"] = sum(result.values()) / len(FRACTIONS)
    return result


class BestModel:
    """epoch마다 점수를 기록하고, 최고 점수가 나오면 모델을 저장.

    out_dir/selection_log.jsonl  epoch마다 한 줄 (r25, r50, r75, score)
    out_dir/best.pt              최고 점수 모델의 state_dict
    out_dir/best.json            최고 점수와 그 epoch
    """

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best_score = float("-inf")
        self.best_epoch = None

    def update(self, epoch, model, method_rows, replay_rows):
        """점수를 기록하고, 새 최고면 모델을 저장하고 True."""
        result = {"epoch": epoch, **selection_score(method_rows, replay_rows)}
        with open(self.out_dir / "selection_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

        if result["score"] <= self.best_score:
            return False
        self.best_score = result["score"]
        self.best_epoch = epoch
        torch.save(model.state_dict(), self.out_dir / "best.pt")
        (self.out_dir / "best.json").write_text(json.dumps(result), encoding="utf-8")
        return True
