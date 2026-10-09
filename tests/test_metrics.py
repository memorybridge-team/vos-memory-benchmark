"""지표 수식과 전환 구간의 경계를 작은 예제로 검증한다.

    python tests/test_metrics.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import settings
from evaluation import cost
from evaluation.scoring import jf, main_metrics
from evaluation.switches import switch_points


def test_metrics():
    pred = np.zeros((20, 20), dtype=bool)
    gt = pred.copy()
    pred[5:10, 6:11] = True
    gt[5:10, 5:10] = True
    assert abs(jf.j_score(pred, gt) - 2 / 3) < 1e-12
    ignore = pred ^ gt
    assert jf.j_score(pred, gt, ignore) == 1.0
    assert jf.f_score(gt, gt) == 1.0
    assert jf.f_score(np.zeros_like(gt), gt) == 0.0

    # 마스크 내부의 무시 영역이 새 경계나 새 F 대응점을 만들면 안 된다.
    solid = np.zeros((40, 40), dtype=bool)
    solid[5:35, 5:35] = True
    void = np.zeros_like(solid)
    void[15:25, 15:25] = True
    assert np.array_equal(jf._boundary(solid & ~void, void), jf._boundary(solid))
    shifted = np.roll(solid, 1, axis=1)
    assert abs(jf.f_score(shifted, solid, void) - jf.f_score(shifted, solid)) < 1e-12
    # 무시 영역 안에만 있는 예측은 유효한 경계를 갖지 않는다.
    assert not jf._boundary(void, void).any()

    # 모든 테두리 점 쌍의 거리로 정답을 직접 계산한다.
    a, b = np.argwhere(jf._boundary(pred)), np.argwhere(jf._boundary(gt))
    distances = np.linalg.norm(a[:, None] - b[None, :], axis=-1)
    threshold = np.hypot(*pred.shape) * settings.BOUNDARY_THRESHOLD
    precision = np.mean(distances.min(axis=1) <= threshold)
    recall = np.mean(distances.min(axis=0) <= threshold)
    expected = 2 * precision * recall / (precision + recall)
    assert abs(jf.f_score(pred, gt) - expected) < 1e-12
    assert expected < 1, "1픽셀 올림 허용 거리를 사용하면 이 경우 잘못 1점이 된다"

    scores = [jf.FrameScore(0.5, 0.2, True), jf.FrameScore(0.6, 0.8, True)]
    columns = main_metrics.score_columns(scores)
    assert abs(columns["j"] - 0.55) < 1e-12
    assert abs(columns["jf"] - (0.55 + 0.5) / 2) < 1e-12
    assert main_metrics.failure_rate(scores) == 0.5
    assert main_metrics.failure_rate([]) is None

    # 준비 2초 + s까지 replay 7초. s+1 뒤 100초는 전환시간에서 제외한다.
    run = SimpleNamespace(setup_seconds=2.0, times={7: 3.0, 8: 4.0, 9: 100.0},
                          gpu_before_mb=100.0, gpu_peaks={8: 140.0, 9: 900.0})
    result = cost.cost_columns(run, 8)
    assert result == {"switch_seconds": 9.0, "switch_gpu_mb": 40.0}, result
    run.gpu_before_mb = None
    run.gpu_peaks[8] = None
    assert cost.cost_columns(run, 8)["switch_gpu_mb"] is None
    # 복사 방법처럼 replay가 없을 때는 준비 시간만 계산한다.
    run.times = {9: 100.0}
    assert cost.cost_columns(run, 8)["switch_seconds"] == 2.0

    assert settings.SWITCH_FRACTIONS == (0.25, 0.5, 0.75)
    assert switch_points(5, 29) == []
    assert switch_points(0, 9) == []
    assert switch_points(0, 11) == []
    assert switch_points(5, 15) == []
    assert switch_points(0, 12) == []
    assert switch_points(0, 32) == [{"name": "25", "frame": 8}, {"name": "50", "frame": 16}, {"name": "75", "frame": 24}]
    assert switch_points(0, 29) == []
    assert switch_points(0, 30)[0]["frame"] == switch_points(0, 31)[0]["frame"] == 8
    assert switch_points(0, 34)[0] == {"name": "25", "frame": 8}  # round(8.5)=8
    assert switch_points(0, 38)[0] == {"name": "25", "frame": 10} # round(9.5)=10
    # 한 영상에 객체가 많아도 영상의 가중치는 동일하다. 가시성이 없는 시간점은 섞지 않는다.
    from evaluation.tables import temporal
    rows = [
        {"dataset": "m3vos", "switch_name": "50", "baseline": "source_only", "video": "a",
         "frame_scores": [{"frames_after_switch": 1, "j": 0.0, "jf": 0.2}]},
        {"dataset": "m3vos", "switch_name": "50", "baseline": "source_only", "video": "a",
         "frame_scores": [{"frames_after_switch": 1, "j": 0.0, "jf": 0.2}]},
        {"dataset": "m3vos", "switch_name": "50", "baseline": "source_only", "video": "b",
         "frame_scores": [{"frames_after_switch": 1, "j": 1.0, "jf": 0.8},
                          {"frames_after_switch": 3, "j": 0.4, "jf": 0.6}]},
    ]
    points = temporal.build(rows)
    assert points[0]["j"] == 50.0 and points[0]["jf"] == 50.0
    assert points[0]["video_count"] == 2 and points[0]["object_count"] == 3
    assert [p["frames_after_switch"] for p in points] == [1, 3]
    assert points[1]["j"] == 40.0 and points[1]["video_count"] == 1
    print("OK: J·F·J&F, 엄격한 J > 0.5, 전환 비용 경계, 50/75%, 경과 프레임 집계")


if __name__ == "__main__":
    test_metrics()
