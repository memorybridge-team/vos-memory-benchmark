"""가짜 모델·데이터로 50/75% 평가, 새 지표, 난이도·시계열 표와 이어하기를 검증한다.

    python tests/test_end_to_end.py
"""

import csv
import json
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import settings
import translator
from baseline import MAIN
from evaluation import records
from evaluation.data import DATASETS, load_dataset, load_video_list, video_list_path
from evaluation.methods import METHODS, to_run
from evaluation.scoring.main_metrics import retention
from model import sam2_check, sam2_runner

import fake_data
import fake_sam2

REMOVED_COLUMNS = {"seconds", "extra_agreement", "extra_failure_rate", "extra_switch_gpu_mb",
                   "seconds_per_frame_after", "reseen_frames", "restoration", "r2"}


def run_script(name: str, *args: str) -> None:
    sys.argv = [name, *args]
    runpy.run_path(str(ROOT / "scripts" / name), run_name="__main__")


def setup(tmp: Path) -> None:
    settings.DATA_ROOT = str(tmp / "data")
    settings.OUTPUT_ROOT = str(tmp / "outputs")
    fake_data.make_all(Path(settings.DATA_ROOT), settings.DATA_FOLDERS)
    sam2_runner.load_runner = fake_sam2.FakeRunner
    translator.load = lambda: setattr(translator, "_translator", fake_sam2.FakeTranslator())


def dataset_rows(dataset: str) -> list[dict]:
    return [r for path in sorted(records.records_path(dataset).parent.glob(f"{dataset}*.jsonl"))
            for r in records.read_rows(path)]


def keep_only_model(dataset: str) -> None:
    path = records.records_path(dataset)
    rows = [r for r in records.read_rows(path)
            if r["baseline"] in ("translator", "source_only", "full_replay")]
    path.unlink()
    records.append_rows(path.with_name(f"{dataset}.model.jsonl"), rows)


def check_rows(dataset: str) -> None:
    rows = dataset_rows(dataset)
    assert rows and len(records.unique_rows(rows)) == len(rows)
    for entry in load_video_list(dataset)["videos"]:
        for obj in entry["objects"]:
            assert [sw["name"] for sw in obj["switches"]] == ["50", "75"]
            for sw in obj["switches"]:
                got = {r["baseline"]: r for r in rows if r["video"] == entry["video"]
                       and r["object"] == obj["object"] and r["switch_name"] == sw["name"]}
                assert set(got) == {m.name for m in METHODS}
                assert got["translator"]["jf"] == got["direct_state_copy"]["jf"]
    for row in rows:
        assert not REMOVED_COLUMNS.intersection(row)
        assert row["evaluation_revision"] == settings.EVALUATION_REVISION
        assert row["jf"] is not None and row["n_frames"] > 0
        assert 0 <= row["failure_rate"] <= 1
        assert row["n_frames"] == len(row["frame_scores"])
        assert abs(row["failure_rate"] - sum(p["j"] <= 0.5 for p in row["frame_scores"]) / row["n_frames"]) < 1e-12
        assert abs(row["jf"] - sum(p["jf"] for p in row["frame_scores"]) / row["n_frames"]) < 1e-12
        for point in row["frame_scores"]:
            assert point["frames_after_switch"] == point["frame"] - row["switch_frame"] > 0
        if row["baseline"] == "source_only":
            assert row["switch_seconds"] is None and row["switch_gpu_mb"] is None
        else:
            assert row["switch_seconds"] > 0
        assert "switch_gpu_mb" in row

    replay = [r for r in rows if r["baseline"] == "full_replay"]
    for key in ("j", "jf"):
        assert abs(retention(replay, replay, key) - 100) < 1e-9
    for row in replay:
        if row["switch_name"] == "50":
            later = next(r for r in replay if r["video"] == row["video"] and r["object"] == row["object"] and r["switch_name"] == "75")
            assert later["switch_seconds"] >= row["switch_seconds"], (row, later)

    # 객체 2는 12~17 프레임에 안 보인다. 경과 프레임을 압축해 새로 번호 매기지 않는다.
    hidden = next(r for r in rows if r["object"] == 2 and r["switch_name"] == "50")
    assert hidden["frame_scores"][0]["frames_after_switch"] == 4
    print(f"  OK {dataset}: {len(rows)}줄, 50/75%, 지표와 전환 비용")


def check_void() -> None:
    ids = {obj["object"] for entry in load_video_list("m3vos")["videos"] for obj in entry["objects"]}
    assert 255 not in ids and ids


def check_tables() -> None:
    tables = Path(settings.OUTPUT_ROOT) / "tables"
    main = (tables / "main.md").read_text(encoding="utf-8")
    difficulty = (tables / "extra.md").read_text(encoding="utf-8")
    for method in METHODS:
        assert f"| {method.label} |" in main and f"| {method.label} |" in difficulty
    for column in ("회복률 J", "회복률 J&F", "전환시간(초)", "전환 GPU 메모리(MB)", "실패 비율(%)"):
        assert column in main
    for gone in ("출력 일치도", "진단 비교군", "25%", "R^2", "복원율(", "전체 추적시간"):
        assert gone not in main + difficulty
    assert "### 전환 50%" in main and "### 전환 75%" in main
    for label in ("OCC 가림", "FM 빠른 움직임", "변형:break", "상태 변화:melt", "상태:solid→liquid"):
        assert label in difficulty
    assert "공통 난이도 유형" in difficulty and "가려짐" in difficulty and "모양·상태 변화" in difficulty
    with (tables / "temporal.csv").open(encoding="utf-8-sig", newline="") as stream:
        points = list(csv.DictReader(stream))
    assert points and {p["switch_name"] for p in points} == {"50", "75"}
    assert {p["baseline"] for p in points} == {m.name for m in METHODS}
    assert all(int(p["frames_after_switch"]) > 0 and 0 <= float(p["jf"]) <= 100 for p in points)
    print("  OK 결과표: 전환별 요약, 난이도 성능, temporal.csv")


def check_legacy_results(dataset: str) -> None:
    rows = dataset_rows(dataset)
    source = next(r for r in rows if r["baseline"] == "source_only")
    last = next(r for r in rows if r["baseline"] == "original_last_visible")
    incompatible = [
        {k: v for k, v in source.items() if k != "evaluation_revision"},
        dict(last, baseline_revision=1, video="legacy_only"),
        dict(source, baseline="original_replay_16"),
        dict(source, baseline="reset", role="extra"),
        dict(source, switch_name="25"),
    ]
    assert records.current_rows(incompatible) == []
    path = records.records_path(dataset).with_name(f"{dataset}.legacy.jsonl")
    records.append_rows(path, incompatible)
    assert records.row_key(incompatible[1]) not in records.done_keys(dataset)
    loaded = runpy.run_path(str(ROOT / "scripts/3_make_tables.py"))["load_rows"]()
    assert all(r["video"] != "legacy_only" and r["switch_name"] != "25" for r in loaded)
    assert all(r["baseline"] not in ("reset", "original_replay_16") for r in loaded)
    print("  OK 이전 평가 기준과 제거된 비교군 결과 제외")


def check_partial_resume(dataset: str) -> None:
    path = records.records_path(dataset)
    rows = records.read_rows(path)
    missing = next(r for r in rows if r["baseline"] == "translator" and r["switch_name"] == "75")
    path.unlink()
    records.append_rows(path, [r for r in rows if r is not missing])
    run_script("2_evaluate.py", "--dataset", dataset)
    completed = records.read_rows(path)
    assert len(completed) == len(rows)
    assert len(records.unique_rows(completed)) == len(completed)
    assert any(records.row_key(r) == records.row_key(missing) for r in completed)
    print("  OK 일부 전환만 빠진 객체도 중복 없이 이어서 평가")


def check_old_list_rejected(dataset: str) -> None:
    path = video_list_path(dataset)
    original = path.read_text(encoding="utf-8")
    data = json.loads(original)
    data["switch_fractions"] = [0.25, 0.5, 0.75]
    path.write_text(json.dumps(data), encoding="utf-8")
    try:
        load_video_list(dataset)
        raise AssertionError("예전 영상 목록을 허용함")
    except ValueError as error:
        assert "1_make_video_list.py" in str(error)
    finally:
        path.write_text(original, encoding="utf-8")


def check_roundtrip() -> None:
    video = load_dataset("lvos_v2_valid")[0]
    runner = fake_sam2.FakeRunner("base_plus")
    r = sam2_check.roundtrip(runner, video, obj_id=1, start=0, cut=10, last=25)
    assert r["identical_frames"] == r["frames"] > 0, r
    print(f"  OK 꺼냈다 넣기: {r['identical_frames']}/{r['frames']} 프레임 동일")



def check_baseline_contract() -> None:
    """정확히 5개 비교군, Replay 범위, 마지막 비어 있지 않은 예측 선택을 확인한다."""
    from types import SimpleNamespace

    import numpy as np

    from baseline import anchors, no_handoff
    from baseline.handoff import HandoffPackage
    from evaluation.evaluate_video import _run_small

    expected = {
        "source_only", "full_replay", "direct_state_copy",
        "original_last_visible", "original_replay_8",
    }
    assert {m.name for m in MAIN} == expected and len(MAIN) == 5
    assert {m.name for m in to_run()} == expected | {"translator"}
    assert settings.REPLAY_FRAMES == 8

    class PromptSession:
        def __init__(self):
            self.prompts = []
            self.loaded = None

        def add_prompt(self, frame, mask):
            self.prompts.append((frame, mask))

        def load_memory(self, memory):
            self.loaded = memory

    prompt = np.ones((2, 2), dtype=bool)
    empty = np.zeros_like(prompt)

    class SmallSession:
        def add_prompt(self, frame, mask):
            pass

        def track(self, first, last):
            for f in range(first, last + 1):
                # 객체 점수가 낮아도 비어 있지 않은 예측을 고른다. 뒤의 빈 예측은 제외한다.
                yield sam2_runner.FrameOut(f, prompt if f in (3, 8) else empty, False)

        def export_memory(self):
            return {"sentinel": "small memory"}

        def close(self):
            pass

    obj = {"start": 0, "end": 12, "switches": [{"frame": 6}, {"frame": 10}]}
    small = SimpleNamespace(start=lambda video: SmallSession())
    keeper = SimpleNamespace(keep=lambda run, frame, mask: None)
    _, packages = _run_small(small, None, obj, prompt, keeper)
    assert packages[6].last_visible[0] == 3
    assert packages[10].last_visible[0] == 8

    session = PromptSession()
    assert anchors.original_last_visible(session, packages[10]) == 11
    assert [f for f, _ in session.prompts] == [0, 8]
    assert session.prompts[1][1] is prompt and session.loaded is None

    # 처음 프레임 중복과 비어 있지 않은 예측이 없는 경우에는 처음 정답만 사용한다.
    for last_visible in (None, (0, empty)):
        pkg = HandoffPackage(10, 0, prompt, {}, last_visible)
        session = PromptSession()
        anchors.original_last_visible(session, pkg)
        assert len(session.prompts) == 1 and session.prompts[0][1] is prompt

    replay = next(m for m in MAIN if m.name == "original_replay_8")
    for switch, expected_first in ((20, 13), (3, 1)):
        pkg = HandoffPackage(switch, 0, prompt, {}, None)
        session = PromptSession()
        first = replay.prepare(session, pkg)
        assert first == expected_first
        assert switch - first + 1 == min(8, switch)
        assert len(session.prompts) == 1 and session.prompts[0][0] == 0
        assert session.loaded is None

    session = PromptSession()
    direct = next(m for m in MAIN if m.name == "direct_state_copy")
    assert direct.prepare(session, packages[10]) == 11
    assert session.loaded is packages[10].small_memory and not session.prompts

    session = PromptSession()
    assert no_handoff.full_replay(session, 0, prompt) == 0
    assert session.prompts[0][1] is prompt and session.loaded is None
    print("  OK 비교군 5개: 마지막 비어 있지 않은 예측, 두 프롬프트, Replay-8 범위")


def test_end_to_end():
    tmp = Path(tempfile.mkdtemp(prefix="benchmark_test_"))
    try:
        setup(tmp)
        run_script("1_make_video_list.py")
        for dataset in DATASETS:
            run_script("2_evaluate.py", "--dataset", dataset)
        keep_only_model("pumavos")
        run_script("2_evaluate.py", "--dataset", "pumavos")
        before = len(dataset_rows("vost_val"))
        run_script("2_evaluate.py", "--dataset", "vost_val")
        assert len(dataset_rows("vost_val")) == before

        for dataset in DATASETS:
            check_rows(dataset)
        check_void()
        check_baseline_contract()
        check_partial_resume("m3vos")
        check_legacy_results("vost_val")
        check_old_list_rejected("m3vos")
        run_script("3_make_tables.py")
        check_tables()
        check_roundtrip()
        print("모두 통과")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_end_to_end()
