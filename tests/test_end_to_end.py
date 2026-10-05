"""가짜 모델·가짜 데이터로 전체 흐름 확인.

    python tests/test_end_to_end.py

확인하는 것:
  - 1 → 2 → 3 이 끝까지 돈다 (LVOS v2 valid, VOST, M3VOS, PUMaVOS)
  - 모든 비교군(주 9 + 추가 2)이 모든 전환 시점에 결과 줄을 남긴다
  - 빼기로 한 열(전환 지연, 프레임당 시간, drift, ID 뒤바뀜, fit/dev ...)이 없다
  - 시간: Source-only·reset 은 없음, 나머지는 있음, Full Replay 는 전환 시점과 무관하게 같음
  - 실패 비율은 0~1, 전환 GPU 메모리 열이 있다 (GPU 가 없으면 값은 None)
  - Full Replay: 회복률 100, 격차 회복률 100 / Source-only 격차 회복률 0
  - M3VOS 정답의 255 는 객체가 아니다
  - 주 표(main.md)와 추가 표(extra.md)가 나온다
  - 다시 실행하면 이미 끝난 객체는 건너뛴다
  - 기억 꺼냈다 넣기 결과가 끊지 않은 결과와 같다 (sam2_check.roundtrip)
  - 본 모델: 받은 칸을 그대로 돌려주는 가짜 translator 면 Direct State Copy 와 점수가 같다
  - 본 모델·비교군을 따로 돌려도 (--methods) 줄이 빠지거나 겹치지 않는다 (PUMaVOS 로 확인)
"""

import runpy
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import settings  # noqa: E402
import translator  # noqa: E402
from evaluation import records  # noqa: E402
from evaluation.data import DATASETS, load_dataset, load_video_list  # noqa: E402
from evaluation.methods import METHODS  # noqa: E402
from baseline import EXTRA, MAIN  # noqa: E402
from model import sam2_check, sam2_runner  # noqa: E402
from evaluation.scoring.main_metrics import gap_retention, retention  # noqa: E402

import fake_data  # noqa: E402
import fake_sam2  # noqa: E402

REMOVED_COLUMNS = ("jf_whole", "j_whole", "jf_post", "reseen_frames", "extra_jf_at_n", "extra_switch_shock",
                   "extra_recovery_frames", "extra_absent_false_alarm", "extra_stratum_occlusion",
                   "switch_seconds", "seconds_per_frame_after", "peak_vram_mb", "extra_drift_jf",
                   "extra_failure", "extra_id_switch_rate", "part")


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
    folder = records.records_path(dataset).parent
    return [r for path in sorted(folder.glob(f"{dataset}*.jsonl")) for r in records.read_rows(path)]


def check_rows(dataset: str) -> None:
    rows = dataset_rows(dataset)
    assert rows, f"{dataset}: 결과 줄 없음"
    assert len(records.unique_rows(rows)) == len(rows), f"{dataset}: 같은 줄이 두 번 있음"
    for entry in load_video_list(dataset)["videos"]:
        for obj in entry["objects"]:
            for sw in obj["switches"]:
                got = {r["baseline"]: r for r in rows if r["video"] == entry["video"]
                       and r["object"] == obj["object"] and r["switch_name"] == sw["name"]}
                assert set(got) == {m.name for m in METHODS}, (dataset, entry["video"], sw, set(got))
                assert got["translator"]["jf"] == got["direct_state_copy"]["jf"], (dataset, entry["video"], sw)
    for r in rows:
        assert not any(k in r for k in REMOVED_COLUMNS), r
        assert r["jf"] is not None and r["n_frames"] > 0, r
        assert r["extra_agreement"] is not None, r
        assert 0 <= r["extra_failure_rate"] <= 1, r
        assert "extra_switch_gpu_mb" in r, r
        if r["baseline"] in ("reset", "source_only"):
            assert r["seconds"] is None and r["extra_switch_gpu_mb"] is None, r
        else:
            assert r["seconds"] > 0, r
        if r["role"] in ("main", "model"):     # [추가] recent_k_only 는 객체가 안 보일 때 시작하면 0점이 맞음
            assert r["jf"] > 0.3, r
    replay = [r for r in rows if r["baseline"] == "full_replay"]
    source = [r for r in rows if r["baseline"] == "source_only"]
    direct = [r for r in rows if r["baseline"] == "direct_state_copy"]
    for key in ("jf", "j"):
        assert abs(retention(replay, replay, key) - 100) < 1e-9
        assert retention(direct, replay, key) is not None, f"{dataset}: 회복률 {key} 없음"
        assert abs(gap_retention(replay, source, replay, key) - 100) < 1e-9
        assert abs(gap_retention(source, source, replay, key)) < 1e-9
    assert all(r["extra_agreement"] == 1.0 for r in replay)
    replay_seconds = {}
    for r in replay:     # Full Replay 는 한 번 돌리고 전환 시점마다 잘라 씀 → 시간이 같아야 함
        replay_seconds.setdefault((r["video"], r["object"]), set()).add(r["seconds"])
    assert all(len(v) == 1 for v in replay_seconds.values()), replay_seconds
    small = [r["jf"] for r in source]
    base = [r["jf"] for r in replay]
    assert sum(base) / len(base) > sum(small) / len(small), "가짜 Base+ 가 Small 보다 좋아야 함"
    print(f"  OK {dataset}: 줄 {len(rows)}개")


def check_void() -> None:
    objects = {o["object"] for e in load_video_list("m3vos")["videos"] for o in e["objects"]}
    assert 255 not in objects and objects, objects
    print(f"  OK m3vos: 255 는 객체 아님 (객체 {sorted(objects)})")


def check_tables() -> None:
    tables = Path(settings.OUTPUT_ROOT) / "tables"
    main_md = (tables / "main.md").read_text(encoding="utf-8")
    extra_md = (tables / "extra.md").read_text(encoding="utf-8")
    for m in MAIN + [translator.MODEL]:
        assert f"| {m.label} |" in main_md, m.label
    for m in EXTRA:
        assert m.label not in main_md and f"| {m.label} |" in extra_md, m.label
    assert "Moment-Matched" not in main_md + extra_md
    for column in ("회복률 J&F", "격차 회복률 J", "시간(초)"):
        assert column in main_md, column
    for column in ("속도 배수", "전환 지연"):
        assert column not in main_md, column
    for section in ("비용 세부", "출력 일치도", "진단 비교군", "실패 분석", "공식 라벨별"):
        assert f"### {section}" in extra_md, section
    for gone in ("drift", "입력 길이", "ID 뒤바뀜", "프레임당"):
        assert gone not in extra_md, gone
    assert "전환 GPU 메모리(MB)" in extra_md and "실패 비율(%)" in extra_md
    assert not list(tables.glob("*.png"))
    vost = main_md.split("## vost_val")[1].split("##")[0]
    assert "주 지표: J\n" in vost
    lvos = main_md.split("## lvos_v2_valid")[1].split("##")[0]
    replay_line = next(line for line in lvos.splitlines() if line.startswith("| Full Replay |"))
    cells = [c.strip() for c in replay_line.strip("|").split("|")]
    assert cells[4] == "100.0" and cells[7] != "-", cells   # 회복률 J&F, 시간(초)
    # 공식 라벨: 데이터셋별 표 + 합친 표
    for label in ("OCC 가림 (영상", "FM 빠른 움직임 (영상", "변형:break (영상", "상태 변화:melt (영상",
                  "상태:solid→liquid (영상"):
        assert label in extra_md, label
    merged = extra_md.split("## 여러 데이터셋 합친 라벨")[1]
    assert "모양·상태 변화 (영상 4)" in merged, merged   # lval_00(DEF) + VOST 2 + M3VOS 1
    print("  OK 표: main.md / extra.md")


def check_roundtrip() -> None:
    video = load_dataset("lvos_v2_valid")[0]
    runner = fake_sam2.FakeRunner("base_plus")
    r = sam2_check.roundtrip(runner, video, obj_id=1, start=0, cut=10, last=25)
    assert r["identical_frames"] == r["frames"] > 0, r
    print(f"  OK 꺼냈다 넣기: {r['identical_frames']}/{r['frames']} 프레임 동일")


def test_end_to_end():
    tmp = Path(tempfile.mkdtemp(prefix="benchmark_test_"))
    try:
        setup(tmp)
        run_script("1_make_video_list.py")
        for dataset in DATASETS:
            if dataset == "pumavos":    # 본 모델 먼저, 비교군은 나중에 따로
                run_script("2_evaluate.py", "--dataset", dataset, "--methods", "model")
                run_script("2_evaluate.py", "--dataset", dataset, "--methods", "baselines")
            else:
                run_script("2_evaluate.py", "--dataset", dataset)

        before = len(records.read_rows(records.records_path("vost_val")))
        run_script("2_evaluate.py", "--dataset", "vost_val")
        assert len(records.read_rows(records.records_path("vost_val"))) == before, "이어하기 실패"

        print("\n확인")
        for dataset in DATASETS:
            check_rows(dataset)
        check_void()
        run_script("3_make_tables.py")
        check_tables()
        check_roundtrip()
        print("\n모두 통과")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_end_to_end()
