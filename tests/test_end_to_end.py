"""가짜 모델·가짜 데이터로 전체 흐름 확인.

    python tests/test_end_to_end.py

확인하는 것:
  - 1 → 2 → 3 → 5 가 끝까지 돈다 (LVOS v2 train dev, LVOS v2 valid, VOST, M3VOS, PUMaVOS)
  - 모든 비교군(주 10 + 추가 2)이 모든 전환 시점에 결과 줄을 남긴다
  - 빼기로 한 열(영상 전체 점수, 다시 본 프레임 수, 옛 추가 지표)이 없다
  - Full Replay: 회복률 100, 속도 배수 1, 격차 회복률 100 / Source-only 격차 회복률 0
  - 주 표(main.md)와 추가 표(extra.md, drift PNG)가 나온다
  - 다시 실행하면 이미 끝난 객체는 건너뛴다
  - MOSEv2 valid: 제출 zip 28개 → 가짜 서버 점수 → 4 → 주 표에 서버 점수, 추가 표에 출력 일치도
  - 기억 꺼냈다 넣기 결과가 끊지 않은 결과와 같다 (sam2_check.roundtrip)
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
from evaluation import records  # noqa: E402
from evaluation.data import load_dataset, load_video_list  # noqa: E402
from baseline import EXTRA, MAIN, BASELINES  # noqa: E402
from model import sam2_check, sam2_runner  # noqa: E402
from evaluation.scoring.main_metrics import gap_retention, retention, speedup  # noqa: E402

import fake_data  # noqa: E402
import fake_sam2  # noqa: E402

LOCAL_DATASETS = ["lvos_v2_valid", "vost_val", "m3vos", "pumavos"]


def run_script(name: str, *args: str) -> None:
    sys.argv = [name, *args]
    runpy.run_path(str(ROOT / "scripts" / name), run_name="__main__")


def setup(tmp: Path) -> None:
    settings.DATA_ROOT = str(tmp / "data")
    settings.OUTPUT_ROOT = str(tmp / "outputs")
    settings.DEV_FRACTION = 0.5
    fake_data.make_all(Path(settings.DATA_ROOT), settings.DATA_FOLDERS)
    sam2_runner.load_runner = fake_sam2.FakeRunner


def check_rows(dataset: str, part=None) -> None:
    rows = records.read_rows(records.records_path(dataset))
    assert rows, f"{dataset}: 결과 줄 없음"
    video_list = load_video_list(dataset)
    evaluated = {r["video"] for r in rows}
    for entry in video_list["videos"]:
        if entry["video"] not in evaluated:
            continue
        assert part is None or entry["part"] == part, entry
        for obj in entry["objects"]:
            for sw in obj["switches"]:
                got = {r["baseline"] for r in rows if r["video"] == entry["video"]
                       and r["object"] == obj["object"] and r["switch_name"] == sw["name"]}
                assert got == {m.name for m in BASELINES}, (dataset, entry["video"], sw, got)
    removed = ("jf_whole", "j_whole", "jf_post", "reseen_frames", "extra_jf_at_n", "extra_switch_shock",
               "extra_recovery_frames", "extra_absent_false_alarm", "extra_stratum_occlusion")
    for r in rows:
        assert not any(k in r for k in removed), r
        assert r["jf"] is not None and r["n_frames"] > 0, r
        assert len(r["extra_drift_jf"]) == len(settings.EXTRA_DRIFT_BINS), r
        assert r["extra_agreement"] is not None, r
        if r["baseline"] in ("reset", "source_only"):
            assert r["switch_seconds"] is None, r
        else:
            assert r["switch_seconds"] >= 0, r
        if r["role"] == "main":     # [추가] recent_k_only 는 객체가 안 보일 때 시작하면 0점이 맞음
            assert r["jf"] > 0.3, r
    replay = [r for r in rows if r["baseline"] == "full_replay"]
    source = [r for r in rows if r["baseline"] == "source_only"]
    direct = [r for r in rows if r["baseline"] == "direct_state_copy"]
    for key in ("jf", "j"):
        assert abs(retention(replay, replay, key) - 100) < 1e-9
        assert retention(direct, replay, key) is not None, f"{dataset}: 회복률 {key} 없음"
        assert abs(gap_retention(replay, source, replay, key) - 100) < 1e-9
        assert abs(gap_retention(source, source, replay, key)) < 1e-9
    assert abs(speedup(replay, replay) - 1) < 1e-9
    assert all(r["extra_agreement"] == 1.0 for r in replay)
    small = [r["jf"] for r in source]
    base = [r["jf"] for r in replay]
    assert sum(base) / len(base) > sum(small) / len(small), "가짜 Base+ 가 Small 보다 좋아야 함"
    print(f"  OK {dataset}: 줄 {len(rows)}개")


def check_mosev2() -> None:
    zips = sorted((Path(settings.OUTPUT_ROOT) / "mosev2" / "submissions").glob("*.zip"))
    assert len(zips) == 28, f"제출 zip {len(zips)}개 (28개여야 함)"
    csv_path = Path(settings.OUTPUT_ROOT) / "fake_server.csv"
    lines = ["submission,video,jf,j"]
    for i, z in enumerate(zips):
        for video in ("mval_00", "mval_01"):
            score = 70 if z.stem.startswith("full_replay") else 50 + i % 10
            lines.append(f"{z.stem},{video},{score},{score - 2}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_script("4_mosev2_scores.py", "--csv", str(csv_path))
    server = records.read_rows(Path(settings.OUTPUT_ROOT) / "mosev2" / "server_rows.jsonl")
    assert {r["baseline"] for r in server} == {m.name for m in MAIN}
    print(f"  OK mosev2_valid: 제출 {len(zips)}개, 서버 줄 {len(server)}개")


def check_tables() -> None:
    tables = Path(settings.OUTPUT_ROOT) / "tables"
    main_md = (tables / "main.md").read_text(encoding="utf-8")
    extra_md = (tables / "extra.md").read_text(encoding="utf-8")
    assert not (tables / "switch_b.md").exists() and "전환 B" not in extra_md
    for m in MAIN:
        assert f"| {m.label} |" in main_md, m.label
    for m in EXTRA:
        assert m.label not in main_md and f"| {m.label} |" in extra_md, m.label
    for column in ("회복률 J&F", "격차 회복률 J", "전환 지연(초)", "속도 배수"):
        assert column in main_md, column
    assert "## mosev2_valid" in main_md and "## lvos_v2_train (dev)" in main_md
    vost = main_md.split("## vost_val")[1].split("##")[0]
    assert "주 지표: J\n" in vost
    for section in ("비용 세부", "출력 일치도", "진단 비교군", "실패 분석", "drift", "공식 라벨별",
                    "입력 길이별"):
        assert f"### {section}" in extra_md, section
    assert (tables / "drift_vost_val.png").exists()
    # 공식 라벨: 데이터셋별 표 + 합친 표
    for label in ("OCC 가림 (영상", "FM 빠른 움직임 (영상", "변형:break (영상", "상태 변화:melt (영상",
                  "상태:solid→liquid (영상"):
        assert label in extra_md, label
    merged = extra_md.split("## 여러 데이터셋 합친 라벨")[1]
    assert "모양·상태 변화 (영상 4)" in merged, merged   # lval_00(DEF) + VOST 2 + M3VOS 1
    # MOSEv2: 주 표는 서버 점수, 추가 표는 출력 일치도만 (실패 분석 등은 없음)
    mose = main_md.split("## mosev2_valid")[1].split("##")[0]
    replay_line = next(line for line in mose.splitlines() if line.startswith("| Full Replay |"))
    cells = [c.strip() for c in replay_line.strip("|").split("|")]
    assert cells[4] == "100.0" and cells[8] == "1.0", cells   # 회복률 J&F, 속도 배수
    mose_extra = extra_md.split("## mosev2_valid")[1].split("\n## ")[0]
    assert "### 출력 일치도" in mose_extra and "### 실패 분석" not in mose_extra
    print("  OK 표: main.md / extra.md / drift PNG")


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
        run_script("2_fit_moment_stats.py")
        run_script("3_evaluate.py", "--dataset", "lvos_v2_train", "--part", "dev", "--max-videos", "3")
        for dataset in LOCAL_DATASETS:
            run_script("3_evaluate.py", "--dataset", dataset)
        run_script("3_evaluate.py", "--dataset", "mosev2_valid")

        before = len(records.read_rows(records.records_path("vost_val")))
        run_script("3_evaluate.py", "--dataset", "vost_val")
        assert len(records.read_rows(records.records_path("vost_val"))) == before, "이어하기 실패"

        print("\n확인")
        check_rows("lvos_v2_train", part="dev")
        for dataset in LOCAL_DATASETS:
            check_rows(dataset)
        check_mosev2()
        run_script("5_make_tables.py")
        check_tables()
        check_roundtrip()
        print("\n모두 통과")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_end_to_end()
