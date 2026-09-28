"""가짜 모델·가짜 데이터로 전체 흐름 확인.

    python tests/test_end_to_end.py

확인하는 것:
  - 1 → 2 → 3 → 5 가 끝까지 돈다 (LVOS v2 train dev, LVOS v2 valid, VOST, M3VOS, PUMaVOS)
  - 모든 비교군(주 10 + 추가 3)이 모든 전환 시점에 결과 줄을 남긴다
  - 회복률 두 기준(영상 전체 / 전환 뒤)이 다 나온다, Full Replay 는 100
  - 주 표와 추가 표가 따로 나온다, 전환 B 표가 따로 나온다
  - 다시 실행하면 이미 끝난 객체는 건너뛴다
  - MOSEv2 valid: 제출 zip 28개 → 가짜 서버 점수 → 4 → 표에 영상 전체 열만
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
from benchmark import records  # noqa: E402
from benchmark.data import load_dataset, load_video_list  # noqa: E402
from benchmark.methods import EXTRA, MAIN, METHODS  # noqa: E402
from benchmark.model import sam2_check, sam2_runner  # noqa: E402
from benchmark.scoring.retention import retention_by_video  # noqa: E402

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
    settings.BOOTSTRAP_SAMPLES = 200
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
                got = {r["method"] for r in rows if r["video"] == entry["video"]
                       and r["object"] == obj["object"] and r["switch_name"] == sw["name"]}
                assert got == {m.name for m in METHODS}, (dataset, entry["video"], sw, got)
    for r in rows:
        assert r["jf_whole"] is not None and r["n_frames_whole"] > 0, r
        assert r["reseen_frames"] >= 0 and r["seconds_to_first_frame"] >= 0
        if r["method"] != "reset":
            assert r["jf_whole"] > 0.3, r
    replay = [r for r in rows if r["method"] == "full_replay"]
    for key in ("jf_whole", "jf_post"):
        ratios, _ = retention_by_video(replay, replay, key)
        assert ratios and all(abs(v - 100) < 1e-9 for v in ratios.values())
        direct, _ = retention_by_video([r for r in rows if r["method"] == "direct_state_copy"],
                                       replay, key)
        assert direct, f"{dataset}: 회복률 {key} 없음"
    assert any(r["switch_set"] == "B" for r in rows), f"{dataset}: 전환 B 없음"
    assert any(r["extra_stratum_occlusion"] for r in rows), f"{dataset}: 가림 분류 없음"
    small = [r["jf_post"] for r in rows if r["method"] == "source_only" and r["jf_post"]]
    base = [r["jf_post"] for r in rows if r["method"] == "full_replay" and r["jf_post"]]
    assert sum(base) / len(base) > sum(small) / len(small), "가짜 Base+ 가 Small 보다 좋아야 함"
    print(f"  OK {dataset}: 줄 {len(rows)}개")


def check_mosev2() -> None:
    zips = sorted((Path(settings.OUTPUT_ROOT) / "mosev2" / "submissions").glob("*.zip"))
    assert len(zips) == 28, f"제출 zip {len(zips)}개 (28개여야 함)"
    csv_path = Path(settings.OUTPUT_ROOT) / "fake_server.csv"
    lines = ["submission,video,jf,j,f"]
    for i, z in enumerate(zips):
        for video in ("mval_00", "mval_01"):
            score = 70 if z.stem.startswith("full_replay") else 50 + i % 10
            lines.append(f"{z.stem},{video},{score},{score - 2},{score + 2}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_script("4_mosev2_scores.py", "--csv", str(csv_path))
    server = records.read_rows(Path(settings.OUTPUT_ROOT) / "mosev2" / "server_rows.jsonl")
    assert {r["method"] for r in server} == {m.name for m in MAIN}
    print(f"  OK mosev2_valid: 제출 {len(zips)}개, 서버 줄 {len(server)}개")


def check_tables() -> None:
    tables = Path(settings.OUTPUT_ROOT) / "tables"
    main_md = (tables / "main.md").read_text(encoding="utf-8")
    extra_md = (tables / "extra.md").read_text(encoding="utf-8")
    b_md = (tables / "switch_b.md").read_text(encoding="utf-8")
    for m in MAIN:
        assert f"| {m.label} |" in main_md, m.label
    for m in EXTRA:
        assert m.label not in main_md and f"| {m.label} |" in extra_md, m.label
    assert "회복률 전체" in main_md and "회복률 전환 뒤" in main_md
    assert "## mosev2_valid" in main_md and "## lvos_v2_train (dev)" in main_md
    assert "ID 뒤바뀜" in extra_md and "가림" in extra_md
    assert "## lvos_v2_valid" in b_md and "## mosev2_valid" not in b_md
    mose = main_md.split("## mosev2_valid")[1].split("##")[0]
    replay_line = next(line for line in mose.splitlines() if line.startswith("| Full Replay |"))
    cells = [c.strip() for c in replay_line.strip("|").split("|")]
    assert cells[3].startswith("100.0") and cells[4] == "-", cells   # 전체 기준만, 전환 뒤는 없음
    print("  OK 표: main.md / extra.md / switch_b.md")


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
