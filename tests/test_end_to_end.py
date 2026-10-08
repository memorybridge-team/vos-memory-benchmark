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
from evaluation import native, records
from evaluation.data import DATASETS, load_dataset, load_video_list, video_list_path
from evaluation.methods import METHODS, to_run
from evaluation.scoring.recovery import ratio
from evaluation.scoring import recovery
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


def keep_only_model(dataset):
    for path in sorted(records.records_path(dataset).parent.glob(f'{dataset}.run*.jsonl')):
        rows = [r for r in records.read_rows(path) if r['baseline'] in ('translator', 'source_only', 'full_replay')]
        path.unlink()
        records.append_rows(path.with_name(path.stem + '.model.jsonl'), rows)

def check_rows(dataset: str) -> None:
    raw = dataset_rows(dataset)
    assert raw and len(records.unique_rows(raw)) == len(raw)
    assert all(r['recovery_reference'] == 'pending' and r['recovery_j'] is None and r['recovery_jf'] is None for r in raw)
    before = json.dumps(raw, sort_keys=True)
    rows = recovery.recompute(raw)
    assert json.dumps(raw, sort_keys=True) == before
    assert all(r['recovery_reference'] == 'native_median' and r['pending_native_n_frames'] == r['pre_pending_native_n_frames'] == 0 for r in rows)
    for entry in load_video_list(dataset)["videos"]:
        for obj in entry["objects"]:
            assert [sw["name"] for sw in obj["switches"]] == ["50", "75"]
            for run_id in range(1, settings.EVALUATION_RUNS + 1):
                for sw in obj["switches"]:
                    got = {r["baseline"]: r for r in rows if r["video"] == entry["video"]
                           and r["object"] == obj["object"] and r["switch_name"] == sw["name"]
                           and r["run_id"] == run_id}
                    assert set(got) == {m.name for m in METHODS}
                    assert got["translator"]["jf"] == got["direct_state_copy"]["jf"]
                    assert len({r['native_reference_id'] for r in got.values()}) == 1
                    small_rows = [r for name, r in got.items() if name != 'full_replay']
                    assert len({r['pre_recovery_j'] for r in small_rows}) == 1
                    assert len({r['pre_recovery_jf'] for r in small_rows}) == 1
    for row in rows:
        assert not REMOVED_COLUMNS.intersection(row)
        assert row["evaluation_revision"] == settings.EVALUATION_REVISION
        assert row["jf"] is not None and row["n_frames"] > 0
        assert 0 <= row["failure_rate"] <= 1
        assert row["n_frames"] == len(row["frame_scores"])
        assert abs(row["failure_rate"] - sum(p["j"] <= 0.5 for p in row["frame_scores"]) / row["n_frames"]) < 1e-12
        assert abs(row["jf"] - sum(p["jf"] for p in row["frame_scores"]) / row["n_frames"]) < 1e-12
        assert row['run_id'] in (1, 2, 3) and row['seed'] == settings.EVALUATION_SEED
        assert row['pre_switch_frame_scores']
        for point in row['pre_switch_frame_scores']:
            assert point['frames_after_switch'] <= 0
            for metric in ('j', 'jf'):
                assert point[f'recovery_{metric}'] == ratio(point[metric], point[f'native_{metric}'])
        for point in row["frame_scores"]:
            assert point["frames_after_switch"] == point["frame"] - row["switch_frame"] > 0
            for metric in ('j', 'jf'):
                assert point[f'recovery_{metric}'] == ratio(point[metric], point[f'native_{metric}'])
        assert row['recovery_statistic'] == 'ratio_of_means'
        for phase, field in (('pre', 'pre_switch_frame_scores'), ('post', 'frame_scores')):
            for metric in ('j', 'jf'):
                pairs = [p for p in row[field] if p[metric] is not None and p[f'native_{metric}'] is not None]
                assert row[f'{phase}_recovery_{metric}_n_frames'] == len(pairs)
                expected = ratio(sum(p[metric] for p in pairs), sum(p[f'native_{metric}'] for p in pairs)) if pairs else None
                actual = row[f'{phase}_recovery_{metric}']
                assert actual is None if expected is None else abs(actual - expected) < 1e-9
        assert row['recovery_j'] == row['post_recovery_j']
        assert row['recovery_jf'] == row['post_recovery_jf']
        if row["baseline"] == "source_only":
            assert row["switch_seconds"] is None and row["switch_gpu_mb"] is None
        else:
            assert row["switch_seconds"] > 0
        assert "switch_gpu_mb" in row
        assert row['restoration_revision'] == 1 and row['restoration_measured_at'] == row['switch_frame']
        assert row['restoration_reference'] == 'native_same_run'
        points = row['restoration_frame_scores']
        if row['baseline'] == 'source_only':
            assert not points and row['restoration_status'] == 'source_only_no_target_memory'
        else:
            assert points and row['restoration_status'] == 'measured'
            assert all(p['frame'] <= row['switch_frame'] and p['frames_before_switch'] >= 0 for p in points)
            assert any(p['maskmem_features']['r2'] is not None for p in points)
            assert all(set(('maskmem_features', 'obj_ptr')).issubset(p) for p in points)
        # 집계를 미리 결정하는 대표 scalar 지표를 추가하지 않는다.
        assert 'r2_maskmem_features' not in row and 'r2_obj_ptr' not in row
        if row['baseline'] == 'full_replay':
            assert all(p[field]['r2'] == 1 for p in points for field in ('maskmem_features', 'obj_ptr'))

    replay = [r for r in rows if r["baseline"] == "full_replay"]
    for row in replay:
        assert row['recovery_j'] == row['recovery_jf'] == 100
    refs = native.load_references(dataset)
    objects = sum(len(v['objects']) for v in load_video_list(dataset)['videos'])
    assert len(refs) == objects * 3
    for ref in refs.values():
        memories = native.load_memories(ref)
        assert set(memories) == {sw['frame'] for sw in ref['switches']}
        assert all(set(entry) == {'maskmem_features', 'obj_ptr', 'is_cond'}
                   for snapshot in memories.values() for entry in snapshot.values())
    assert len(rows) == objects * 3 * 2 * len(METHODS)
    assert all(len({r['native_reference_id'] for r in rows if r['run_id'] == ref['run_id']
                    and r['video'] == ref['video'] and r['object'] == ref['object']}) == 1
               for ref in refs.values())
    for row in replay:
        if row["switch_name"] == "50":
            later = next(r for r in replay if r["video"] == row["video"] and r["object"] == row["object"] and r["switch_name"] == "75" and r["run_id"] == row["run_id"])
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
    for column in ("전환 전 회복률 J", "전환 전 회복률 J&F", "전환 후 회복률 J", "전환 후 회복률 J&F", "전환시간(초)", "전환 GPU 메모리(MB)", "실패 비율(%)", "복원율 R² (spatial)", "복원율 R² (pointer)", "R² 유효 객체"):
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
    assert {p['phase'] for p in points} == {'pre', 'post'}
    assert any(int(p['frames_after_switch']) < 0 for p in points)
    assert all(0 <= float(p['jf']) <= 100 and int(p['jf_run_count']) == 3 for p in points)
    assert all(p['recovery_j'] != '' and p['recovery_jf'] != '' for p in points if p['phase'] == 'pre')
    native_points = [p for p in points if p['baseline'] == 'full_replay' and p['phase'] == 'post']
    assert all(float(p['recovery_j']) == float(p['recovery_jf']) == 100 for p in native_points)
    for name in ('summary.csv', 'per_run.csv', 'per_video.csv'):
        with (tables / name).open(encoding='utf-8-sig', newline='') as stream:
            result = list(csv.DictReader(stream))
        assert result
        for metric in ('pre_recovery_j', 'pre_recovery_jf', 'post_recovery_j', 'post_recovery_jf'):
            assert all(r[metric + ('_mean' if name == 'summary.csv' else '')] != '' for r in result)
        if name == 'summary.csv':
            assert all(int(r['j_run_count']) == 3 and float(r['j_std']) == 0 for r in result)
        else:
            assert {int(r['run_id']) for r in result} == {1, 2, 3}
    assert '±' in main and '회차 수' in main and 'Native 기준=median' in main
    with (tables / 'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
        r2_rows = list(csv.DictReader(stream))
    for row in r2_rows:
        if row['baseline'] == 'source_only':
            assert row['r2_maskmem_features_mean'] == '' and row['r2_obj_ptr_mean'] == ''
        else:
            assert row['r2_maskmem_features_mean'] != '' and row['r2_obj_ptr_mean'] != ''
        if row['baseline'] == 'full_replay':
            assert float(row['r2_maskmem_features_mean']) == float(row['r2_obj_ptr_mean']) == 1
        assert 'r2_obj_ptr_object_count' in row and 'r2_maskmem_features_object_count' in row
    with (tables / 'native_reference.csv').open(encoding='utf-8-sig', newline='') as stream:
        reference = list(csv.DictReader(stream))
    assert reference and all(r['native_statistic'] == 'median' and int(r['native_reference_count']) == 3 for r in reference)
    assert all(r['native_reference_ready'] == 'True' for r in reference)
    snapshots = list((Path(settings.OUTPUT_ROOT) / 'analysis').glob('*.median.ratio_of_means.jsonl'))
    assert snapshots and all(r['recovery_reference'] == 'native_median' for r in records.read_rows(snapshots[0]))
    with (tables / 'recovery_common_window.csv').open(encoding='utf-8-sig', newline='') as stream:
        curves = list(csv.DictReader(stream))
    assert curves and {p['switch_name'] for p in curves} == {'50', '75'}
    assert {p['baseline'] for p in curves} == {m.name for m in METHODS}
    figure_dir = Path(settings.OUTPUT_ROOT) / 'figures' / 'recovery_common.seed0.runs3.median'
    windows = json.loads((figure_dir / 'windows.json').read_text(encoding='utf-8'))
    assert len(windows) == len(DATASETS) * 2
    for window in windows:
        assert window['status'] == 'ok' and window['window_basis'] == 'planned_object_ranges'
        assert window['cohort_object_count'] == window['planned_object_count']
        chosen = [p for p in curves if p['dataset'] == window['dataset'] and p['switch_name'] == window['switch_name']]
        assert {int(p['frames_after_switch']) for p in chosen} == set(range(-window['window_n'], window['window_n'] + 1))
        assert all(int(p['cohort_object_count']) == window['cohort_object_count'] for p in chosen)
        for extension in ('png', 'pdf'):
            assert (figure_dir / f"{window['dataset']}.switch{window['switch_name']}.{extension}").stat().st_size > 1000
    print("  OK 결과표: 전환별 요약, 난이도 성능, temporal.csv, 공통 구간 회복률 CSV/PNG/PDF")


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
        dict(source, evaluation_revision=2),
        {k: v for k, v in source.items() if k != 'run_id'},
    ]
    assert records.current_rows(incompatible) == []
    path = records.records_path(dataset).with_name(f"{dataset}.legacy.jsonl")
    records.append_rows(path, incompatible)
    assert records.row_key(incompatible[1]) not in records.done_keys(dataset)
    loaded = runpy.run_path(str(ROOT / "scripts/3_make_tables.py"))["load_rows"]()
    assert all(r["video"] != "legacy_only" and r["switch_name"] != "25" for r in loaded)
    assert all(r["baseline"] not in ("reset", "original_replay_16") for r in loaded)
    print("  OK 이전 평가 기준과 제거된 비교군 결과 제외")


def check_partial_resume(dataset):
    path = records.records_path(dataset).with_name(f'{dataset}.run2.jsonl')
    rows = records.read_rows(path)
    missing = next(r for r in rows if r['baseline'] == 'translator' and r['switch_name'] == '75')
    path.unlink()
    records.append_rows(path, [r for r in rows if r is not missing])
    old_track = fake_sam2.FakeSession.track
    old_export = fake_sam2.FakeSession.export_memory
    tracked, exported = [], []

    def forbid_native(self, first, last):
        if self.runner.name == 'base_plus' and self.cond and first == min(self.cond):
            raise AssertionError('이어하기에서 Native를 다시 실행함')
        tracked.append((self.runner.name, first, last))
        yield from old_track(self, first, last)

    def count_export(self):
        if self.runner.name == 'small':
            exported.append(max(list(self.cond) + list(self.non_cond)))
        return old_export(self)

    fake_sam2.FakeSession.track = forbid_native
    fake_sam2.FakeSession.export_memory = count_export
    try:
        run_script('2_evaluate.py', '--dataset', dataset)
    finally:
        fake_sam2.FakeSession.track = old_track
        fake_sam2.FakeSession.export_memory = old_export
    completed = records.read_rows(path)
    assert len(completed) == len(rows)
    assert len(records.unique_rows(completed)) == len(completed)
    restored = next(r for r in completed if records.row_key(r) == records.row_key(missing))
    assert restored['native_reference_id'] == missing['native_reference_id']
    assert restored['frame_scores'] == missing['frame_scores']
    assert restored['recovery_j'] == missing['recovery_j']
    assert restored['restoration_frame_scores'] == missing['restoration_frame_scores']
    assert restored['pre_switch_frame_scores'] == missing['pre_switch_frame_scores']
    s = missing['switch_frame']
    assert tracked == [('small', missing['start'], s), ('base_plus', s + 1, missing['end'])], tracked
    assert exported == [s], exported
    print('  OK 회차별 일부 전환 이어하기, Native 기준 재실행 없이 유지')

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
        raw_before = {d: dataset_rows(d) for d in DATASETS}
        run_script('3_make_tables.py', '--native-statistic', 'mean')
        assert (Path(settings.OUTPUT_ROOT) / 'analysis' / 'recovery.seed0.runs3.mean.ratio_of_means.jsonl').exists()
        run_script("3_make_tables.py")
        assert all(dataset_rows(d) == raw_before[d] for d in DATASETS)
        check_tables()
        check_roundtrip()
        print("모두 통과")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_end_to_end()
