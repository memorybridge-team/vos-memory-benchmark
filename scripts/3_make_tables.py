"""[3] SQLite → 반복 통계·영상별 점수·전환 전후 곡선. 낮은 점수도 모두 포함.

python scripts/3_make_tables.py
python scripts/3_make_tables.py --runs 1 --seed 0
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings
from evaluation import records, store
from evaluation.scoring import recovery, restoration
from evaluation.tables import extra_tables, main_tables, summaries, temporal, recovery_curves


def load_raw_rows(seed=None, runs=None, experiment_id=None):
    selected = store.select_experiment(experiment_id)
    return store.read_results(seed=seed, runs=runs, visible_only=True,
                              experiment_id=selected['experiment_id'])


def load_rows(seed=None, runs=None, native_statistic='median', experiment_id=None):
    selected = store.select_experiment(experiment_id)
    raw = load_raw_rows(seed, runs, selected['experiment_id'])
    if not raw:
        raise ValueError('선택한 실험·seed·회차에 집계할 결과가 없습니다.')
    with store.analysis_settings(selected):
        return restoration.recompute(recovery.recompute(raw, range(1, (runs or settings.EVALUATION_RUNS) + 1), native_statistic))


def load_object_labels(rows):
    """추론 시점에 고정한 라벨을 사용한다. 서로 다른 회차의 라벨 충돌은 오류다."""
    labels = {}
    for row in rows:
        key = (row['dataset'], row['video'], row['object'])
        value = row['extra_labels']
        if key in labels and labels[key] != value:
            raise ValueError(f'동일 객체의 난이도 라벨이 다릅니다: {key}')
        labels[key] = value
    return labels


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'저장: {path}')


def show_experiments():
    experiments = store.list_experiments()
    print(f'DB: {store.database_path()}, 실험 {len(experiments)}개')
    for entry in experiments:
        config = entry['configuration']
        print(f"{entry['experiment_id']}\n"
              f"  결과 {entry['result_count']}개, Native {entry['native_count']}개, "
              f"데이터셋={entry['datasets']}, seed={entry['seeds']}, 회차={entry['run_ids']}\n"
              f"  평가 버전={config.get('evaluation_revision')}, 실행 버전={config.get('runtime_revision')}, "
              f"방법 revision={entry['baseline_revisions']}")


def main():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--experiment-id', help='DB에 저장된 실험 ID. 하나면 자동 선택')
    selection.add_argument('--list-experiments', action='store_true', help='DB 실험 목록만 표시')
    parser.add_argument('--seed', type=int, default=settings.EVALUATION_SEED)
    parser.add_argument('--runs', type=int, choices=(1, 2, 3), default=settings.EVALUATION_RUNS,
                        help='집계할 반복 횟수: 평가 실행의 --runs와 같은 값 사용 (기본 %(default)s)')
    parser.add_argument('--native-statistic', choices=('median', 'mean'), default='median')
    parser.add_argument('--skip-recovery-plots', action='store_true',
                        help='공통 구간 회복률 CSV/메타데이터만 저장하고 PNG/PDF 생성은 생략')
    args = parser.parse_args()
    if args.list_experiments:
        show_experiments()
        return
    try:
        selected = store.select_experiment(args.experiment_id)
        raw = load_raw_rows(args.seed, args.runs, selected['experiment_id'])
        if not raw:
            raise ValueError('선택한 실험·seed·회차에 결과가 없습니다. 기존 표는 변경하지 않습니다.')
        with store.analysis_settings(selected):
            make_tables(selected, args, raw)
    except ValueError as exc:
        parser.error(str(exc))


def make_tables(selected, args, raw):
    identity = selected['experiment_id']
    print(f'집계 실험: {identity}, 회차: 1~{args.runs}, Native 기준: {args.native_statistic}')
    print(f"방법 revision: {selected['baseline_revisions']} (DB에 저장된 방법별 최신 버전)")
    rows = restoration.recompute(recovery.recompute(raw, range(1, args.runs + 1), args.native_statistic))
    pending = sum(r['pre_pending_native_n_frames'] + r['post_pending_native_n_frames'] for r in rows)
    if pending:
        print(f'주의: 전환 전후 Native 기준이 미완료인 프레임 결과 {pending}개는 구간 회복률의 양쪽 평균에서 제외됩니다.')
    groups = defaultdict(list)
    for row in rows:
        groups[row['dataset']].append(row)
    groups = dict(sorted(groups.items()))
    labels = load_object_labels(rows)
    summary, per_run, per_video = summaries.build(rows)
    reference_rows = recovery.reference_rows(raw, range(1, args.runs + 1), args.native_statistic)
    video_lists = store.analysis_video_lists(identity, seed=args.seed, runs=args.runs)
    curves, windows = recovery_curves.build(rows, video_lists, range(1, args.runs + 1))
    analysis_id = store.save_analysis(rows, range(1, args.runs + 1), args.native_statistic, args.seed,
                                     experiment_id=identity)
    print(f'SQLite 집계: {analysis_id}')
    out_dir = Path(settings.OUTPUT_ROOT) / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    note = f'experiment_id={identity}, seed={args.seed}, 대상 회차=1~{args.runs}, Native 기준={args.native_statistic}, 구간 회복률=평균 점수의 비율. 회차 수와 표본 수를 확인하세요.\n'
    for name, lines in {'main.md': main_tables.build(groups),
                        'extra.md': extra_tables.build(groups, labels)}.items():
        (out_dir / name).write_text(note + '\n'.join(lines) + '\n', encoding='utf-8')
        print(f'저장: {out_dir / name}')
    (out_dir / 'experiment.json').write_text(json.dumps(dict(selected, analysis_id=analysis_id), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    # 기본 결과 폴더는 마지막 집계를 보여준다. 정의별 JSONL은 원본과 별도로 보존한다.
    snapshot = Path(settings.OUTPUT_ROOT) / 'analysis' / identity / f'recovery.seed{args.seed}.runs{args.runs}.{args.native_statistic}.ratio_of_means.jsonl'
    records.write_rows(snapshot, rows)
    print(f'저장: {snapshot}')
    write_csv(out_dir / 'native_reference.csv', recovery.REFERENCE_FIELDS,
              reference_rows)
    write_csv(out_dir / 'summary.csv', summaries.SUMMARY_FIELDS, summary)
    write_csv(out_dir / 'per_run.csv', summaries.PER_RUN_FIELDS, per_run)
    write_csv(out_dir / 'per_video.csv', summaries.PER_VIDEO_FIELDS, per_video)
    write_csv(out_dir / 'temporal.csv', temporal.CSV_FIELDS, temporal.build(rows))
    write_csv(out_dir / 'recovery_common_window.csv', recovery_curves.CSV_FIELDS, curves)
    curve_dir = Path(settings.OUTPUT_ROOT) / 'figures' / identity / f'recovery_common.seed{args.seed}.runs{args.runs}.{args.native_statistic}'
    curve_dir.mkdir(parents=True, exist_ok=True)
    (curve_dir / 'windows.json').write_text(json.dumps(windows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for window in windows:
        print(f"공통 구간: {window['dataset']} 전환 {window['switch_name']}%, "
              f"[-{window['window_n']}, +{window['window_n']}], "
              f"영상 {window['cohort_video_count']}/{window['planned_video_count']}, "
              f"객체 {window['cohort_object_count']}/{window['planned_object_count']}")
    if not args.skip_recovery_plots:
        for path in recovery_curves.plot(curves, windows, curve_dir):
            print(f'저장: {path}')


if __name__ == '__main__':
    main()
