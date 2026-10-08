"""[3] JSONL → 반복 통계·영상별 점수·전환 전후 곡선. 낮은 점수도 모두 포함.

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
from evaluation import records
from evaluation.scoring import recovery, restoration
from evaluation.tables import extra_tables, main_tables, summaries, temporal, recovery_curves
from evaluation.data import load_video_list, video_list_path


def load_raw_rows(seed=None, runs=None):
    rows = []
    for path in sorted((Path(settings.OUTPUT_ROOT) / 'records').glob('*.jsonl')):
        rows += records.read_rows(path)
    rows = records.unique_rows(records.current_rows(rows))
    return [r for r in rows if (seed is None or r['seed'] == seed)
            and (runs is None or r['run_id'] <= runs)]


def load_rows(seed=None, runs=None, native_statistic='median'):
    raw = load_raw_rows(seed, runs)
    return restoration.recompute(recovery.recompute(raw, range(1, (runs or settings.EVALUATION_RUNS) + 1), native_statistic))


def load_object_labels():
    labels = {}
    for path in sorted((Path(settings.OUTPUT_ROOT) / 'lists').glob('*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        for entry in data['videos']:
            for obj in entry['objects']:
                labels[(data['dataset'], entry['video'], obj['object'])] = obj.get('extra_labels', [])
    return labels


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'저장: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=settings.EVALUATION_SEED)
    parser.add_argument('--runs', type=int, default=settings.EVALUATION_RUNS)
    parser.add_argument('--native-statistic', choices=('median', 'mean'), default='median')
    parser.add_argument('--skip-recovery-plots', action='store_true',
                        help='공통 구간 회복률 CSV/메타데이터만 저장하고 PNG/PDF 생성은 생략')
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs는 1 이상이어야 합니다.')
    raw = load_raw_rows(args.seed, args.runs)
    rows = restoration.recompute(recovery.recompute(raw, range(1, args.runs + 1), args.native_statistic))
    pending = sum(r['pre_pending_native_n_frames'] + r['post_pending_native_n_frames'] for r in rows)
    if pending:
        print(f'주의: 전환 전후 Native 기준이 미완료인 프레임 결과 {pending}개는 구간 회복률의 양쪽 평균에서 제외됩니다.')
    groups = defaultdict(list)
    for row in rows:
        groups[row['dataset']].append(row)
    groups = dict(sorted(groups.items()))
    out_dir = Path(settings.OUTPUT_ROOT) / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    note = f'seed={args.seed}, 대상 회차=1~{args.runs}, Native 기준={args.native_statistic}, 구간 회복률=평균 점수의 비율. 회차 수와 표본 수를 확인하세요.\n'
    for name, lines in {'main.md': main_tables.build(groups),
                        'extra.md': extra_tables.build(groups, load_object_labels())}.items():
        (out_dir / name).write_text(note + '\n'.join(lines) + '\n', encoding='utf-8')
        print(f'저장: {out_dir / name}')
    # 기본 결과 폴더는 마지막 집계를 보여준다. 정의별 JSONL은 원본과 별도로 보존한다.
    snapshot = Path(settings.OUTPUT_ROOT) / 'analysis' / f'recovery.seed{args.seed}.runs{args.runs}.{args.native_statistic}.ratio_of_means.jsonl'
    records.write_rows(snapshot, rows)
    print(f'저장: {snapshot}')
    write_csv(out_dir / 'native_reference.csv', recovery.REFERENCE_FIELDS,
              recovery.reference_rows(raw, range(1, args.runs + 1), args.native_statistic))
    summary, per_run, per_video = summaries.build(rows)
    write_csv(out_dir / 'summary.csv', summaries.SUMMARY_FIELDS, summary)
    write_csv(out_dir / 'per_run.csv', summaries.PER_RUN_FIELDS, per_run)
    write_csv(out_dir / 'per_video.csv', summaries.PER_VIDEO_FIELDS, per_video)
    write_csv(out_dir / 'temporal.csv', temporal.CSV_FIELDS, temporal.build(rows))
    video_lists = {dataset: load_video_list(dataset) for dataset in groups if video_list_path(dataset).exists()}
    curves, windows = recovery_curves.build(rows, video_lists, range(1, args.runs + 1))
    write_csv(out_dir / 'recovery_common_window.csv', recovery_curves.CSV_FIELDS, curves)
    curve_dir = Path(settings.OUTPUT_ROOT) / 'figures' / f'recovery_common.seed{args.seed}.runs{args.runs}.{args.native_statistic}'
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
