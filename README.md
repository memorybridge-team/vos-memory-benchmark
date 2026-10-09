# vos-memory-benchmark

SAM2 Small → Base+ 기억 넘기기 평가. 설정·주요 평가 지표는 `docs/PROTOCOL.md`, 보조 평가 지표는 `docs/EXTRAS.md`.

비교군은 Source-only, Full Replay (Base+ Native), Direct State Copy, Original + Last-Visible, Original-Prompt(s)+Replay-8의 5개다. 본 모델(translator)을 포함해 총 6개 방법을 평가한다.

## 폴더

| 폴더 | 무엇 |
|---|---|
| `model/` | SAM2 켜기·추적·기억 꺼내기/넣기 (SAM2 내부를 건드리는 곳은 여기뿐) |
| `baseline/` | 비교군: 전환 때 Base+ 에 무엇을 넘기나 (기억 상자 포함) |
| `translator/` | 본 모델: Small 기억 칸을 팀 translator 로 바꿔 넘긴다 (`settings.TRANSLATOR_*` 에 전달본 위치) |
| `evaluation/` | 평가: 데이터셋·전환 시점·영상 하나 평가·비용·결과 저장, `scoring/`(지표), `tables/`(표) |
| `scripts/` | 사람이 실행하는 것 (번호 = 순서) |
| `settings.py` | 모든 숫자·경로 |

## 준비

1. SAM2 설치 (github.com/facebookresearch/sam2), `pip install numpy pillow opencv-python`
2. `settings.py` 에 `DATA_ROOT`, `DATA_FOLDERS`, `SAM2_CHECKPOINT_DIR` 확인 (지금 값은 RunPod 서버 위치)
   LVOS에 `val`·`valid`가 모두 있으면 `LVOS_SPLIT_FOLDER`를 지정한다. 속성 JSON이 여러 개면 `LVOS_ATTRIBUTE_FILE`을 지정한다.
3. 데이터가 읽히는지 확인 (영상 1개씩 프레임 수·객체 수·무시 영역 값 출력)

```bash
python -m evaluation.data.lvos_v2
python -m evaluation.data.vost
python -m evaluation.data.m3vos
python -m evaluation.data.pumavos
```

## 실행 순서

```bash
python scripts/0_check_sam2.py
python scripts/1_make_video_list.py
```

`0_check_sam2.py`는 검사 하나라도 실패하거나 이어 추적 프레임이 누락되면 종료 코드 1을 반환한다.
모두 통과하면 `outputs/checks/sam2.json`에 모델·체크포인트·SAM2 코드·장치 설정을 기록한다.
`2_evaluate.py`는 이 기록이 없거나 현재 환경과 다르면 본평가를 시작하지 않는다.
모델/코드/장치/기억 창을 바꾼 뒤에는 같은 환경에서 검사를 다시 실행한다. CPU 검사 기록으로 CUDA 평가를 시작할 수 없다.
데이터 로더는 파일명 불일치, 중복 RGB 프레임, 정답 누락을 오류로 처리한다. 현재 네 데이터셋은 모든 프레임의 정답을 요구한다.

개발 중:

```bash
python scripts/2_evaluate.py --dataset lvos_v2_valid --max-videos 5
python scripts/3_make_tables.py
```

검증·평가:

```bash
python scripts/2_evaluate.py --dataset lvos_v2_valid
python scripts/2_evaluate.py --dataset vost_val
python scripts/2_evaluate.py --dataset m3vos
python scripts/2_evaluate.py --dataset pumavos
python scripts/3_make_tables.py
```

GPU 2개 (데이터셋마다 영상을 반씩 나눠 동시에, 데이터셋 순서는 위와 같음):

```bash
mkdir -p outputs/logs
for ds in lvos_v2_valid vost_val m3vos pumavos; do
  CUDA_VISIBLE_DEVICES=0 python scripts/2_evaluate.py --dataset $ds --shard 0/2 > outputs/logs/${ds}_0.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python scripts/2_evaluate.py --dataset $ds --shard 1/2 > outputs/logs/${ds}_1.log 2>&1 &
  wait
done
python scripts/3_make_tables.py
```

기본 실행은 모든 방법을 3회 평가한다. 회차별 원점수와 Native 기준은 `outputs/benchmark.sqlite`에 저장한다. Python 표준 `sqlite3`를 사용하며 별도 DB 서버는 필요 없다.
실행 시 `--runs 1`, `--runs 2`, `--runs 3`으로 전체 방법의 반복 횟수를 선택한다.
`--runs 2`는 1·2회차를 실행한다. 평가와 집계에 같은 값을 지정해야 Native 기준도 해당 회차 수로 계산된다.

```bash
# 1회 평가·집계
python scripts/2_evaluate.py --dataset vost_val --runs 1
python scripts/3_make_tables.py --runs 1
# 2회 평가·집계
python scripts/2_evaluate.py --dataset vost_val --runs 2
python scripts/3_make_tables.py --runs 2
# 3회 평가·집계 (기본값)
python scripts/2_evaluate.py --dataset vost_val --runs 3
python scripts/3_make_tables.py --runs 3
# 특정 회차만 실행/재개
python scripts/2_evaluate.py --dataset vost_val --run-id 2
```

이어하기 키는 반복 번호·seed·영상·객체·전환 이름·실제 전환 프레임·방법이다.
GPU 수를 바꾸어 재개할 수 있지만 같은 객체를 두 프로세스에서 동시에 맡기지는 않는다.
Native와 Source-only는 회차마다 객체당 한 번씩 실행하고, 같은 결과에서 25%·50%·75% 구간을 잘라 쓴다.
Native 원점수와 전환 비용을 DB에 먼저 저장하고 이어하기에서도 재사용한다. `outputs/benchmark.sqlite`와 `outputs/native_memory`를 함께 보존한다.

프레임별 J·J&F 회복률은 같은 영상·객체·프레임의 방법 점수 / Native 반복 중앙값 × 100이다.
표에는 전환 전·후 각각의 구간 평균 점수 / 같은 프레임의 Native 기준 평균 × 100을 네 열로 표시한다. 프레임별 비율 평균과는 다르다. 전환 전은 객체 최초 등장~s, 전환 후는 s+1~끝이다.
Native를 제외한 방법의 전환 전 점수는 공통 Small 예측이며, Native 행은 자체 예측을 사용한다.
J의 중앙값과 J&F의 중앙값을 별도로 구한다. J&F의 기준은 각 회차의 (J+F)/2를 구한 뒤 그 값들의 중앙값이다.
추론 DB에는 원점수와 `recovery_reference=pending`을 저장한다. 회복률은 집계의 `--runs N`으로 지정한 1~N회 Native가 모인 뒤 3_make_tables.py에서 계산한다.
Native=80,80,0이면 중앙값80을 모든 방법·회차에 공유하며 실패0도 중앙값 표본에서 삭제하지 않는다.
평균 기준으로 바꾸려면 추론 없이 아래 명령만 실행한다.

```bash
python scripts/3_make_tables.py --native-statistic median  # 기본값
python scripts/3_make_tables.py --native-statistic mean
```

집계는 DB 원점수를 변경하지 않는다. 선택 회차·통계 규칙별 `analysis_id`로 집계 결과를 DB에 보존한다. 정의별 회복률 JSONL은 `outputs/analysis/<experiment_id>/recovery.seed0.runs3.median.ratio_of_means.jsonl` 또는 `.mean.ratio_of_means.jsonl`에 별도 저장한다.
`outputs/tables`의 CSV/Markdown은 마지막 집계 결과로 갱신된다.
선택한 실험 설정·방법 revision·분석 ID는 `outputs/tables/experiment.json`에도 저장한다.

집계는 현재 환경에서 ID를 새로 계산하지 않고 DB에 저장된 실험을 선택한다. 실험이 하나면 자동 선택하고, 여러 개면 ID를 지정해야 한다.

```bash
python scripts/3_make_tables.py --list-experiments
python scripts/3_make_tables.py --experiment-id <ID> --runs 3 --seed 0
```

집계에는 체크포인트·데이터·Native 기억 파일·SAM2 검사 보고서가 필요하지 않다. DB만 다른 컴퓨터로 복사해도 저장된 평가 설정을 사용한다. 방법별로 DB에 저장된 최신 `baseline_revision`을 선택하고 이를 출력한다. 결과가 없는 선택은 오류로 종료하며 기존 표를 보존한다. 평가·재개는 계속 현재 환경의 ID와 일치하는 기록만 사용한다.

객체별 전환 전/후 구간 평균의 비율 → 영상 내 객체 평균 → 영상 평균으로 회차 점수를 만든 뒤, 반복 평균·표본 표준편차(ddof=1)·분산을 보고한다.
Native 반복 기준이 0인 프레임도 구간 평균에 포함한다. 구간 Native 평균이 0이면 구간 회복률만 N/A이며 원점수와 실패비율은 유지한다. 누락/미완료 Native 프레임은 분자·분모에서 함께 제외하고 개수를 저장한다. 100%를 넘는 회복률도 그대로 저장한다.
낮은 성능과 빈 예측은 모든 회차에서 포함한다. 실행 오류는 점수 0으로 바꾸지 않고 중단 후 이어한다.
전환 이전은 Small(또는 Native 비교군)의 원점수와 Native 원점수를 저장하고, 전환 이후 전체의 회복률 곡선을 보고한다.
전환 축은 객체 최초 등장~영상 끝의 25%·50%·75%이며, 경과 프레임 번호를 압축하지 않는다.
복원율(R²)은 전환 시점의 기억 프레임별로 `maskmem_features`와 `obj_ptr`를 각각 계산한다.
같은 회차 Native와 비교한 R²·SSE·SST·Native 평균·원소 수·N/A 사유를 DB의 `restoration_frames`·`restoration_fields`에 저장한다.
Native 기억 기준은 `outputs/native_memory/<native_reference_id>.pt`에 보존하며 이어하기에 재사용한다.
DB와 native_memory 폴더를 함께 보존한다. CPU RAM과 디스크 사용량은 늘어난다.
복원율은 준비된 Target 전체 기억 원소의 R²을 재계산한 뒤 객체→영상→회차 순서로 평균한다.
기존 표에 spatial/pointer 두 R² 열을 평균 ± 반복 표준편차로 보고한다. 칸별 R²의 단순 평균은 아니다.
프레임별 충분통계량은 그대로 보존하므로 재추론 없이 집계하며, 복원율은 원래 R² 척도로 표시한다. 자세한 정의는 docs/PROTOCOL.md를 참고한다.

현재 결과 버전은 7, 영상 목록 버전은 4, 실행 시간 버전은 5이다. 25% 전환·객체 전체 사전 제외·SQLite 전체 프레임 저장을 적용한다. 이전 DB 실험은 저장 당시 정의로 별도 집계한다. 결과 버전 5 이하 JSONL은 보존하지만 새 평가·집계·재개에서는 읽지 않는다. 새 정의의 결과는 다시 추론한다.
translator의 방법 revision은 2이며, 출력 dtype을 원래 spatial/pointer dtype으로 되돌린다.
기존 영상 목록은 `1_make_video_list.py`로 다시 만든다. 세 전환 중 하나라도 `s-start < 8`이면 해당 객체를 모든 전환·방법·회차에서 제외한다. 정확히 8이면 포함한다. 유효 객체가 없는 영상은 목록에 넣지 않는다. 등장 0·끝 12이면 객체 전체를 제외하고, 등장 0·끝 32이면 25%=8을 포함한 세 전환 모두 평가한다.
조건별 seed는 반복 번호와 영상·객체·방법으로 결정되어 분할/이어하기 순서에 영향을 받지 않는다.
seed 변경이 추론 결과의 변동을 보장하지는 않는다. 동일한 결과가 반복되면 표준편차는 0이다.
시간과 GPU 메모리 비교에는 같은 종류의 GPU를 사용한다.


SQLite 스키마는 `evaluation/schema.sql`이다. `native_frame_scores`·`frame_scores`는 객체 등장부터 영상 끝까지 모든 프레임의 J·F·J&F를 저장한다. GT가 없으면 `has_gt=0`, `gt_visible`과 점수는 NULL이다. GT가 있지만 객체가 보이지 않으면 `has_gt=1`, `gt_visible=0`이며 계산한 점수를 보존한다. 현재 표·회복률·곡선은 `has_gt=1 AND gt_visible=1`만 사용한다.

`native_frame_times`·`result_frame_times`에는 실제 측정한 Base+ replay 프레임 시간과 준비 이후 누적 GPU peak를 저장한다. Small과 측정하지 않은 미래 프레임은 NULL이다. 프롬프트 준비 비용은 `switch_seconds`에 포함되며 프레임 시간에 나누어 넣지 않는다. 프레임 시간이 없는 것을 0초로 해석하지 않는다.

원점수는 실행 조건별 트랜잭션으로 저장하고, 중단하면 완료된 조건만 재개에서 건너뛴다. 체크포인트 해시·설정으로 `experiment_id`를 구분하며 방법 revision도 고유 키에 포함한다. 예측 마스크와 방법별 기억 tensor는 저장하지 않는다. 복원율은 저장한 충분통계량으로 재계산할 수 있는 범위만 지원한다.

`PREFETCH_FRAMES=True`이면 Small 및 Base+의 측정하지 않는 미래 구간에서만 CPU worker가 다음 한 장을 준비한다. 측정할 준비/replay 프레임은 동기 로드하며 GPU 추론은 순차 실행한다. 실제 GPU 활용률과 전체 시간 개선은 서버에서 측정해야 한다.

`SCORING_WORKERS=1`이면 CPU 채점과 다음 GPU 프레임 추론을 겹친다. `SCORING_QUEUE_FRAMES=4`로 실행/대기 중인 마스크 수를 제한한다. 전환 준비 및 측정할 replay 구간에는 채점 worker를 만들지 않으며, Native의 측정 구간은 동기 채점한다. 각 추적이 끝나면 worker와 이미지 프리페치를 종료한 뒤 다음 측정을 시작한다. 저장 전에 모든 채점 완료를 확인하고 오류는 전파한다.

`GT_CACHE_MB=128`은 객체별 정답 마스크·경계·거리 변환 LRU의 보관 용량(MiB)이다. 객체의 여러 방법/전환이 같은 정답을 재사용한다. 긴 영상에서 캐시가 부족하면 재계산하며, 실제 RAM에는 대기 마스크와 worker 임시 배열도 추가된다. `SCORING_WORKERS=0`으로 동기 채점, `GT_CACHE_MB=0`으로 캐시를 끌 수 있다. OpenCV 내부 병렬화와 중복되지 않도록 worker는 기본 1개다.

DB 원점수 메타데이터의 `cpu_profile`·`pre_cpu_profile`에 추적 전체 시간, RGB 읽기/준비 시간, GT 읽기/준비·J/F 채점 시간, 큐 대기시간·캐시 적중 수를 남긴다. 프로파일은 구간별 비용이 아니라 추적 한 번의 기록이며, 여러 전환에서 공유한 기록은 `tracking_id`로 중복을 제거한다. worker 시간은 겹칠 수 있으므로 전체 시간처럼 합산하지 않는다. GPU 순수 실행시간은 새로 측정하지 않으며 기존 전환 비용 측정 범위를 유지한다.

`RGB_CACHE_MB=256`은 객체 실행 안에서 Small/Base+가 공유하는 정규화된 CPU 입력 LRU다. 세션 생성·프롬프트 준비·측정 replay에서는 읽기와 저장 모두 우회하며, 측정 없는 프레임에서만 사용한다. 캐시 키에는 파일 경로·크기·수정 시각·전처리 설정을 넣고 반환 tensor는 독립 복사한다. 실제 적중 수는 `cpu_profile.image_preparation.cache_hits`로 확인한다.

`RESTORATION_CACHE_MB=128`은 같은 객체/회차의 변경하지 않는 Native snapshot에 대한 float64 입력·평균·SST 캐시다. 기존 중심화된 SST, SSE, R² 공식과 N/A 사유를 유지한다. `restoration_cache_profile`에 결과별 적중·누락 수를 남긴다. 두 캐시는 각각 0으로 끌 수 있고, 작업 중 임시 배열은 보관 상한 외 RAM을 사용한다.

`BENCHMARK_SKIP_VISIBLE=True`는 benchmark가 사용하지 않는 presence scalar 조회를 생략한다. benchmark 출력의 `visible`은 None이며, 일반 Session은 기존 bool 값을 반환한다. SAM2 내부 presence logits와 last-visible 마스크 선택은 유지한다. SQLite는 평가 루프 동안 연결만 재사용하고 각 결과의 트랜잭션은 별도로 커밋한다. GPU 추론 중에는 DB 트랜잭션을 열어 두지 않는다.

추가 최적화는 실행 버전 5로 구분한다. 이전 실험과 재개 ID가 달라지며 `0_check_sam2.py`도 다시 실행해야 한다. 기존 실험은 저장된 ID를 선택해 계속 집계할 수 있다. 구현과 검증 범위는 [EXTRA_OPTIMIZATION.md](docs/EXTRA_OPTIMIZATION.md)에 있다.

SQLite journal은 기본 DELETE다. WAL은 로컬 디스크에서만 명시적으로 설정한다. 공유 볼륨의 파일 잠금 지원을 확인하고, 같은 조건을 두 프로세스에 배정하지 않는다. WAL 사용 시 실행 중 `.sqlite`만 복사하지 말고 SQLite backup API로 백업한다.

결과 파일:

- `outputs/tables/main.md`: 25/50/75%별 반복 평균 ± 표준편차.
- `outputs/tables/extra.md`: 공식 난이도 유형별 반복 통계.
- `outputs/tables/temporal.csv`: 전환 전후 전체의 J·J&F·Native와 프레임별 두 회복률 곡선, 반복 분산과 유효 표본 수.
- `outputs/tables/per_video.csv`: 회차별 영상 점수·전환 전/후 회복률과 구간별 평가/0분모/기준 미완료 프레임 수.
- `outputs/tables/native_reference.csv`: 영상·객체·프레임별 Native 원점수 분포의 평균·중앙값·표준편차·분산·J실패 횟수. 값은 원점수 척도(0~1).
- `outputs/tables/per_run.csv`: 회차별 데이터셋 대표 점수.
- `outputs/tables/summary.csv`: 데이터셋·전환·방법별 평균·표준편차·분산·회차 수.

Native의 선택 회차가 모두 모이지 않은 프레임은 기준을 확정하지 않으며 회복률은 N/A이다.
`--runs 1`로 명시하면 1회 기준으로 분석한다. 기본 `--runs 3`에서 부족한 회차를 자동으로 1~2회 중앙값으로 대체하지 않는다.
중앙값이 양수이면 낮은 기준도 그대로 사용한다. Native 실패(J≤0.5)를 이유로 회복률 대상을 자동 제외하지 않는다.
Native 비교군의 실제 3회 점수를 중앙값 기준으로 나누어 집계하므로 Native 회복률 평균이 항상100%인 것은 아니다.
`temporal.csv`의 native_j/native_jf는 집계 기준, native_run_j/native_run_jf는 실제 회차별 Native 점수의 집계이다.
평균±표준편차의 회복률은 고정된 Native 기준에 대한 방법의 반복 변동이며 기준값 자체의 추정 오차를 모두 포함하는 신뢰구간은 아니다.
미완료 회차가 섞이면 반복 통계는 회차 간 공통 유효 객체 기준이고, 곡선도 시간점별 공통 객체 기준이다.
표본 수와 회차 수를 반드시 확인한다. 한 회차만 있을 때 표준편차·분산은 N/A이며 0으로 쓰지 않는다.
`--seed`로 별도 seed 결과를 선택하고 `3_make_tables.py --runs N`으로 1~N회만 집계한다.

## 테스트 (GPU·데이터 없이, 가짜 모델·가짜 데이터)

```bash
python tests/test_metrics.py
python tests/test_recovery.py
python tests/test_restoration.py
python tests/test_recovery_curves.py
python tests/test_optimizations.py
python tests/test_end_to_end.py
python tests/test_validation.py
python tests/test_sqlite_store.py
python tests/test_experiment_selection.py
python tests/test_cpu_scoring.py
```

전환 시점의 `round`는 Python의 ties-to-even 규칙이다. 예를 들어 구간 차이가 9이면 50%는 `round(4.5)=4`다.
전환 전 열은 공통 Small 예측과 최초 정답 프롬프트 프레임을 포함한다. Native를 제외한 방법의 독립 측정으로 해석하지 않는다.
`per_run.csv`와 `summary.csv`의 기본 영상/객체 수는 J 기준이다. 회복률·R²은 각 지표의 `_video_count`·`_object_count`를 확인한다.

### 최단 구간 회복률 그래프

`python scripts/3_make_tables.py`는 데이터셋×전환 비율별 공통 구간 회복률 그래프도 만든다. Matplotlib이 필요하다(`pip install matplotlib`).

- `outputs/tables/recovery_common_window.csv`: 프레임별 J·J&F 회복률 평균, 반복 표준편차·분산, 지표별 유효 영상·객체·회차 수.
- `outputs/figures/<experiment_id>/recovery_common.seed<S>.runs<N>.<median|mean>/<데이터셋>.switch50.png` / `.pdf`, `switch25.png` / `.pdf`, `switch75.png` / `.pdf`: 그림 하나에 J·J&F 두 패널, 각 방법의 평균 곡선.
- 같은 폴더의 `windows.json`: n, 계획/완료 영상·객체 수, 구간을 제한한 객체, 기술적 미완료로 제외한 공통 사례, 그림에서 잘린 프레임 수.

n은 선택한 실험·seed·회차의 DB에 Native가 저장된 모든 객체의 `min(s-start, end-s)` 중 최솟값이다. 객체 최초 등장 기준이므로 늦게 등장한 객체가 최단 영상보다 짧은 범위를 만들 수 있다. 원래 상대 프레임 -n~+n을 쓰고 0=s는 마지막 Small 프레임, +1은 첫 전환 후 프레임이다. 같은 방법·회차 조건이 모두 완료된 객체를 공통 모집단으로 고정한다. 시간점마다 객체→영상→회차 평균을 계산한다. 공통 Small 전환 전 선은 Source-only로 표시한다. 로컬 영상 목록은 읽지 않는다. Native도 시작하지 않은 객체는 DB만으로 알 수 없으므로 범위에 포함되지 않는다. 이 기준은 `windows.json`에 `stored_native_object_ranges`로 기록한다.

**한계:** 긴 영상의 구간 밖은 그림에서만 잘리므로 장기 회복/악화를 대표하지 않는다. 같은 프레임 수는 같은 실제 시간·사건·난이도를 뜻하지 않는다. GT 가려짐/누락 또는 Native=0 때문에 유효 N(t)는 여전히 달라질 수 있다. 이를 CSV와 그림에 표시하고 보간하거나 0점으로 채우지 않는다. 실제 실패를 이유로 사례를 제외하지 않는다. 전체 구간 표와 원본 프레임 값은 보존한다. 곡선은 프레임별 비율의 평균이고, 표의 구간 평균 점수 비율과 구별한다. 최종 논문 표현은 추후 재검토한다.

그림을 생략하려면 `python scripts/3_make_tables.py --skip-recovery-plots`를 쓴다. 공통 구간 CSV와 메타데이터는 계속 저장한다.

## 평가 실행 최적화

중복 복사·측정·재추론을 줄이고 SQLite 완료 키를 조회하도록 변경했다. 변경 사항, 합성 데이터 검증 결과와 비용 비교 시 실행 버전 구분은 [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md)를 참고한다.
