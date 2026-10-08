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

기본 실행은 모든 방법을 3회 평가한다. 회차별 결과는 `outputs/records/<데이터셋>.run1[.shard0of2].jsonl` 형식이다.

```bash
# 한 회차만 시험 실행
python scripts/2_evaluate.py --dataset vost_val --runs 1
# 특정 회차만 실행/재개
python scripts/2_evaluate.py --dataset vost_val --run-id 2
# 기본 3회 결과를 집계
python scripts/3_make_tables.py
```

이어하기 키는 반복 번호·seed·영상·객체·전환 이름·실제 전환 프레임·방법이다.
GPU 수를 바꾸어 재개할 수 있지만 같은 객체를 두 프로세스에서 동시에 맡기지는 않는다.
Native와 Source-only는 회차마다 객체당 한 번씩 실행하고, 같은 결과에서 50%·75% 구간을 잘라 쓴다.
Native 원점수와 전환 비용은 `outputs/native/<데이터셋>.run1[.shard0of2].jsonl`에 먼저 저장한다.
이 파일을 이어하기에서도 재사용하므로 회차별 Native 원점수가 바뀌지 않는다. `outputs/records`와 `outputs/native`를 함께 보존한다.

프레임별 J·J&F 회복률은 같은 영상·객체·프레임의 방법 점수 / Native 반복 중앙값 × 100이다.
표에는 전환 전·후 각각의 구간 평균 점수 / 같은 프레임의 Native 기준 평균 × 100을 네 열로 표시한다. 프레임별 비율 평균과는 다르다. 전환 전은 객체 최초 등장~s, 전환 후는 s+1~끝이다.
Native를 제외한 방법의 전환 전 점수는 공통 Small 예측이며, Native 행은 자체 예측을 사용한다.
J의 중앙값과 J&F의 중앙값을 별도로 구한다. J&F의 기준은 각 회차의 (J+F)/2를 구한 뒤 그 값들의 중앙값이다.
추론 JSONL에는 원점수와 `recovery_reference=pending`을 저장한다. 회복률은 3회 Native가 모인 뒤 3_make_tables.py에서 계산한다.
Native=80,80,0이면 중앙값80을 모든 방법·회차에 공유하며 실패0도 중앙값 표본에서 삭제하지 않는다.
평균 기준으로 바꾸려면 추론 없이 아래 명령만 실행한다.

```bash
python scripts/3_make_tables.py --native-statistic median  # 기본값
python scripts/3_make_tables.py --native-statistic mean
```

원본 records/native는 변경하지 않는다. 정의별 회복률 JSONL은 `outputs/analysis/recovery.seed0.runs3.median.ratio_of_means.jsonl` 또는 `.mean.ratio_of_means.jsonl`에 별도 저장한다.
`outputs/tables`의 CSV/Markdown은 마지막 집계 결과로 갱신된다.
객체별 전환 전/후 구간 평균의 비율 → 영상 내 객체 평균 → 영상 평균으로 회차 점수를 만든 뒤, 반복 평균·표본 표준편차(ddof=1)·분산을 보고한다.
Native 반복 기준이 0인 프레임도 구간 평균에 포함한다. 구간 Native 평균이 0이면 구간 회복률만 N/A이며 원점수와 실패비율은 유지한다. 누락/미완료 Native 프레임은 분자·분모에서 함께 제외하고 개수를 저장한다. 100%를 넘는 회복률도 그대로 저장한다.
낮은 성능과 빈 예측은 모든 회차에서 포함한다. 실행 오류는 점수 0으로 바꾸지 않고 중단 후 이어한다.
전환 이전은 Small(또는 Native 비교군)의 원점수와 Native 원점수를 저장하고, 전환 이후 전체의 회복률 곡선을 보고한다.
전환 축은 객체 최초 등장~영상 끝의 50%·75%이며, 경과 프레임 번호를 압축하지 않는다.
복원율(R²)은 전환 시점의 기억 프레임별로 `maskmem_features`와 `obj_ptr`를 각각 계산한다.
같은 회차 Native와 비교한 R²·SSE·SST·Native 평균·원소 수·N/A 사유를 결과 JSONL의 `restoration_frame_scores`에 저장한다.
Native 기억 기준은 `outputs/native_memory/<native_reference_id>.pt`에 보존하며 이어하기에 재사용한다.
records/native/native_memory 세 폴더를 함께 보존한다. CPU RAM과 디스크 사용량은 늘어난다.
복원율은 준비된 Target 전체 기억 원소의 R²을 재계산한 뒤 객체→영상→회차 순서로 평균한다.
기존 표에 spatial/pointer 두 R² 열을 평균 ± 반복 표준편차로 보고한다. 칸별 R²의 단순 평균은 아니다.
프레임별 충분통계량은 그대로 보존하므로 재추론 없이 집계하며, 복원율은 원래 R² 척도로 표시한다. 자세한 정의는 docs/PROTOCOL.md를 참고한다.

현재 결과 버전은 4이며 예전 결과 파일은 보존하되 새 집계에 섞지 않는다.
객체 기준 50%·75%로 만든 버전 2 영상 목록은 그대로 사용할 수 있다. 더 오래된 목록은 1_make_video_list.py로 다시 만든다.
조건별 seed는 반복 번호와 영상·객체·방법으로 결정되어 분할/이어하기 순서에 영향을 받지 않는다.
seed 변경이 추론 결과의 변동을 보장하지는 않는다. 동일한 결과가 반복되면 표준편차는 0이다.
시간과 GPU 메모리 비교에는 같은 종류의 GPU를 사용한다.

결과 파일:

- `outputs/tables/main.md`: 50/75%별 반복 평균 ± 표준편차.
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
python tests/test_end_to_end.py
```

### 최단 구간 회복률 그래프

`python scripts/3_make_tables.py`는 데이터셋×전환 비율별 공통 구간 회복률 그래프도 만든다. Matplotlib이 필요하다(`pip install matplotlib`).

- `outputs/tables/recovery_common_window.csv`: 프레임별 J·J&F 회복률 평균, 반복 표준편차·분산, 지표별 유효 영상·객체·회차 수.
- `outputs/figures/recovery_common.seed<S>.runs<N>.<median|mean>/<데이터셋>.switch50.png` / `.pdf`, `switch75.png` / `.pdf`: 그림 하나에 J·J&F 두 패널, 각 방법의 평균 곡선.
- 같은 폴더의 `windows.json`: n, 계획/완료 영상·객체 수, 구간을 제한한 객체, 기술적 미완료로 제외한 공통 사례, 그림에서 잘린 프레임 수.

n은 전환 비율별 모든 계획 객체의 `min(s-start, end-s)` 중 최솟값이다. 객체 최초 등장 기준이므로 늦게 등장한 객체가 최단 영상보다 짧은 범위를 만들 수 있다. 원래 상대 프레임 -n~+n을 쓰고 0=s는 마지막 Small 프레임, +1은 첫 전환 후 프레임이다. 같은 방법·회차 조건이 모두 완료된 객체를 공통 모집단으로 고정한다. 시간점마다 객체→영상→회차 평균을 계산한다. 공통 Small 전환 전 선은 Source-only로 표시한다.

**한계:** 긴 영상의 구간 밖은 그림에서만 잘리므로 장기 회복/악화를 대표하지 않는다. 같은 프레임 수는 같은 실제 시간·사건·난이도를 뜻하지 않는다. GT 가려짐/누락 또는 Native=0 때문에 유효 N(t)는 여전히 달라질 수 있다. 이를 CSV와 그림에 표시하고 보간하거나 0점으로 채우지 않는다. 실제 실패를 이유로 사례를 제외하지 않는다. 전체 구간 표와 원본 프레임 값은 보존한다. 곡선은 프레임별 비율의 평균이고, 표의 구간 평균 점수 비율과 구별한다. 최종 논문 표현은 추후 재검토한다.

그림을 생략하려면 `python scripts/3_make_tables.py --skip-recovery-plots`를 쓴다. 공통 구간 CSV와 메타데이터는 계속 저장한다.
