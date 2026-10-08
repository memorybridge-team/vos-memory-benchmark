# 평가 프로토콜

## 연구 주제

SAM2 Small이 객체를 추적하며 쌓은 기억을 SAM2 Base+로 넘긴 뒤의 성능과 전환 비용을 평가한다.
현재 비교군 5개와 본 모델 1개를 함께 실행한다.
J·J&F 회복률을 프레임별로 계산하며 전체 평가를 기본 3회 반복한다. 복원율(R²)은 전환 시점의 기억 프레임별로 계산·저장하며 최종 집계와 표현 방식은 미정이다.

- 본 모델은 팀 translator 전달본의 선정 epoch 27을 사용한다.
- 모델은 실행 전에 미리 로드한다. GPU가 여러 개여도 시간과 메모리 비교는 같은 종류의 GPU에서 한다.
- 각 회차의 객체마다 Small과 Base+ Native를 한 번씩 실행하고, 50/75% 전환에 공유한다.
- 정답은 객체가 처음 나타난 프레임의 프롬프트 한 번에만 사용한다.

## 데이터
정답이 공개된 split 만 쓴다 (전환 뒤 프레임만 채점하려면 정답이 손에 있어야 함).
이 코드는 검증·평가만 돌린다.
| 용도 | 데이터 | 비고 |
|---|---|---|
| 학습 (본 모델) | MOSEv2 train + LVOS v2 train | 참고만. 이 코드에서는 쓰지 않음 |
| 검증 | LVOS v2 valid | 방법·설정을 고를 때 본다. 논문 결과로 보고하지 않음 |
| 평가 | M3VOS + PUMaVOS + VOST val | 모든 결정이 끝난 뒤 같은 프로토콜로 3회 실행해 보고. VOST·M3VOS 는 무시 영역(255)을 채점에서 뺌 |

- 안 쓰는 것: MOSEv2 valid·LVOS v2 test (정답이 첫 프레임뿐), VOST test (정답 비공개, 서버 채점)
- M3VOS·PUMaVOS 는 train/test 구분이 없는 평가 전용 데이터셋 (M3VOS 의 `ImageSets/val.txt` 는 공개 영상 471개 전체 목록)
- PUMaVOS 는 공식 라벨이 없어 조건별 분류(EXTRAS)에서 빠진다
- 서버 위치는 `settings.DATA_ROOT` / `DATA_FOLDERS`

## 평가 단위

반복 번호 × seed × 영상 하나 × 객체 하나 × 전환 시점 하나 × 방법 하나 = 결과 한 줄
각 객체가 처음 보인 프레임에서 정답 마스크를 주고 시작해 영상 끝까지 
**정답은 이 처음 프롬프트에만 들어간다.** 나머지 전환 때 넘기는 것은 모두 Small이 스스로 낸 것

## 전환 시점

객체가 처음 나타난 프레임을 a, 영상 마지막 프레임을 e라고 할 때
s = a + round(f × (e − a)), f ∈ {0.50, 0.75}이다.
전환 이름은 "50", "75"이고, Small은 s까지, Base+는 s+1부터 처리한다.
짧은 구간에서는 s를 a+1과 e−1 사이로 제한한다.
객체 최초 등장 기준을 유지하므로 같은 영상의 객체라도 전환 프레임은 다를 수 있다.

## 비교군 5개

| 비교군 | Base+가 전달받거나 사용하는 데이터 |
|---|---|
| Source-only | Base+를 사용하지 않음. Small이 자신의 기억으로 영상 끝까지 낸 예측 |
| Full Replay (Base+ Native) | 처음 실제 정답 마스크와 해당 프레임 이미지 + 전환 전까지 모든 프레임. Base+가 처음부터 처리한 결과를 전환 뒤 구간에서 채점 |
| Direct State Copy | Small의 maskmem_features, obj_ptr와 기억 메타데이터를 변환 없이 복사 |
| Original + Last-Visible | 처음 실제 정답 마스크·해당 이미지 + Small의 마지막 비어 있지 않은 예측 마스크·해당 이미지. Small 기억은 전송하지 않음 |
| Original-Prompt(s)+Replay-8 | 처음 실제 정답 마스크·해당 이미지 + 전환 직전 8프레임을 Base+가 다시 보고 s+1부터 추적 |

현재 입력 프로토콜은 객체마다 처음 정답 프롬프트 한 번만 주며 수정 기록은 없다.
Full Replay도 그 실제 입력을 동일하게 사용한다. 향후 수정 프롬프트를 지원하면 프레임·마스크 기록을 함께 재현해야 한다.
Original + Last-Visible은 object_score_logits와 관계없이 비어 있지 않은 예측을 고른다.
마지막 예측이 처음 프레임이거나 한 번도 비어 있지 않은 예측이 없으면 처음 정답만 사용한다.
Replay는 s−7~s를 다시 본다. 처음 프롬프트 이후 구간이 8프레임보다 짧으면 있는 구간만 다시 본다.

## 본 모델

| 방법 | Base+ 가 받는 것 |
|---|---|
| 본 모델 (translator) | Small memory bank 의 칸마다 `maskmem_features`·`obj_ptr` 를 translator 로 바꾼 것 (다시 보기 없음) |

팀 평가 코드(`vos_memory_inspector.lvos_evaluation.no_replay_case`)와 같게 한다:
- 칸마다 따로 바꾼다 (translator 는 칸끼리 섞지 않음). 바꾸는 동안 autocast 를 끄고 fp32 로 계산하고,
  결과는 원래 dtype (`maskmem_features` bf16, `obj_ptr` fp32) 으로 넣는다.
- 나머지 칸 (`maskmem_pos_enc`, `pred_masks`, `object_score_logits`) 은 Direct State Copy 처럼 Small 것 그대로.
  Base+ 가 s+1 부터 이어갈 때 SAM2 는 지난 칸의 `maskmem_features`·`maskmem_pos_enc`·`obj_ptr` 만 읽고,
  `maskmem_pos_enc` 는 두 모델이 같은 값이다 (`0_check_sam2.py` 로 확인).
- translator 를 GPU 에 올리는 시간은 모델 로딩이라 시간 열에 넣지 않는다. 바꾸는 시간은 넣는다 (준비 시간).

## 평가 지표

### J, F, J&F

- J = 예측 마스크와 정답 마스크 면적의 IoU.
- F = 테두리 precision과 recall의 조화평균.
  예측 테두리의 점은 가장 가까운 정답 테두리까지의 유클리드 거리가 이미지 대각선 × 0.008 픽셀 이하이면 맞은 점이다.
  recall도 반대 방향에서 같은 허용 거리를 사용한다.
- 허용 거리를 정수 픽셀로 올림하지 않는다.
- J&F = (대상 프레임의 J 평균 + 대상 프레임의 F 평균) ÷ 2.
- VOST·M3VOS의 무시값 255는 J·F 계산에서 제외한다.

채점 대상은 기존처럼 전환 뒤 정답에 객체가 보이는 프레임이다.
LVOS v2·PUMaVOS의 주 지표는 J&F, VOST·M3VOS의 주 지표는 J이며 모든 데이터셋에 J·J&F를 함께 보고한다.
영상별 객체 평균을 구한 뒤 영상 평균을 보고하며, 50%·75%는 별도 표로 만든다.

결과 줄은 `run_id`, `seed`, `native_reference_id`, `j`, `jf`, `n_frames`, `failure_rate`를 저장한다.
`frame_scores`는 전환 이후 평가 프레임의 원래 번호, 상대 프레임(f−s), J·F·J&F,
동일 회차 Native의 J·J&F 원점수(`native_run_j`, `native_run_jf`)를 담는다.
추론 결과에서는 `recovery_reference=pending`, 회복률은 None이다. 집계 후 별도 분석 JSONL에
Native 반복 기준(`native_j`, `native_jf`)과 `recovery_j`, `recovery_jf`를 저장한다.
`pre_switch_frame_scores`는 처음 등장~s의 원점수와 Native 원점수를 담으며 회복률은 None이다.
Native 비교군의 전환 전 점수는 Native, 나머지 방법은 Small의 동일 전환 전 예측이다.
정답이 없거나 객체가 보이지 않는 프레임은 제외하되 상대 프레임 번호는 유지한다.
원본 FPS를 추정해 초 단위로 바꾸지 않는다.

### J 회복률과 J&F 회복률(%)

각 방법의 반복 r, 영상 v, 객체 o, 프레임 t에 대해:

`recovery_Q(r,v,o,t) = 100 × Q_method(r,v,o,t) / median_r′(Q_Native(r′,v,o,t))`, Q ∈ {J, J&F}.

- 같은 영상·객체·프레임의 Native 3회 중앙값을 모든 방법·회차의 공통 분모로 사용한다.
- J와 J&F의 중앙값은 별도로 계산한다. median((J+F)/2)는 (median(J)+median(F))/2와 다를 수 있다.
- 실패0도 Native 표본에 포함한다. 중앙값을 계산하기 전에 낮은 점수를 삭제하지 않는다.
- 선택한 모든 회차의 해당 Native 원점수가 있어야 기준을 확정한다. 미완료면 회복률은 None이다.
- `3_make_tables.py --native-statistic mean`이면 같은 원점수에서 평균 분모로 재집계한다.
- Native는 처음 실제 정답 프롬프트부터 끝까지 실행한다. 50/75%는 같은 실행에서 구간만 나눈다.
- 분모가 0 또는 누락이면 해당 회복률은 None(N/A). 분모에 작은 상수를 더하지 않는다.
- 양수인 낮은 Native 기준은 자동 제외하지 않는다. Native 실패(J≤0.5) 횟수는 별도 기록하며 기준의 안정성 판단은 별도 문제이다.
- 추후 처리 계획: 현재는 반복 실패의 발생 빈도를 모르므로 중앙값을 기본으로 유지한다. 반복 측정에서 Native 실패가 연속해서 빈번히 나타나면 실패 사례를 제외한 표를 별도 재집계하는 방안을 검토한다. 구체적인 제외 기준·단위는 그때 정하고 제외 건수를 명시한다. 모든 방법에 같은 사례 선택을 적용하고 원본과 실패 포함 결과를 보존하므로 추론 없이 사후 분석할 수 있다. 현재 자동 제외는 구현하지 않는다.
- Native 비교군도 실제 회차 원점수/중앙값 기준으로 집계하므로 평균 회복률이 항상100%가 되는 것은 아니다.
- 방법 점수가 0이고 Native가 양수이면 회복률 0으로 포함한다. 100% 초과도 유지한다.
- 구간 요약은 **프레임별 회복률의 평균**이다. 구간 원점수 평균끼리 나누는 기존 정의는 제거한다.
- `recovery_j_n_frames`, `recovery_jf_n_frames`와 각 지표의 0/누락 Native 프레임 수를 저장한다.

집계 순서:

1. 객체마다 s+1~끝의 유효 프레임 회복률 평균을 구해 별도 분석 JSONL 줄에 저장한다.
2. 회차마다 영상 안의 객체 평균 → 영상 평균을 구한다.
3. 회차 대표 점수의 평균, 표본 표준편차와 표본 분산(ddof=1)을 보고한다.
4. 경과 프레임별 곡선도 같은 시간점에서 객체 → 영상 → 회차 순서로 집계한다.

평균은 실제 실패도 반복 성능에 반영한다. 회차별 원점수를 별도 보존해 실패의 영향을 확인할 수 있다.
낮은 성능/빈 예측은 제거하거나 최고 점수로 대체하지 않는다. 프로그램 오류는 0점으로 바꾸지 않는다.
한 회차만 있으면 반복 표준편차/분산은 None이다. 표본 수가 작다는 한계를 회차 수와 함께 명시한다.
부분 완료 결과 또는 모든 프레임의 Native가 0인 객체는 지표별 회차 간 공통 유효 객체로 집계하며 표본 수를 함께 보고한다.
Native 점수도 회차별 집계 후 평균·표준편차·분산을 동일하게 제공한다.
회복률의 반복 표준편차는 계산된 Native 공통 기준에 대한 방법의 변동이다. Native 기준값의 추정 불확실성을 모두 반영하는 신뢰구간으로 해석하지 않는다.

### R² 복원율: 기억 프레임별 계산만

각 회차의 영상·객체·전환 s에서 Base+가 s+1을 처리하기 직전, 준비된 기억과 그 회차의 Native 기억을 비교한다.
여기서 프레임별이란 전환 시점의 memory bank 안에 있는 기억 칸의 원래 프레임 번호이다.
전환 이후 매 추적 프레임의 R² 곡선을 계산하는 것은 아니다.
Native는 같은 첫 정답 프롬프트와 영상으로 처음부터 s까지 추적한 상태이다.
J/J&F 회복률의 3회 중앙값과 별개로, 기억 R²의 기준은 같은 회차의 실제 Native tensor이다.

기억 칸의 `maskmem_features`와 `obj_ptr`를 각각 펼쳐 계산한다.

`R²(field, frame) = 1 − sum((prepared − native)²) / sum((native − mean(native))²)`.

- 두 필드를 섞지 않고, Native를 기준값(y_true), 준비된 Target 기억을 비교값(y_pred)으로 둔다.
- 비교 전 정확한 tensor shape를 확인한다. 계산은 CPU float64이고 입력 dtype/값은 바꾸지 않는다.
- 1은 동일함, 0은 Native의 평균값으로 대체했을 때와 같은 오차이다. 음수도 유지하고 %로 바꾸지 않는다.
- Native 분산이 0이면 동일한 tensor라도 R²은 N/A이다. 0이나 1로 대체하지 않는다.
- 누락 칸/필드, 형상 불일치, 원소 수 부족, 비유한 값은 N/A와 사유를 저장한다.
- 양쪽 memory bank의 프레임 합집합으로 대응하므로 누락된 기억도 기록한다. 0 tensor로 채우지 않는다.
- 정답 가시성이나 J 실패를 이유로 기억 칸을 제외하지 않는다.
- Native 기준은 자기 기억끼리 비교한다. Source-only에는 Target 기억이 없어 적용하지 않는다.
- Direct State Copy/translator는 실제로 Base+에 넣고 준비한 기억을 비교한다.
- Anchor/Replay도 준비된 Base+ 기억을 비교한다. 기억 칸 구성이 다르므로 공통 칸에서만 값이 있으며 누락 사유를 함께 봐야 한다.
- 값에는 전환 전 Small의 누적 추적 오차가 포함될 수 있다. translator만의 변환 오차로 해석하지 않는다.

결과 JSONL의 `restoration_frame_scores`에 기억 프레임, s에서 떨어진 프레임 수, cond 여부,
필드별 `r2`, `sse`, `sst`, `native_mean`, `n_elements`, shape, status를 저장한다.
회차·영상·객체·전환·방법·Native ID는 상위 결과 줄의 메타데이터를 사용한다.
SSE/SST/평균/원소 수는 이후 칸 평균 또는 전체 tensor 기준으로 다시 집계할 때 사용할 수 있다.
이 충분통계량만으로 임의의 새 특징 유사도 지표를 계산할 수 있는 것은 아니다.

<!-- TODO: R²의 칸/객체/영상/회차 집계와 최종 표·그래프를 결정한다. 현재 대표 평균 R²이나 복원율 CSV/표를 만들지 않는다. -->

R² 계산·통계용 CPU 기억 추출·파일 저장은 전환시간과 GPU peak 측정 밖에서 실행한다.
Native는 50/75%에서 두 필드의 CPU 기억을 snapshot으로 꺼내 `outputs/native_memory/<native_reference_id>.pt`에 먼저 보존한다.
이어하기에서는 이 파일을 재사용한다. 유실되면 Native를 다시 실행해 조용히 기준을 바꾸지 않고 오류를 낸다.
`outputs/records`, `outputs/native`, `outputs/native_memory`를 함께 보존한다.
Native tensor snapshot 때문에 CPU RAM과 디스크 사용량이 늘어난다. tensor dtype/기억 칸 수에 따라 달라지며,
(1,64,64,64) BF16 feature 하나는 약 0.5 MiB이다. 같은 회차에 두 전환 snapshot을 저장한다.
준비된 Target의 전체 tensor는 별도 파일로 저장하지 않고 프레임별 R² 및 충분통계량만 기록한다.
버전 3 원점수에는 기억 snapshot이 없으므로 R²을 소급 계산할 수 없다. 기존 파일은 보존하며 버전 4 평가 결과와 섞지 않는다.

### 전환시간(초)

`switch_seconds` = 준비 시간 + 전환 프레임 s까지의 replay 시간.
기억을 Target에 옮기는 시간, Target 형식으로 변환하고 넣는 시간, 프롬프트를 기억으로 인코딩하는 시간을 포함한다.
Small에서 기억을 꺼내는 시간, 모델 로딩, Base+ 세션 생성, s+1 이후 추적·채점 시간은 제외한다.
두 모델과 Target 세션이 준비된 상태에서 측정한다. GPU는 측정 전후 동기화한다.

| 방법 | 측정 구간 |
|---|---|
| Source-only | 전환 없음 → "-" |
| Full Replay (Base+ Native) | 처음 정답 인코딩 + 처음부터 s까지 처리 |
| Direct State Copy | Small 기억을 Target에 복사·넣기 |
| Original + Last-Visible | 두 프롬프트 추가·기억 인코딩 |
| Original-Prompt(s)+Replay-8 | 처음 프롬프트 추가·인코딩 + 전환 직전 최대 8프레임 replay |
| 본 모델 | translator 변환 + Target에 기억 넣기 |

### 전환 GPU 메모리(MB)

`switch_gpu_mb` = 전환 구간 최고 GPU 메모리 사용량 − 전환 준비 직전 GPU 메모리 사용량.
전환시간과 같은 구간을 사용하며 s+1 이후 추적에서 발생한 최고치는 포함하지 않는다.
PyTorch가 현재 GPU에 할당한 메모리를 MiB(2²⁰ 바이트) 단위로 측정한다.
Source-only는 전환이 없고, GPU가 없는 테스트 환경에서는 값이 없다.

### 난이도 유형별 성능

공식 라벨과 데이터셋 영상명에서 읽은 라벨로 J·J&F·실패비율 및 프레임별 회복률을 나눠 보고한다.
공통 유형은 가려짐, 모양·상태 변화, 비슷한 객체이며 매핑은 `evaluation/scoring/extra_groups.py`에 둔다.
라벨이 없는 데이터셋을 추정으로 분류하지 않는다.
라벨은 영상·객체 전체에 붙어 있으므로 전환 뒤에 그 사건이 일어났는지는 구분하지 않는다.
출처와 매핑은 `docs/EXTRAS.md`를 참고한다.

### 실패비율

`failure_rate` = 1 − J_Recall.
J_Recall = (J > 50인 대상 프레임 수) ÷ (대상 프레임 수).
내부 점수는 0~1이므로 J > 0.5로 계산한다. J가 정확히 0.5인 프레임은 실패이다.
대상은 J·J&F와 같은 프레임이며 대상이 없으면 None이다. 표에서는 %로 보고한다.

## 결과 저장과 이어하기

결과 버전은 `evaluation_revision = 4`이며 방법별 `baseline_revision`을 함께 저장한다.
이전 결과는 삭제하지 않지만 새 회복률/반복 통계와 섞지 않는다.
객체 기준 50/75% 영상 목록의 버전은 2로 유지하여 기존 목록을 재사용한다.
Original + Last-Visible의 방법 버전은 2, 나머지는 1이다.

- `outputs/records/<dataset>.run<r>[.shard<i>of<n>].jsonl`: 회차별 결과.
- `outputs/native/<dataset>.run<r>[.shard<i>of<n>].jsonl`: 회차별 Native 기준 원점수와 전환 비용.
- `outputs/native_memory/<native_reference_id>.pt`: 같은 Native 회차의 전환 시점 기억 기준(두 필드의 CPU tensor).
- Native 기준은 생성 즉시 먼저 저장한다. 이어하기에서는 원래 기준을 재사용한다.
- Native 파일이 유실되었는데 해당 회차 결과가 남아 있다면 새 분모를 섞지 않고 오류를 낸다.
- records/native/native_memory 세 폴더를 함께 보존한다. 같은 조건을 여러 작업에 동시에 맡기지 않는다.

이어하기 키는 반복 번호·seed·영상·객체·전환 이름·실제 전환 프레임·방법이다.
각 회차는 다른 회차의 완료 기록에 의해 건너뛰지 않는다. 일부 전환만 빠진 경우 누락된 줄만 추가한다.
조건별 seed를 고정하지만 SAM2 실행의 결정성을 보장하거나 인위적으로 변동을 만들지는 않는다.

기본 3회 실행: `python scripts/2_evaluate.py --dataset vost_val`.
단일 회차: `--run-id 2`. 1회만 실행: `--runs 1`. 기준 seed: `--seed 0`.
GPU 분할은 기존 `--shard i/n`을 사용한다.

<!-- TODO: 최종 논문 표현 방식은 미정이다. 실제 상대 프레임/% 축 및 곡선/분포 구성은 R² 복원율 정의·구현 후 재논의한다. 원본 프레임별 값을 보존하며 현재 CSV/표는 중간 집계이다. -->

## 보고 파일

- `main.md`: 데이터셋·전환별 평균 ± 반복 표준편차와 회차 수.
- `extra.md`: 난이도 유형별 동일한 집계.
- `temporal.csv`: 전환 전후 원점수와 전환 후 J/J&F 회복률 곡선, 평균·분산·표준편차·표본 수.
- `per_video.csv`: 영상별 회차 결과와 유효/0분모/기준 미완료 프레임 수.
- `native_reference.csv`: 원점수 척도(0~1)의 프레임별 Native 평균·중앙값·표준편차·분산·J실패 횟수.
- `per_run.csv`: 데이터셋·전환·방법별 회차 대표 점수.
- `summary.csv`: 같은 조건의 반복 평균·표준편차·분산·유효 회차 수.

위 CSV/Markdown은 `outputs/tables/`에 저장하며 마지막 집계로 갱신된다.
재계산한 회복률 JSONL은 `outputs/analysis/recovery.seed<S>.runs<N>.<median|mean>.jsonl`에 정의별로 보존한다.
원본 records/native는 집계에서 변경하지 않는다.
`temporal.csv`의 native_j/native_jf는 반복 기준, native_run_j/native_run_jf는 실제 회차 Native 점수이다. `3_make_tables.py --runs N --seed S`로 1~N회/seed를 선택한다.
전환 전 곡선의 `frames_after_switch`는 0 이하, `phase=pre`이며 전환 후는 양수, `phase=post`이다.
점수는 100점 척도, 회복률/실패비율은 %, 전환시간은 초, GPU 메모리는 MiB이다.

실패 제외 분석은 원본을 변경하지 않고 별도 분석에서 적용한다. 방법마다 서로 다른 사례를 제외하여 비교하지 않는다.
복원율의 칸/객체/영상/반복 요약값과 표·그래프는 아직 만들지 않는다. 출력 일치도, 전체 추적시간, 속도 배수 등 아직 요청되지 않은 지표는 추가하지 않는다.
