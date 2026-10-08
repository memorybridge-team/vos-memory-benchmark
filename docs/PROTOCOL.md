# 평가 프로토콜

## 연구 주제

SAM2 Small이 객체를 추적하며 쌓은 기억을 SAM2 Base+로 넘긴 뒤의 성능과 전환 비용을 평가한다.
현재 비교군 5개와 본 모델 1개를 함께 실행한다.
회복률은 기존 계산을 유지하며, 회복률 변경과 복원율(R²)은 추후 논의한다.

- 본 모델은 팀 translator 전달본의 선정 epoch 27을 사용한다.
- 모델은 실행 전에 미리 로드한다. GPU가 여러 개여도 시간과 메모리 비교는 같은 종류의 GPU에서 한다.
- 객체마다 Small과 Base+ Native를 한 번씩 실행하고, 각 전환 이후 구간을 채점한다.
- 정답은 객체가 처음 나타난 프레임의 프롬프트 한 번에만 사용한다.

## 데이터
정답이 공개된 split 만 쓴다 (전환 뒤 프레임만 채점하려면 정답이 손에 있어야 함).
이 코드는 검증·평가만 돌린다.
| 용도 | 데이터 | 비고 |
|---|---|---|
| 학습 (본 모델) | MOSEv2 train + LVOS v2 train | 참고만. 이 코드에서는 쓰지 않음 |
| 검증 | LVOS v2 valid | 방법·설정을 고를 때 본다. 논문 결과로 보고하지 않음 |
| 평가 | M3VOS + PUMaVOS + VOST val | 모든 결정이 끝난 뒤 한 번 돌려 보고. VOST·M3VOS 는 무시 영역(255)을 채점에서 뺌 |

- 안 쓰는 것: MOSEv2 valid·LVOS v2 test (정답이 첫 프레임뿐), VOST test (정답 비공개, 서버 채점)
- M3VOS·PUMaVOS 는 train/test 구분이 없는 평가 전용 데이터셋 (M3VOS 의 `ImageSets/val.txt` 는 공개 영상 471개 전체 목록)
- PUMaVOS 는 공식 라벨이 없어 조건별 분류(EXTRAS)에서 빠진다
- 서버 위치는 `settings.DATA_ROOT` / `DATA_FOLDERS`

## 평가 단위

영상 하나 × 객체 하나 × 전환 시점 하나 × 비교군 하나 = 결과 한 줄
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

결과 줄은 `j`, `jf`, `n_frames`와 `frame_scores`를 저장한다.
`frame_scores`는 각 채점 프레임의 원래 번호, 전환 후 경과 프레임 수(f−s), J·F·J&F를 담는다.
경과 축은 우선 프레임 수를 사용한다. 원본 FPS를 추정해 초 단위로 바꾸지 않는다.
정답이 없거나 객체가 보이지 않는 프레임은 점수에 포함하지 않되 경과 프레임 번호는 유지한다.

`outputs/tables/temporal.csv`는 데이터셋·전환 비율·방법·경과 프레임마다
객체 평균 → 영상 평균으로 집계한 J·J&F(100점 만점)와 그 시간점의 기여 영상·객체 수를 기록한다.

### 기존 회복률(%)

회복률 계산식은 이번 수정에서 유지한다.
각 전환 뒤 구간의 방법 점수를 같은 구간의 Full Replay(Base+ Native) 점수로 나누고 100을 곱한다.

1. 영상마다 객체 점수를 평균한다.
2. 영상마다 방법 ÷ Full Replay × 100을 계산한다. Full Replay가 0인 영상은 제외한다.
3. 영상별 비율을 평균한다.

J 회복률과 J&F 회복률을 함께 보고한다. 복원율(R²)은 아직 계산하지 않는다.

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

공식 라벨과 데이터셋 영상명에서 읽은 라벨로 J·J&F·실패비율 및 기존 회복률을 나눠 보고한다.
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

현재 평가 결과는 `evaluation_revision = 2`와 방법별 `baseline_revision`을 저장한다.
이전 결과에는 경과 프레임별 점수와 새 전환 비용이 없어 재사용하지 않는다. 기존 파일을 삭제하지 않고 새 결과를 추가한다.
표와 이어하기는 현재 평가 버전·전환 이름·방법 버전이 맞는 결과만 사용한다.
Original + Last-Visible의 방법 버전은 2, 나머지는 1이다.

이어하기는 영상·객체·전환 이름·실제 전환 프레임·방법 단위이다.
50%만 완료된 방법은 75%도 완료된 것으로 처리하지 않는다.
이전 영상 목록은 사용할 수 없으므로 `python scripts/1_make_video_list.py`로 먼저 재생성한다.

## 보고 파일

- `outputs/tables/main.md`: 50/75%별 전체 평가표.
- `outputs/tables/extra.md`: 공식 라벨 및 공통 난이도 유형별 성능.
- `outputs/tables/temporal.csv`: 전환 후 경과 프레임별 J·J&F.

출력 일치도, 전체 추적시간, 속도 배수, 다시 본 프레임 수 등 목록에 없는 지표는 계산·보고하지 않는다.
