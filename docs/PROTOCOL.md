# Protocol (확정)

## 연구 주제

Video Segmentation을 실행하던 도중 작은 모델(SAM2 Small)에서 큰 모델(SAM2 Base+)로 전환할 때
**Small이 지금까지 쌓은 기억을 Base+ 에 얼마나 잘 넘기는지**를 측정한다.
주 측정 방식인 회복률은 (현재 방식으로 산출한 점수) / (Full Replay(Base+로 처음부터 끝까지 돌림)로 산출한 점수) * 100 으로 계산한다.

## 평가하는 방법: 본 모델 1개 + 비교군 9개 (+ extra 진단 비교군 2개, EXTRAS.md)

- 본 모델 = 팀 translator. 전달본 `official_state_loss_final_delivery` 의 선정 epoch 27
  (`selected_state_loss_best/translator_weights.pth`, SHA256 `92802842…`). 비교군과 같은 영상 목록·전환 시점·지표·회복률 계산.
- 본 모델과 비교군은 `2_evaluate.py` 한 번에 같이 돌린다. 따로 돌리면 Full Replay 를 두 번 계산하게 되고,
  출력 일치도는 Full Replay 마스크가 필요해 나중에 결과 파일만으로 계산할 수 없다.
- 이어하기는 (영상, 객체, 방법) 단위 → 예전에 본 모델만 따로 돌린 결과(`<데이터셋>.model.jsonl`)가 있으면 그 객체는 나머지 방법만 돈다.
  같은 Full Replay·Source-only 줄이 두 파일에 있으면 표는 한 번만 센다.
- 시간 열을 서로 비교하려면 모든 실행을 같은 종류의 GPU 에서 돌린다 (`--shard` 로 나눌 때도).

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

전환 프레임 s = Small이 마지막으로 본 프레임. Base+ 는 s+1 부터.

객체 구간 [시작, 영상 끝]의 25 / 50 / 75% 지점 (`settings.SWITCH_FRACTIONS`). 전환 이름은 "25", "50", "75".

전환 전 구간(시작 ~ s)은 모든 비교군이 **같은 Small 결과**를 쓴다. 비교군끼리 다른 것은 전환 뒤뿐.

## 비교군 9개

| 비교군 | Base+ 가 받는 것 |
|---|---|
| Source-only | 안 넘김. Small이 끝까지 |
| Full Replay | 처음 정답, 그리고 처음부터 s 까지 전부 다시 봄 (100점 기준) |
| Direct State Copy | Small memory bank 그대로 |
| Original-Prompt(s) | 처음 정답 마스크만 |
| Last-Visible | Small이 마지막으로 "보인다"고 한 프레임의 Small 마스크만 |
| Original+Last-Visible | 위 둘 다 |
| Original+Replay-4/8/16 | 처음 정답 + 전환 직전 K 프레임을 Base+ 가 다시 봄 |

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

## 주요 평가 지표

### J, J&F

- **J** = 겹친 넓이 ÷ 합친 넓이
- **F** = 예측 테두리가 정답 테두리에 얼마나 가까운가 (대각선 × 0.008 픽셀 허용)
- **J&F** = 둘의 평균
- 무시 영역 픽셀(VOST, M3VOS 의 255)은 J, F 모두에서 뺀다

| 데이터셋 | 채점 구간 | 주 지표 |
|---|---|---|
| LVOS v2, PUMaVOS | 전환 뒤, 정답에 객체가 보이는 프레임만 | J&F |
| VOST, M3VOS | 전환 뒤, 정답에 객체가 보이는 프레임만 | **J** (VOST 는 그 특징 때문에 F 를 쓰지 않고, M3VOS 논문도 F 로 평가하지 않음 — 이유는 밝히지 않음) |

결과 줄 열: `j`, `jf` (`settings.J_MAIN_DATASETS` 가 주 지표가 J 인 데이터셋).

### 회복률(%)

LLM 모델 전환 논문(arXiv 2608.03893)의 Retention 과 같다: 전환된 Base+ 점수 ÷ Base+ 로 전부 본 점수(Full Replay) × 100.
J&F 회복률과 J 회복률 둘 다 낸다.
1. 영상마다 점수 하나로 모은다 (그 영상 객체·전환 시점 줄들의 평균)
2. 영상마다 방법 ÷ Full Replay × 100 (Full Replay 가 0점인 영상은 뺌)
3. 영상들의 비율을 평균

### 격차 회복률(%)

(방법 − Source-only) ÷ (Full Replay − Source-only) × 100. C2C 논문(arXiv 2510.03215)의 PGR 과 같다.
0% = Small 이 계속 돌린 것과 같음, 100% = Full Replay 와 같음.
**영상 평균 점수로 한 번만 나눈다** (LLM 논문도 벤치마크 전체 점수로 한 번 계산).
영상마다 나누면 Source-only 와 Full Replay 가 거의 같은 영상에서 분모가 0 에 가까워 값이 튄다.
Full Replay 가 Source-only 보다 높을 때만 낸다 (낮거나 같으면 표에 "-").

### 회복률 범위 (논문 보고)

논문의 주 지표는 J&F 회복률과 J&F 격차 회복률이다. 평가 세트에서는 PUMaVOS 만 쓴다
(VOST 는 F 를 쓰지 않고, M3VOS 도 F 로 평가하지 않음).
J 회복률·J 격차 회복률과 VOST·M3VOS 의 값도 코드가 계산해 결과 표에 보고하지만, 논문에는 안 쓸 수 있다.

### 시간(초)

결과 줄 열 `seconds`. 전환 뒤 Base+ 가 한 일 전부에 걸린 시간 = 준비(프롬프트·기억 넣기, 기억 칸으로 바꾸기) + Base+ 가 추적한 모든 프레임.
표는 영상 평균.

| 비교군 | 들어가는 것 |
|---|---|
| Full Replay | Base+ 로 처음 ~ 끝 전체 (처음 정답 넣기 + 처음 ~ 끝 추적. 전환 시점과 무관) |
| Original+Replay-K | 처음 정답 넣기 + s-K+1 ~ s 다시 보기 + s+1 ~ 끝 |
| Direct State Copy | Small 기억을 Base+ 에 넣기 + s+1 ~ 끝 |
| Original-Prompt(s), Last-Visible, Original+Last-Visible | 마스크 프롬프트 넣기 + 기억 칸으로 바꾸기 + s+1 ~ 끝 |
| 본 모델 | translator + s+1 ~ 끝 (같은 식으로 잰다) |
| Source-only | 전환 없음 → "-" |

Small 에서 기억을 꺼내는 시간, 모델 로딩·Base+ 세션 만들기는 넣지 않는다 (두 모델이 미리 올라가 있다고 봄).

### 뺀 것

전환 지연·속도 배수(지표 정의서에서 뺌), 다시 본 프레임 수(본 모델은 항상 0장), 기억 닮음(프롬프트로 넘기는 비교군에는 쓸 수 없음), 기억 부분별 효과(비용이 큼).
비교군 Moment-Matched Copy 는 뺐다.

## 보고 방식

VOS 벤치마크 관례대로 평균 숫자 하나만 보고한다 (신뢰구간·부트스트랩 없음).
표: `outputs/tables/main.md` (주), `extra.md` (보조 — `docs/EXTRAS.md`).
