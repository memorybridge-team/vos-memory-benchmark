# Protocol (확정)

## 연구 주제

Video Segmentation을 실행하던 도중 작은 모델(SAM2 Small)에서 큰 모델(SAM2 Base+)로 전환할 때
**Small이 지금까지 쌓은 기억을 Base+ 에 얼마나 잘 넘기는지**를 측정한다.
주 측정 방식인 회복률은 (현재 방식으로 산출한 점수) / (Full Replay(Base+로 처음부터 끝까지 돌림)로 산출한 점수) * 100 으로 계산한다.

## Only Baseline: 본 모델은 포함x

- 비교군 10개와 extra 진단 비교군 2개로 구성 -> extra는 EXTRAS.md 참고
- 본 모델의 점수는 이 코드로 나오지 않으며, 결과 표(`outputs/tables/`)에 있는 것도 전부 베이스라인 점수다.
- 본 모델을 같은 조건에서 비교하려면 똑같은 영상 목록·전환 시점·지표·회복률 계산을 써야 한다.

## 데이터
baseline에서 학습·개발용 데이터는 Moment-Matched Copy에만 fit(통계용)이 존재한다.
| 용도 | 데이터 | 비고 |
|---|---|---|
| 학습·개발 (본모델) | MOSEv2 train + LVOS v2 train | fit(통계용) / dev(개발 확인용)으로 나눔 |
| 평가 | MOSEv2 valid + LVOS v2 valid | MOSEv2 valid 는 공개된 정답이 첫 프레임뿐 → 서버 채점 |
| 외부 평가 | M3VOS + PUMaVOS + VOST val | VOST는 무시 영역(255px)을 채점에서 뺌 |

## 평가 단위

영상 하나 × 객체 하나 × 전환 시점 하나 × 비교군 하나 = 결과 한 줄
각 객체가 처음 보인 프레임에서 정답 마스크를 주고 시작해 영상 끝까지 
**정답은 이 처음 프롬프트에만 들어간다.** 나머지 전환 때 넘기는 것은 모두 Small이 스스로 낸 것

## 전환 시점

전환 프레임 s = Small이 마지막으로 본 프레임. Base+ 는 s+1 부터.

객체 구간 [시작, 영상 끝]의 25 / 50 / 75% 지점 (`settings.SWITCH_FRACTIONS`). 전환 이름은 "25", "50", "75".

전환 전 구간(시작 ~ s)은 모든 비교군이 **같은 Small 결과**를 쓴다. 비교군끼리 다른 것은 전환 뒤뿐.

## 비교군 10개

| 비교군 | Base+ 가 받는 것 |
|---|---|
| Source-only | 안 넘김. Small이 끝까지 |
| Full Replay | 처음 정답, 그리고 처음부터 s 까지 전부 다시 봄 (100점 기준) |
| Direct State Copy | Small memory bank 그대로 |
| Moment-Matched Copy | Small memory bank을 Base+ 값 분포(평균·표준편차)에 맞춘 뒤 |
| Original-Prompt(s) | 처음 정답 마스크만 |
| Last-Visible | Small이 마지막으로 "보인다"고 한 프레임의 Small 마스크만 |
| Original+Last-Visible | 위 둘 다 |
| Original+Replay-4/8/16 | 처음 정답 + 전환 직전 K 프레임을 Base+ 가 다시 봄 |


Moment-Matched 통계는 train fit 영상에서 모델마다 채널별로 계산 (`scripts/2_fit_moment_stats.py`).

## 주요 평가 지표

### J, J&F

- **J** = 겹친 넓이 ÷ 합친 넓이
- **F** = 예측 테두리가 정답 테두리에 얼마나 가까운가 (대각선 × 0.008 픽셀 허용)
- **J&F** = 둘의 평균
- 무시 영역 픽셀(VOST)은 J, F 모두에서 뺀다

| 데이터셋 | 채점 구간 | 주 지표 |
|---|---|---|
| MOSEv2 valid | 영상 전체 — 서버 채점 (정답이 공개되지 않아 전환 뒤만 따로 볼 수 없음) | J&F |
| LVOS v2, PUMaVOS | 전환 뒤, 정답에 객체가 보이는 프레임만 | J&F |
| VOST, M3VOS | 전환 뒤, 정답에 객체가 보이는 프레임만 | **J** (물체가 변형돼 테두리 채점이 어려움) |

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

### 첫 결과까지 시간, 속도 배수

- 첫 결과까지 시간(초): 전환 순간부터 전환 뒤 첫 프레임 결과를 낼 때까지 (준비 + 다시 보기 포함)
- 속도 배수: 영상마다 Full Replay 첫 결과 시간 ÷ 방법 첫 결과 시간 → 영상 평균

### 뺀 것

다시 본 프레임 수(본 모델은 항상 0장), 기억 닮음(프롬프트로 넘기는 비교군에는 쓸 수 없음), 기억 부분별 효과(비용이 큼).

## 보고 방식

VOS 벤치마크 관례대로 평균 숫자 하나만 보고한다 (신뢰구간·부트스트랩 없음).
표: `outputs/tables/main.md` (주), `extra.md` (보조 — `docs/EXTRAS.md`).

## MOSEv2 valid 제출

Source-only 1번 + 나머지 주 비교군 9개 × 전환 시점 3개 = **28번**. [추가] 비교군은 제출하지 않는다.
객체들을 한 장으로 합칠 때 겹치면 번호 작은 객체가 이긴다 (SAM2 공식 평가 코드와 같은 규칙)
