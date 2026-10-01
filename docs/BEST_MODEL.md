# Best model 고르기 (`best_model.py`)

## 하는 일

translator를 학습하면서 epoch마다 점수를 매기고, 점수가 가장 높은 epoch의 모델을 `best.pt`로 남긴다.

- 점수 = 전환 뒤 J&F 회복률을 25 / 50 / 75% 전환에서 각각 구한 뒤 평균한 값
- 회복률(%) = 방법의 점수 ÷ Full Replay의 점수 × 100
  - 영상마다 비율을 구한 뒤 영상들의 평균 (전체 평균끼리 나누지 않음)
  - Full Replay 점수가 0인 영상은 나눌 수 없어서 뺀다

**`best_model.py`는 계산만 한다.** 모델을 돌리거나 마스크를 채점하지 않는다.
이미 채점된 J&F 점수 두 묶음을 받아야 한다.

| 넘기는 것 | 무엇 | 언제 만드나 |
|---|---|---|
| `replay_rows` | Full Replay 점수 (100점 기준선) | 학습 시작 전에 한 번 |
| `method_rows` | 학습 중인 모델의 점수 | 매 epoch |

## 1. Full Replay 점수 — 반드시 먼저 돌려야 함

회복률의 분모라서 **이게 없으면 회복률을 못 구한다.**

1. Base+ 혼자 dev 영상을 처음부터 끝까지 돌린다 (전환 없음).
   객체마다 처음 보인 프레임에 정답 마스크를 주고 시작한다.
2. 25 / 50 / 75% 전환 지점마다 **전환 뒤 구간만** 채점해서 줄을 하나씩 만든다.
   (전환 지점이 다르면 채점하는 구간도 달라서 시점마다 따로 줄이 있다.)

학습과 상관없이 값이 바뀌지 않으므로 한 번 만들어 저장해 두고 매 epoch 다시 쓴다.

## 2. 학습 중인 모델 점수 — 매 epoch

dev 영상 × 객체 × 전환 시점(25 / 50 / 75%)마다:

1. Small이 객체가 처음 보인 프레임(정답 마스크)부터 전환 프레임 s까지 추적한다.
2. s에서 Small의 기억(memory bank)을 꺼낸다.
3. 학습 중인 translator로 그 기억을 Base+ 형식으로 바꾼다.
4. 바꾼 기억을 Base+에 넣고 s+1부터 영상 끝까지 추적한다.
5. 전환 뒤(s+1 ~ 끝)에서 **정답에 객체가 보이는 프레임만** J와 F를 계산한다. J&F = (J + F) ÷ 2

Full Replay와 **같은 영상, 같은 전환 프레임, 같은 채점 규칙**을 써야 한다.

## 점수 줄 형식

두 묶음 모두 아래 모양의 줄 목록이다.

```python
{"video": "abc", "fraction": 0.25, "jf": 0.81}
```

- `fraction`은 정확히 `0.25`, `0.50`, `0.75` 중 하나
- 객체가 여러 개면 객체마다 한 줄씩 넣는다 (영상별 평균은 `best_model.py`가 낸다)
- `jf`는 두 묶음이 같은 단위여야 한다 (둘 다 0~1이거나 둘 다 0~100)
- 두 묶음의 영상 목록이 다르면 `ValueError`로 멈춘다

## 학습 루프에 넣는 코드

```python
from best_model import BestModel

replay_rows = load_full_replay_scores()   # 1번에서 미리 만든 Full Replay 점수
best = BestModel("outputs/train_run1")

for epoch in range(n_epochs):
    train_one_epoch(model)
    method_rows = score_on_dev(model)     # 2번: 이번 epoch 모델로 dev 채점
    best.update(epoch, model, method_rows, replay_rows)
```

`load_full_replay_scores`, `score_on_dev`는 학습 코드 쪽에서 만들어야 하는 함수 이름(예시)이다.

## 나오는 파일 (`outputs/train_run1/`)

| 파일 | 내용 |
|---|---|
| `selection_log.jsonl` | epoch마다 한 줄: epoch, r25, r50, r75, score |
| `best.pt` | 점수가 가장 높은 epoch 모델의 `state_dict` |
| `best.json` | 그 epoch와 점수 |

점수가 같으면 먼저 나온 epoch가 best로 남는다.

## 실행 조건

- 학습 스크립트를 저장소 맨 위 폴더에서 실행해야 `from best_model import BestModel`이 된다.
- torch가 깔려 있어야 한다.
