# "추가" 목록과 이유

확정 표에 없는 것. 코드에는 남기되 표시해서 따로 모은다.

- 파일 이름: `extra_` 로 시작 (`methods/extra_diagnostic.py`, `scoring/extra_metrics.py`, `scoring/extra_strata.py`)
- 비교군: `role = "extra"`, 표 이름 앞에 `[추가]`
- 결과 열: `extra_` 로 시작
- 표: `outputs/tables/extra.md`, 전환 B 는 `outputs/tables/switch_b.md` (주 표 `main.md` 와 분리)

## 전환 B — 재등장 직전

| 무엇 | 이유 |
|---|---|
| 객체가 `SWITCH_B_MIN_ABSENT` 정답 프레임 이상 안 보이다가 다시 나타나기 바로 전 프레임에서 전환. 객체마다 안 보인 기간이 가장 긴 것 `SWITCH_B_PER_OBJECT` 개 | 기억이 가장 중요한 순간. 넘기자마자 "아까 그 물체"를 다시 찾아야 해서 방법 차이가 가장 크게 드러난다 |

## 진단 비교군

| 이름 | 무엇 | 이유 |
|---|---|---|
| reset | 아무것도 안 넘김 → 전환 뒤 빈 마스크 | 바닥 점수 확인 |
| last_mask | 전환 프레임의 Small 마스크만 (보이든 안 보이든) | Last-Visible 과 비교: "보이는 프레임을 골라 주는 것"의 효과 |
| recent_k_only | 처음 정답 없이, Small 마스크 한 장(s−K+1)에서 시작해 최근 K 프레임만 다시 봄 (`EXTRA_RECENT_K`) | Original+Replay-K 와 비교: "처음 정답"의 효과 |

## 추가 지표 (전환 뒤 구간)

| 열 | 무엇 | 이유 |
|---|---|---|
| extra_jf_at_n | 전환 뒤 처음 보이는 N 프레임의 J&F | 넘긴 직후 충격은 전체 평균에 묻힌다 |
| extra_switch_shock | 위 값 − 전환 직전 보이는 N 프레임(Small)의 J&F | 넘기는 순간 점수가 떨어지는지 오르는지 |
| extra_id_switch_rate | 예측이 자기보다 다른 객체와 더 겹친 프레임 비율 | 기억이 틀린 물체에 붙는 실패는 J 만으로 구분 안 됨 |
| extra_recovery_frames | 처음 보이는 프레임부터 J ≥ 0.5 까지 걸린 프레임 | 처음에 흔들려도 곧 따라잡는지 |
| extra_absent_false_alarm | 정답에 객체가 없는 프레임 중 무언가를 칠한 비율 | 영상 전체 기준에선 빈 프레임이 섞여 잘 안 보임 |
| extra_failure | 전환 뒤 평균 J < 0.1 이면 1 | 완전히 놓친 경우가 몇 번인지 |

## 조건별 분류 (정답만 보고, 전환 뒤 구간)

| 열 | 무엇 |
|---|---|
| extra_stratum_occlusion | 가림: 사라졌다 다시 나타남 |
| extra_stratum_crossing | 교차: 다른 객체와 상자가 `EXTRA_CROSSING_BOX_IOU` 이상 겹침 |
| extra_stratum_small | 작은 객체: 보이는 넓이 중앙값 < 화면의 `EXTRA_SMALL_AREA` |
| extra_stratum_fast | 빠른 움직임: 프레임당 중심 이동 중앙값 > 대각선의 `EXTRA_FAST_MOTION` |

이유: 평균 하나로는 "어떤 상황에서 기억 넘기기가 무너지는지" 알 수 없다.

모든 추가 항목은 정답이 모든 프레임에 있어야 해서 MOSEv2 valid 에는 없다.
