# extra
"extra" 목록과 이유

## 전환 B — 재등장 직전

| 무엇 | 이유 |
|---|---|
| 객체가 `SWITCH_B_MIN_ABSENT` 정답 프레임 이상 안 보이다가 다시 나타나기 바로 전 프레임에서 전환. 객체마다 안 보인 기간이 가장 긴 것 `SWITCH_B_PER_OBJECT` 개 | 기억이 가장 중요한 순간. 넘기자마자 "아까 그 물체"를 다시 찾아야 해서 방법 차이가 가장 크게 드러난다 |

## 진단 비교군

:실제로 쓰지는 않을 방법이지만, 다른 베이스라인에서 점수가 왜 그렇게 나왔는지 알아보는 용도

| 이름 | 무엇 | 이유 |
|---|---|---|
| reset | 아무것도 안 넘김 → 전환 뒤 빈 마스크 | 바닥 점수 확인 |
| last_mask | 전환 프레임의 Small 마스크만 (보이든 안 보이든) | Last-Visible 과 비교: "보이는 프레임을 골라 주는 것"의 효과 |
| recent_k_only | 처음 정답 전달 없이, Small 마스크 한 장(s−K+1)에서 시작해 최근 K 프레임만 다시 봄 (`EXTRA_RECENT_K`) | Original+Replay-K 와 비교: "처음 정답"의 효과 |

## 추가 지표 (전환 뒤 구간)

| 열 | 무엇 | 이유 |
|---|---|---|
| extra_jf_at_n | 전환 뒤 처음 보이는 N 프레임의 J&F | 넘긴 직후 충격은 전체 평균에 묻힌다 |
| extra_switch_shock | 전환 직전 보이는 N 프레임(Small)의 J&F | 넘기는 순간 점수가 떨어지는지 오르는지 |
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

## 공식 라벨별 분류 (데이터셋이 준 라벨)

위 네 가지는 정답 그림으로 계산한 것이고, 이것은 데이터셋 제작자가 붙인 라벨을 그대로 쓴다.
데이터셋마다 라벨이 달라서 **데이터셋별 표**를 따로 만들고, 뜻이 같은 것은 **합친 표**에 한 줄로 모은다.

| 데이터셋 | 라벨 | 단위 | 어디서 |
|---|---|---|---|
| LVOS v2 | 영상 속성 13종 (OCC 가림, FM 빠른 움직임, DEF 모양 변형 ...) | 영상 | `<split>/*attribute*.json`. 논문에는 영상마다 붙였다고 나오고 README 에 파일 모양도 있지만, 공식 meta 다운로드 폴더에는 train/valid/test_meta.json 뿐이라 **배포되는지 미확인**. 없으면 LVOS 라벨 표는 비어서 나옴 |
| M3VOS | 상태 변화 종류 (`상태 변화:separate`), 변하기 전→후 (`상태:solid→liquid`) | 객체 | `meta/all_phase_transition.json` |
| VOST | 동작 (`변형:break`) | 영상 | 라벨 파일이 없어 공식 영상 이름 `<번호>_<동작>_<물체>` 에서 꺼냄 |
| MOSEv2, PUMaVOS | 없음 | | |

- 라벨은 `1_make_video_list.py` 가 목록의 객체마다 `extra_labels` 로 적어 둔다.
- 합치는 규칙: `benchmark/scoring/extra_labels.py` 의 `SAME_AS`. 지금은 LVOS `DEF 모양 변형` + VOST 전부 + M3VOS 전부 → `모양·상태 변화` 하나. 새로 합칠 라벨은 여기에 한 줄씩 적는다.
- 합친 표는 평가 데이터셋만 쓴다 (train 은 뺌).
- 주의: 라벨은 영상·객체 전체에 붙어 있어서, 그 일이 **전환 뒤에** 일어났는지는 모른다.

### ※ 미확인: LVOS v2 속성 파일 (2026-09-29)

LVOS v2 공식 라벨 파일이 실제로 있는지 아직 확인하지 못했다.

- 있다고 볼 근거: LVOS 논문 Table II 에 영상마다 속성 13종을 붙였다고 나오고, README 에 `x_meta_attribute.json` 모양이 설명돼 있다.
- 없을 수 있는 근거: README 가 안내하는 공식 meta 다운로드 폴더에는 `train_meta.json`, `valid_meta.json`, `test_meta.json` 세 개뿐이다 (객체 등장 구간만 담음). 영상 zip (`train.zip`, `valid.zip`) 안에 있는지는 못 봤다.
- 확인 방법: LVOS v2 를 서버에 받은 뒤 `find <LVOSv2 폴더> -iname "*attribute*"`
- 없으면: 코드는 멈추지 않는다. LVOS 공식 라벨 표는 "(라벨 없음)" 으로 나오고, 합친 표에서 LVOS 몫이 빠진다.
- 있는데 모양이 README 와 다르면: `benchmark/data/lvos_v2.py` 의 `load_labels()` 를 고친다.

**코드 돌리기 전에 꼭 확인하기**

모든 추가 항목은 정답이 모든 프레임에 있어야 해서 MOSEv2 valid 에는 없다.
