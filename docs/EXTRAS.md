# extra (보조 평가 지표)

주 표(`main.md`)에는 넣지 않고 추가 표(`outputs/tables/extra.md`)에만 나오는 것과 그 이유.
코드: `evaluation/scoring/extra_metrics.py`, `extra_groups.py`, `evaluation/cost.py`(GPU 메모리), `baseline/extra_diagnostic.py`, 표는 `evaluation/tables/extra_tables.py`.
결과 줄 열 이름은 모두 `extra_` 로 시작.

## 비용 세부: 전환 GPU 메모리

| 열 | 무엇 | 이유 |
|---|---|---|
| extra_switch_gpu_mb | 전환 구간 GPU 메모리 최고치 − 전환 직전 GPU 사용량 (MB) | 넘기는 방법(translator) 때문에 더 큰 GPU 가 필요해지는지 |

- 전환 구간 = Small 이 s 까지 처리한 뒤 ~ Base+ 가 s+1 을 처리할 준비가 끝날 때 (바꿔 넣기 + 다시 보기)
- 전환 직전 = Base+ 세션을 연 직후 (켜 둔 두 모델·세션 무게는 여기 들어 있어 빼진다)
- Full Replay 는 처음 ~ s 를 다시 보는 동안의 최고치. Source-only·reset 은 없음 ("-"), GPU 가 없으면 없음

## 실패 분석 (전환 뒤, 정답에 객체가 보이는 프레임)

| 열 | 무엇 | 이유 |
|---|---|---|
| extra_failure_rate | 실패 비율 = 1 − J_Recall, J_Recall = (J > `EXTRA_RECALL_J`(0.5) 인 프레임 수) ÷ (대상 프레임 수) | 전환 뒤 실패한 프레임이 얼마나 되는지 |

줄(영상 × 객체 × 전환 시점)마다 계산 → 표는 영상 평균.

## 조건별 분류 (데이터셋 공식 라벨)

데이터셋 제작자가 붙인 라벨마다 회복률(J&F, J)을 따로 계산한다. 데이터셋별 표 + 뜻이 같은 라벨을 합친 표.

| 데이터셋 | 라벨 | 단위 | 어디서 |
|---|---|---|---|
| LVOS v2 | 영상 속성 13종 (OCC 가림, FM 빠른 움직임, DEF 모양 변형 ...) | 영상 | `valid/val_meta_attribute.json` — 영상 zip 에 없어 따로 받음 (아래) |
| M3VOS | 상태 변화 종류 (`상태 변화:separate`), 변하기 전→후 (`상태:solid→liquid`) | 객체 | `meta/all_phase_transition.json` |
| VOST | 동작 (`변형:break`) | 영상 | 라벨 파일이 없어 공식 영상 이름 `<번호>_<동작>_<물체>` 에서 꺼냄 |
| PUMaVOS | 없음 → 제외 | | |

- 라벨은 `1_make_video_list.py` 가 목록의 객체마다 `extra_labels` 로 적어 둔다.
- 합치는 규칙: `evaluation/scoring/extra_groups.py` 의 `SAME_AS`. 지금은 LVOS `DEF 모양 변형` + VOST 전부 + M3VOS 전부 → `모양·상태 변화` 하나. 새로 합칠 라벨은 여기에 한 줄씩 적는다.
- 주의: 라벨은 영상·객체 전체에 붙어 있어서, 그 일이 **전환 뒤에** 일어났는지는 모른다.

#### LVOS v2 속성 파일 (2026-10-05 확인)

- 논문 (arXiv 2404.19326) Table II 에 13종 정의, "we label each sequence with 13 challenges".
- 영상 zip 과 기본 meta json (`meta.json`, `val_meta.json`) 에는 없다 — 둘 다 객체마다 `frame_range` 만 있음 (서버 확인).
- 공식 홈페이지 dataset 페이지의 "Jsons with attributes" (Google Drive 폴더) 에 따로 있다: `val_meta_attribute.json` (116KB). 같은 폴더에 train / test / vt 도 있다.
- 받아서 서버 `<LVOSv2>/extracted/valid/` 에 넣었다 (2026-10-05). `load_labels()` 가 `valid/*attribute*.json` 을 읽는다.
- 모양은 README 와 같다: 맨 위 `sets`·`attributes`(13종 약자)·`videos`, 영상마다 `attributes` 약자 목록 (예: `0tCWPOrc` → BC, LR, SV, DB, SC, AC).
- 안 넣으면: 코드는 멈추지 않는다. LVOS 라벨 표는 "(없음)" 으로 나오고, 합친 표에서 LVOS 몫이 빠진다.
- 받은 파일 모양이 README 와 다르면: `evaluation/data/lvos_v2.py` 의 `load_labels()` 를 고친다.

## 결과 분석: 출력 일치도

| 열 | 무엇 |
|---|---|
| extra_agreement | 전환 뒤 모든 프레임마다 방법 마스크와 Full Replay 마스크의 IoU → 평균 |

정답이 아니라 Full Replay 와 비교한다 (정답이 필요 없음).
Full Replay 마스크는 평가 중 압축해(`np.packbits`) 들고 있다.

## 진단 비교군

실제로 쓰지는 않을 방법이지만, 다른 비교군 점수가 왜 그렇게 나왔는지 알아보는 용도. 주 표와 같은 열.

| 이름 | 무엇 | 비교 상대 |
|---|---|---|
| reset | Base+ 에게 아무것도 넘기지 않음 → 전환 뒤 전부 빈 마스크 | 바닥 점수 |
| recent_k_only | 처음 정답 마스크 없이, Small 마스크 한 장(s−K+1)에서 시작해 전환 직전 최근 K 프레임만 다시 봄 (`EXTRA_RECENT_K`) | Original+Replay-K (처음 정답을 줌) — 같은 표에 나란히 |

## 뺀 것

지표 정의서에서 뺀 것: 전환 뒤 프레임당 시간, 틀린 픽셀 분류, drift 곡선, 입력 길이별 분류. 정의서에 없는 ID 뒤바뀜 비율도 뺐다.
