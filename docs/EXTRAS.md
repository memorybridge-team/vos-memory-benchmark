# extra (보조 평가 지표)

주 표(`main.md`)에는 넣지 않고 추가 표(`outputs/tables/extra.md`)에만 나오는 것과 그 이유.
코드: `evaluation/scoring/extra_metrics.py`, `extra_groups.py`, `baseline/extra_diagnostic.py`, 표는 `evaluation/tables/extra_tables.py`.
결과 줄 열 이름은 모두 `extra_` 로 시작.

MOSEv2 valid 는 정답이 첫 프레임뿐이라 **비용 세부·출력 일치도·진단 비교군의 비용 열만** 있다.

## 비용 세부

| 열 | 무엇 |
|---|---|
| seconds_per_frame_after | 전환 뒤 프레임 한 장을 처리하는 데 걸린 평균 시간(초) |
| peak_vram_mb | 그 실행 동안 GPU 메모리를 가장 많이 쓴 순간의 양(MB) |

## 실패 분석 (전환 뒤, 정답에 객체가 보이는 프레임)

| 열 | 무엇 | 이유 |
|---|---|---|
| extra_id_switch_rate | ID 뒤바뀜: 예측이 자기 물체보다 다른 물체와 더 많이 겹친 프레임 비율. 겹침 = IoU (겹친 넓이 ÷ 합친 넓이) | 기억이 틀린 물체에 붙는 실패는 J 만으로 구분 안 됨 |
| extra_failure | 실패: 평균 J 가 `EXTRA_FAILURE_J`(0.1) 보다 낮으면 1 | 완전히 놓친 경우가 몇 번인지 |

## 언제 잘 되고 언제 안 되는지

### drift 곡선

전환 뒤 경과 프레임 구간(`EXTRA_DRIFT_BINS`: 1~10, 11~50, 51~100, 101~300, 301~)마다
(방법 J&F − 같은 전환의 Full Replay J&F) 를 구해 영상 평균 → 표 + `drift_<데이터셋>.png`.
시간이 지나면서 Full Replay 와의 차이가 줄어드는지, 그대로인지, 커지는지 본다.
결과 줄에는 구간별 J&F 목록(`extra_drift_jf`)만 남기고, 빼기는 표를 만들 때 한다.

### 조건별 분류 (데이터셋 공식 라벨)

데이터셋 제작자가 붙인 라벨마다 회복률(J&F, J)을 따로 계산한다. 데이터셋별 표 + 뜻이 같은 라벨을 합친 표.

| 데이터셋 | 라벨 | 단위 | 어디서 |
|---|---|---|---|
| LVOS v2 | 영상 속성 13종 (OCC 가림, FM 빠른 움직임, DEF 모양 변형 ...) | 영상 | `<split>/*attribute*.json` — **배포되는지 미확인** (아래) |
| M3VOS | 상태 변화 종류 (`상태 변화:separate`), 변하기 전→후 (`상태:solid→liquid`) | 객체 | `meta/all_phase_transition.json` |
| VOST | 동작 (`변형:break`) | 영상 | 라벨 파일이 없어 공식 영상 이름 `<번호>_<동작>_<물체>` 에서 꺼냄 |
| MOSEv2, PUMaVOS | 없음 → 제외 | | |

- 라벨은 `1_make_video_list.py` 가 목록의 객체마다 `extra_labels` 로 적어 둔다.
- 합치는 규칙: `evaluation/scoring/extra_groups.py` 의 `SAME_AS`. 지금은 LVOS `DEF 모양 변형` + VOST 전부 + M3VOS 전부 → `모양·상태 변화` 하나. 새로 합칠 라벨은 여기에 한 줄씩 적는다.
- 합친 표는 평가 데이터셋만 쓴다 (train 은 뺌).
- 주의: 라벨은 영상·객체 전체에 붙어 있어서, 그 일이 **전환 뒤에** 일어났는지는 모른다.

#### ※ 미확인: LVOS v2 속성 파일 (2026-09-29)

- 있다고 볼 근거: LVOS 논문 Table II 에 영상마다 속성 13종을 붙였다고 나오고, README 에 `x_meta_attribute.json` 모양이 설명돼 있다.
- 없을 수 있는 근거: 공식 meta 다운로드 폴더에는 `train_meta.json`, `valid_meta.json`, `test_meta.json` 세 개뿐이다.
- 확인 방법: LVOS v2 를 서버에 받은 뒤 `find <LVOSv2 폴더> -iname "*attribute*"`
- 없으면: 코드는 멈추지 않는다. LVOS 라벨 표는 "(없음)" 으로 나오고, 합친 표에서 LVOS 몫이 빠진다.
- 있는데 모양이 README 와 다르면: `evaluation/data/lvos_v2.py` 의 `load_labels()` 를 고친다.

### 입력 길이별 분류

전환 전 Small 이 본 프레임 수(전환 프레임 − 시작 + 1)를 구간(`EXTRA_INPUT_LENGTH_BINS`: 1~50, 51~200, 201~500, 501~)으로 나누고
구간마다 회복률(J&F, J). 오래 쌓인 기억일수록 넘기기가 어려워지는지 본다.

## 결과 분석: 출력 일치도

| 열 | 무엇 |
|---|---|
| extra_agreement | 전환 뒤 프레임마다 방법 마스크와 Full Replay 마스크의 IoU → 평균 |

drift 는 정답과 비교하고, 이것은 Full Replay 와 비교한다 → 정답이 필요 없어 **MOSEv2 valid 에서도** 계산한다.
Full Replay 마스크는 평가 중 압축해(`np.packbits`) 들고 있다.

## 진단 비교군

실제로 쓰지는 않을 방법이지만, 다른 비교군 점수가 왜 그렇게 나왔는지 알아보는 용도. 주 표와 같은 열.

| 이름 | 무엇 | 비교 상대 |
|---|---|---|
| reset | Base+ 에게 아무것도 넘기지 않음 → 전환 뒤 전부 빈 마스크 | 바닥 점수 |
| recent_k_only | 처음 정답 마스크 없이, Small 마스크 한 장(s−K+1)에서 시작해 전환 직전 최근 K 프레임만 다시 봄 (`EXTRA_RECENT_K`) | Original+Replay-K (처음 정답을 줌) — 같은 표에 나란히 |
