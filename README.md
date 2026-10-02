# vos-memory-benchmark

SAM2 Small → Base+ 기억 넘기기 평가. 설정·주요 평가 지표는 `docs/PROTOCOL.md`, 보조 평가 지표는 `docs/EXTRAS.md`.

## 폴더

| 폴더 | 무엇 |
|---|---|
| `model/` | SAM2 켜기·추적·기억 꺼내기/넣기 (SAM2 내부를 건드리는 곳은 여기뿐) |
| `baseline/` | 비교군: 전환 때 Base+ 에 무엇을 넘기나 (기억 상자, Moment-Matched 통계 포함) |
| `evaluation/` | 평가: 데이터셋·전환 시점·영상 하나 평가·비용·결과 저장, `scoring/`(지표), `tables/`(표) |
| `scripts/` | 사람이 실행하는 것 (번호 = 순서) |
| `settings.py` | 모든 숫자·경로 |

## 준비

1. SAM2 설치 (github.com/facebookresearch/sam2), `pip install numpy pillow opencv-python matplotlib`
2. `settings.py` 에 `DATA_ROOT`, `DATA_FOLDERS`, `SAM2_CHECKPOINT_DIR` 적기
3. 데이터가 읽히는지 확인 (영상 1개씩 프레임 수·객체 수·무시 영역 값 출력)

```bash
python -m evaluation.data.mosev2
python -m evaluation.data.lvos_v2
python -m evaluation.data.vost
python -m evaluation.data.m3vos
python -m evaluation.data.pumavos
```

## 실행 순서

```bash
python scripts/0_check_sam2.py
python scripts/1_make_video_list.py
python scripts/2_fit_moment_stats.py
```

개발 중:

```bash
python scripts/3_evaluate.py --dataset lvos_v2_train --part dev --max-videos 5
python scripts/5_make_tables.py
```

최종:

```bash
python scripts/3_evaluate.py --dataset lvos_v2_valid
python scripts/3_evaluate.py --dataset vost_val
python scripts/3_evaluate.py --dataset m3vos
python scripts/3_evaluate.py --dataset pumavos
python scripts/3_evaluate.py --dataset mosev2_valid
```

`outputs/mosev2/submissions/*.zip` 을 서버에 직접 제출 → 점수를 CSV 로 정리 (모양은 `scripts/4_mosev2_scores.py` 맨 위) →

```bash
python scripts/4_mosev2_scores.py --csv mosev2_scores.csv
python scripts/5_make_tables.py
```

표: `outputs/tables/main.md` (주요 평가 지표), `extra.md` (보조 평가 지표), `drift_<데이터셋>.png`

## 테스트 (GPU·데이터 없이, 가짜 모델·가짜 데이터)

```bash
python tests/test_end_to_end.py
```
