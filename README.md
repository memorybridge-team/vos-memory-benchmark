# vos-memory-benchmark

SAM2 Small → Base+ 기억 넘기기 평가. 설정 설명은 `docs/PROTOCOL.md`, 추가 항목은 `docs/EXTRAS.md`.

## 준비

1. SAM2 설치 (github.com/facebookresearch/sam2), `pip install numpy pillow opencv-python`
2. `settings.py` 에 `DATA_ROOT`, `DATA_FOLDERS`, `SAM2_CHECKPOINT_DIR` 적기
3. 데이터가 읽히는지 확인 (영상 1개씩 프레임 수·객체 수·무시 영역 값 출력)

```bash
python -m benchmark.data.mosev2
python -m benchmark.data.lvos_v2
python -m benchmark.data.vost
python -m benchmark.data.m3vos
python -m benchmark.data.pumavos
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

표: `outputs/tables/main.md` (주), `extra.md` (추가), `switch_b.md` (전환 B)

## 테스트 (GPU·데이터 없이, 가짜 모델·가짜 데이터)

```bash
python tests/test_end_to_end.py
```
