# vos-memory-benchmark

SAM2 Small → Base+ 기억 넘기기 평가. 설정·주요 평가 지표는 `docs/PROTOCOL.md`, 보조 평가 지표는 `docs/EXTRAS.md`.

비교군은 Source-only, Full Replay (Base+ Native), Direct State Copy, Original + Last-Visible, Original-Prompt(s)+Replay-8의 5개다. 본 모델(translator)을 포함해 총 6개 방법을 평가한다.

## 폴더

| 폴더 | 무엇 |
|---|---|
| `model/` | SAM2 켜기·추적·기억 꺼내기/넣기 (SAM2 내부를 건드리는 곳은 여기뿐) |
| `baseline/` | 비교군: 전환 때 Base+ 에 무엇을 넘기나 (기억 상자 포함) |
| `translator/` | 본 모델: Small 기억 칸을 팀 translator 로 바꿔 넘긴다 (`settings.TRANSLATOR_*` 에 전달본 위치) |
| `evaluation/` | 평가: 데이터셋·전환 시점·영상 하나 평가·비용·결과 저장, `scoring/`(지표), `tables/`(표) |
| `scripts/` | 사람이 실행하는 것 (번호 = 순서) |
| `settings.py` | 모든 숫자·경로 |

## 준비

1. SAM2 설치 (github.com/facebookresearch/sam2), `pip install numpy pillow opencv-python`
2. `settings.py` 에 `DATA_ROOT`, `DATA_FOLDERS`, `SAM2_CHECKPOINT_DIR` 확인 (지금 값은 RunPod 서버 위치)
3. 데이터가 읽히는지 확인 (영상 1개씩 프레임 수·객체 수·무시 영역 값 출력)

```bash
python -m evaluation.data.lvos_v2
python -m evaluation.data.vost
python -m evaluation.data.m3vos
python -m evaluation.data.pumavos
```

## 실행 순서

```bash
python scripts/0_check_sam2.py
python scripts/1_make_video_list.py
```

개발 중:

```bash
python scripts/2_evaluate.py --dataset lvos_v2_valid --max-videos 5
python scripts/3_make_tables.py
```

검증·평가:

```bash
python scripts/2_evaluate.py --dataset lvos_v2_valid
python scripts/2_evaluate.py --dataset vost_val
python scripts/2_evaluate.py --dataset m3vos
python scripts/2_evaluate.py --dataset pumavos
python scripts/3_make_tables.py
```

GPU 2개 (데이터셋마다 영상을 반씩 나눠 동시에, 데이터셋 순서는 위와 같음):

```bash
mkdir -p outputs/logs
for ds in lvos_v2_valid vost_val m3vos pumavos; do
  CUDA_VISIBLE_DEVICES=0 python scripts/2_evaluate.py --dataset $ds --shard 0/2 > outputs/logs/${ds}_0.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python scripts/2_evaluate.py --dataset $ds --shard 1/2 > outputs/logs/${ds}_1.log 2>&1 &
  wait
done
python scripts/3_make_tables.py
```

결과는 `outputs/records/<데이터셋>.shard0of2.jsonl`, `shard1of2.jsonl` 로 따로 저장되고 표 만들 때 합쳐진다.
GPU 수를 중간에 바꿔도 된다 (1개 → 2개 등): 끝난 (영상, 객체, 방법) 은 그 데이터셋의 결과 파일 전부(`<데이터셋>*.jsonl`)에서 찾아 건너뛴다.
단, 같은 방법들을 두 실행(`--shard` 없이 / 있게)으로 **동시에** 돌리지는 않는다 (같은 객체를 둘 다 맡아 시간만 버림).

본 모델과 비교군은 한 번에 돈다. Full Replay와 Source-only는 객체마다 한 번만 계산한다.
예전에 따로 돌린 `<데이터셋>.model.jsonl` · `.baselines.jsonl` 이 있으면 끝난 (영상, 객체, 방법) 은 건너뛰고,
같은 Full Replay·Source-only 줄이 두 파일에 있으면 표는 한 번만 센다.
전환 시점은 객체 최초 등장부터 영상 끝까지의 50%·75%다.
지표는 J·J&F, 전환시간·GPU 메모리, 난이도 유형별 성능과 실패비율이다. 기존 회복률은 유지하며 복원율은 추후 논의한다.
경과 프레임별 J·J&F를 `outputs/tables/temporal.csv`로 보고한다.
전환시간은 준비 및 s까지의 replay만 포함하며 s+1 이후 추적은 제외한다.

평가 기준이 바뀌어 기존 목록과 결과는 재사용하지 않는다. `python scripts/1_make_video_list.py`로 목록을 다시 만든 뒤 평가한다.
기존 결과 파일은 보존하며 새 평가 버전의 결과만 이어하기와 표에서 사용한다.
시간 열을 비교하려면 모든 실행을 같은 종류의 GPU 에서 돌린다.

결과: `outputs/tables/main.md` (전환별 평가), `extra.md` (난이도별 성능), `temporal.csv` (경과 프레임별 점수)

## 테스트 (GPU·데이터 없이, 가짜 모델·가짜 데이터)

```bash
python tests/test_metrics.py
python tests/test_end_to_end.py
```
