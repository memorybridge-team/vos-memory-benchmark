# vos-memory-benchmark

SAM2 Small → Base+ 기억 넘기기 평가. 설정·주요 평가 지표는 `docs/PROTOCOL.md`, 보조 평가 지표는 `docs/EXTRAS.md`.

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

본 모델과 비교군 따로 (예: Pod 두 개에서 동시에):

```bash
python scripts/2_evaluate.py --dataset lvos_v2_valid --methods model       # → lvos_v2_valid.model.jsonl
python scripts/2_evaluate.py --dataset lvos_v2_valid --methods baselines   # → lvos_v2_valid.baselines.jsonl
```

Full Replay·Source-only 는 회복률의 기준이라 둘 다 낸다. 이어하기는 (영상, 객체, 방법) 단위라 이미 있는 방법은 건너뛰고,
두 실행이 같은 Full Replay·Source-only 줄을 썼으면 표는 한 번만 센다.
시간 열을 비교하려면 두 실행을 같은 종류의 GPU 에서 돌린다.

표: `outputs/tables/main.md` (주요 평가 지표), `extra.md` (보조 평가 지표)

## 테스트 (GPU·데이터 없이, 가짜 모델·가짜 데이터)

```bash
python tests/test_end_to_end.py
```
