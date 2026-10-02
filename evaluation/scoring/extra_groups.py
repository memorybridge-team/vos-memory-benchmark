"""[추가] 묶어서 보기: 데이터셋 공식 라벨, 입력 길이 구간, (drift 가 쓰는) 구간 이름.

공식 라벨
  data/<데이터셋>.py 의 load_labels() 가 읽고, 1_make_video_list.py 가 목록의 객체마다
  "extra_labels" 로 적어 둔다. 표(tables/extra_tables.py)는 목록에서 라벨을 꺼내 쓴다.
      lvos_v2  영상 속성 13종 (예: "OCC 가림")                       영상 단위  ※ 파일이 있는지 미확인 (data/lvos_v2.py)
      m3vos    상태 변화 종류 / 변하기 전→후 (예: "상태 변화:separate")  객체 단위
      vost     영상 이름의 동작 (예: "변형:break")                    영상 단위
      나머지   없음 (MOSEv2, PUMaVOS)
  주의: 라벨은 영상·객체 전체에 붙어 있다. "전환 뒤에" 그 일이 일어났는지는 모른다.

  같은 뜻 합치기: SAME_AS 에 "<데이터셋 파일 이름>/<라벨>" 또는 "<데이터셋 파일 이름>/<':' 앞부분>"
  → 공통 이름을 적는다. 여기 적힌 것만 합친 표에 들어간다.

입력 길이
  전환 전 Small 이 본 프레임 수 = 전환 프레임 − 시작 프레임 + 1. 구간은 settings.EXTRA_INPUT_LENGTH_BINS.
"""

from __future__ import annotations

import settings
from evaluation.data import DATASETS

SAME_AS = {
    "lvos_v2/DEF 모양 변형": "모양·상태 변화",   # LVOS 속성 파일이 없으면 이 줄은 아무 효과 없음 (미확인)
    "vost/변형": "모양·상태 변화",          # VOST 영상은 모두 물체가 변형되는 영상
    "m3vos/상태 변화": "모양·상태 변화",    # M3VOS 객체는 모두 상태 변화 종류가 붙어 있음
}


def common_name(dataset: str, label: str) -> str | None:
    """데이터셋 라벨 → 합친 이름 (합치지 않는 라벨이면 None)."""
    base = DATASETS[dataset][0]
    return SAME_AS.get(f"{base}/{label}") or SAME_AS.get(f"{base}/{label.split(':')[0]}")


def bin_names(starts: tuple) -> list[str]:
    """구간 시작점 (1, 11, 51) → ["1~10", "11~50", "51~"]."""
    ends = [f"{b - 1}" for b in starts[1:]] + [""]
    return [f"{a}~{b}" for a, b in zip(starts, ends)]


def bin_index(starts: tuple, value: int) -> int | None:
    """value 가 들어가는 구간 번호 (첫 시작점보다 작으면 None)."""
    inside = [i for i, a in enumerate(starts) if value >= a]
    return inside[-1] if inside else None


def input_length_bin(row: dict) -> str:
    """결과 줄 → 입력 길이 구간 이름."""
    bins = settings.EXTRA_INPUT_LENGTH_BINS
    return bin_names(bins)[bin_index(bins, row["switch_frame"] - row["start"] + 1)]
