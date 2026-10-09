"""[추가] 묶어서 보기: 데이터셋 공식 라벨.

공식 라벨
  data/<데이터셋>.py 의 load_labels() 가 읽고, 1_make_video_list.py 가 목록의 객체마다
  "extra_labels" 로 적어 둔다. 추론 결과에 라벨을 고정하고 표는 결과의 라벨을 사용한다.
      lvos_v2  영상 속성 13종 (예: "OCC 가림")                       영상 단위  (따로 받은 valid/val_meta_attribute.json)
      m3vos    상태 변화 종류 / 변하기 전→후 (예: "상태 변화:separate")  객체 단위
      vost     영상 이름의 동작 (예: "변형:break")                    영상 단위
      pumavos  없음
  주의: 라벨은 영상·객체 전체에 붙어 있다. "전환 뒤에" 그 일이 일어났는지는 모른다.

  같은 뜻 합치기: SAME_AS 에 "<데이터셋 파일 이름>/<라벨>" 또는 "<데이터셋 파일 이름>/<':' 앞부분>"
  → 공통 이름을 적는다. 여기 적힌 것만 합친 표에 들어간다.
"""

from __future__ import annotations

from evaluation.data import DATASETS

SAME_AS = {
    "lvos_v2/OCC 가림": "가려짐",
    "lvos_v2/CTC 닮은 물체 번갈아 등장": "비슷한 객체",
    "lvos_v2/DEF 모양 변형": "모양·상태 변화",   # LVOS 영상 속성 중 모양 변형만
    "vost/변형": "모양·상태 변화",          # VOST 영상은 모두 물체가 변형되는 영상
    "m3vos/상태 변화": "모양·상태 변화",    # M3VOS 객체는 모두 상태 변화 종류가 붙어 있음
}


def common_name(dataset: str, label: str) -> str | None:
    """데이터셋 라벨 → 합친 이름 (합치지 않는 라벨이면 None)."""
    base = DATASETS[dataset][0]
    return SAME_AS.get(f"{base}/{label}") or SAME_AS.get(f"{base}/{label.split(':')[0]}")
