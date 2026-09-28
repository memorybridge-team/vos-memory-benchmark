# 2026-09-23 · Task 03 loader validation

DAVIS 2017, MOSEv2, LVOS v2의 동결 manifest를 실제 RGB·mask·prompt 입력과 대조한 실행 기록이다.

| 분할 | manifest cases | unique prompt inputs | failures |
|---|---:|---:|---:|
| DAVIS train | 599 | 143 | 0 |
| DAVIS val | 249 | 61 | 0 |
| MOSEv2 train | 20,841 | 7,087 | 0 |
| MOSEv2 valid | 1,720 | 574 | 0 |
| LVOS v2 train | 1,803 | 601 | 0 |
| LVOS v2 valid | 714 | 238 | 0 |

원본 검증 결과는 같은 폴더의 데이터셋·분할별 JSON에 보존한다. 여섯 분할 모두 `failure_count=0`이다.
