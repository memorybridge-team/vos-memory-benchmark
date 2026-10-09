"""평가: 비교군을 돌려 전환 뒤 점수를 매기고 표로 만든다.

    data/            데이터셋 읽기
    switches.py      25/50/75% 전환과 최소 8프레임 자격
    evaluate_video.py 영상 하나 × 전환 시점 × 비교군
    cost.py          시간·GPU 메모리
    store.py         SQLite 원점수·재개·집계 저장
    records.py       현재 평가 필터·분석 JSONL 내보내기
    scoring/         [주] / [추가] 지표
    tables/          main.md / extra.md
"""
