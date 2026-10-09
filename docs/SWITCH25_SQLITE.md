# 25%·50%·75% 공통 평가와 SQLite

## 객체 전체 제외

전환은 객체 최초 등장 a부터 영상 끝 e까지의 25/50/75%에서 계산한다. 기존 round와 프레임 범위 제한은 유지한다.
세 전환 중 하나라도 `s-a <= 8`이면 그 영상의 해당 객체를 **모든 전환·방법·회차**에서 제외한다. 정확히 8도 제외한다.
다른 유효 객체는 유지한다. 제외 사유·조건·건수는 영상 목록과 DB의 documents에 저장한다.
새 목록은 세 전환에 동일한 객체 집합을 제공한다. 목록 생성, 모델 로딩 전 검증, 직접 evaluate_object 호출, 현재 결과 집계에 같은 규칙을 적용한다. Replay-8은 짧은 구간으로 축소하지 않는다.

평가 버전은 5, 목록 버전은 3이다. 이전 50/75% 목록은 다시 만든다. 이전 버전 원본은 보존하지만 새 결과와 섞지 않는다. 이전 Native에는 25% 기억 snapshot이 없으므로 새 프로토콜의 Native를 다시 실행한다.

## SQL 저장

Python의 sqlite3를 사용하며 별도 SQLite 서버가 필요 없다. 환경 확인:

```bash
python -c "import sqlite3; print(sqlite3.sqlite_version)"
```

기본 DB는 `OUTPUT_ROOT/benchmark.sqlite3`, 경로 변경은 `settings.DATABASE_PATH`다. 여러 GPU 프로세스가 같은 DB를 공유한다. DB는 파일 잠금이 정상 동작하는 로컬 디스크에 둔다.

| 테이블 | 저장값 |
| --- | --- |
| evaluations | 영상·객체·세 전환·방법·회차·seed·지표 요약·비용·실행 버전 |
| frame_scores | 전환 전후 원래 프레임별 J·F·J&F, 같은 회차 Native 원점수, 전체 프레임 메타데이터 |
| restoration_scores | 기억 프레임별 spatial/pointer R²·SSE·SST·평균·원소 수·shape·N/A 사유 |
| native_references / native_scores | 고정 Native 기준 ID·조건·비용과 프레임별 원점수 |
| native_memory | 모든 전환의 CPU 기억 tensor를 torch.save 형식의 BLOB으로 보존 |
| documents | 영상 목록·제외 사유, 최신 표/곡선 결과 |
| analysis_rows | median/mean 등 정의별 재집계 결과. 추론 원본과 분리 |
| legacy_imports | 이전 파일 이관 상태 |

프레임별 값은 전용 테이블의 숫자 열로 조회할 수 있다. 추가 metadata는 JSON payload로 함께 보존해 import/export에서 손실하지 않는다.
Native scalar와 tensor는 한 transaction으로 저장한다. 객체의 모든 방법 결과도 한 transaction으로 확정한다. 오류는 rollback하며 중복 조건은 고유 키로 막는다. Native 기준 ID가 충돌하면 조용히 덮어쓰지 않고 오류를 낸다.
다중 writer는 SQLite 잠금과 최대 60초 대기로 순차 commit한다. DB 쓰기는 전환시간·GPU 최고 메모리 측정 밖에서 수행한다. DB 한 파일에 큰 tensor도 들어가므로 실제 디스크 용량을 확인한다. 백업은 실행 종료 후 DB 파일을 보존하거나 실행 중이면 SQLite backup API를 사용한다.

이전 JSONL와 Native .pt는 자동 이관하며 원본 파일을 수정·삭제하지 않는다. 재이관은 중복을 만들지 않는다.

```bash
python scripts/4_import_jsonl.py
python scripts/4_import_jsonl.py --dataset pumavos
```

CSV/Markdown/JSONL/PNG/PDF는 공유용 export다. 분석 원본과 Native tensor는 DB에 있다. DB의 예전 평가 버전은 보존하고 새 집계에는 버전 5의 공통 유효 객체만 쓴다.

예시 SQL:

```sql
SELECT e.dataset,e.video,e.object_id,e.switch_name,e.baseline,e.run_id,
       f.frame,f.j,f.jf
FROM evaluations e JOIN frame_scores f ON f.evaluation_id=e.id
WHERE e.revision=5 AND f.phase='post';
```

## GPU 실행 성능과 측정

CPU worker 하나가 이미지 읽기·resize·정규화를 기본 두 프레임 앞서 준비한다. 큐는 제한되어 영상 전체를 RAM에 올리지 않는다. 모델에 입력되는 값·순서, 모델 정밀도 및 메모리 유지 길이는 그대로다. 세션 종료/예외 시 worker를 정리한다.
GPU의 모델 추론과 전환 작업은 프로세스 내에서 순차 실행한다. 다른 GPU에는 영상 shard를 나누어 배정한다. 같은 물리 GPU의 협력 평가 프로세스끼리는 OS 잠금으로 중복 실행을 차단한다. 외부 학습/다른 프로그램의 GPU 점유까지 차단하는 것은 아니므로 비용 측정 GPU를 전용으로 사용한다.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/2_evaluate.py --dataset pumavos --shard 0/2
CUDA_VISIBLE_DEVICES=1 python scripts/2_evaluate.py --dataset pumavos --shard 1/2
# 선행 준비 비활성화
python scripts/2_evaluate.py --dataset pumavos --frame-prefetch 0
```

측정은 동일한 선행 준비 설정 및 같은 GPU 종류에서 수행한다. 원본의 runtime_revision=3 및 frame_prefetch로 설정을 기록한다. 전환시간은 준비와 replay, GPU 메모리는 같은 구간의 allocated 최고치 차이를 사용한다. Small 기억 export, R² 계산, DB 저장, 이후 추적·채점은 제외한다.
CPU 선행 준비로 입력을 기다리는 시간을 줄인다. 실제 GPU 사용률·속도 개선량은 서버에서 측정해야 하며 CPU 테스트로 GPU 성능 개선 비율을 주장하지 않는다.

## 유지한 정의와 검증

감사 의견의 수정해야 하는 것 2번은 보류한다. 프레임별 Native 3회 중앙값, 구간 평균 점수 비율 회복률, 같은 회차 Native 기억 R² 및 기존 집계 방식은 변경하지 않는다.
9개 테스트 스크립트로 지표·회복률·R²·곡선·전체 평가 외에 <=8 전체 제외, SQL 조회/원자성/BLOB 복원/이관/동시 쓰기, CPU 선행 준비의 값·순서·정리와 GPU 잠금을 검증한다. 실제 SAM2 CUDA 추론은 이 로컬 검증에 포함되지 않는다.
