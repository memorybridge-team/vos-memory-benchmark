# 추가 최적화: 실행 버전 5

`feat/switch25-sqlite`에 다음 네 변경을 적용했다. 모델 가중치, 추론 정밀도·해상도, 기억 창, GPU 추론 순서, 전환 자격과 점수 집계 정의는 유지한다. `EVALUATION_RUNTIME_REVISION=5`와 새 설정을 실험 ID에 포함하여 이전 실행의 시간/결과를 재개 시 섞지 않는다. 방법 정의가 바뀌지 않았으므로 baseline revision은 올리지 않는다.

## 구현

| 기능 | 범위와 기본 설정 | 보존 조건 |
| --- | --- | --- |
| CPU RGB 입력 LRU | 객체/회차 실행 안에서 Small·Base+ 공유, `RGB_CACHE_MB=256` MiB | 세션 초기화·프롬프트 준비·측정 replay에서 읽기/저장 모두 우회. 정규화 계산은 동일하며 tensor는 독립 복사 |
| Native R² 기준 LRU | 같은 객체/회차의 변경하지 않는 Native snapshot, `RESTORATION_CACHE_MB=128` MiB | float64 입력·유한성·평균·중심화된 SST 재사용. SSE, R²와 상태/N/A 정의 유지 |
| presence scalar 생략 | evaluator가 명시적으로 `configure_benchmark()`를 호출한 Session만, `BENCHMARK_SKIP_VISIBLE=True` | `FrameOut.visible=None`으로 미측정 명시. 일반 Session은 bool 반환. SAM2 내부 presence logits와 마스크 기반 last-visible 선택 유지 |
| SQLite 연결 재사용 | `2_evaluate.py`의 평가 루프에서 연결 한 개 유지 | 결과별 SAVEPOINT/commit/rollback은 별도. 추론 동안 활성 트랜잭션과 쓰기 잠금 없음. scope 종료/오류 때 연결 닫음 |

RGB 키에는 파일의 실제 경로·크기·mtime/ctime, 전처리 크기·평균·표준편차가 들어간다. 파일 오류는 기존처럼 전파한다. 캐시가 부족하면 다시 계산하며, 긴 영상의 순차 접근에서는 적중률이 낮을 수 있다. 입력과 설정을 실행 중 변경하는 사용법은 지원하지 않는다.

Native 통계는 원본 tensor의 참조·수정 version으로 구분한다. inference tensor에는 version counter가 없으므로 evaluator가 만든 변경하지 않는 snapshot만 사용한다. tensor ID 재사용은 원본 참조를 유지하여 방지한다. 캐시는 객체/회차 밖으로 공유하지 않는다.

두 새 캐시는 해당 설정을 0으로 끌 수 있다. `BENCHMARK_SKIP_VISIBLE=False`는 benchmark에서도 기존 가시성 조회를 수행한다. 보관 상한은 worker 임시 배열·현재 입력/독립 복사·이미 보관 중인 Native snapshot을 포함한 프로세스 전체 RAM 상한이 아니다.

`cpu_profile.image_preparation`에 RGB 적중/누락/우회 수를, `restoration_cache_profile`에 결과별 Native 통계 적중/누락 및 보관량을 저장한다. 공유 추적 프로파일은 기존처럼 tracking_id로 중복을 제거한다.

## 검증

`tests/test_extra_optimizations.py`는 입력 tensor 소유권, 파일 변경, 크기별 키, 캐시 상한/비활성화, 프리페치와 측정 프레임 우회, 일반/benchmark 가시성의 의미, DB 연결 재사용과 닫기, 결과별 commit, 중첩 rollback, 실제 부모/자식 행 저장 실패와 재개를 검사한다. SQLite의 자동 전체 rollback에서 원래 오류를 보존하고, 다른 읽기 트랜잭션이 commit을 막아도 잠금을 해제한 뒤 재시도할 수 있음을 확인했다. SQLite 동시 초기화와 측정 구간의 worker 격리도 통과했다.

Native R²는 고정된 변경 전 commit `b05a6371fe95059343ce93c6eedaa147cc0b0209`의 함수를 독립 fixture로 보관하여 비교했다. Git 이력이 없어도 테스트할 수 있다. bf16/fp32/fp64, 비연속 tensor, 일치/불일치, 누락, 모양 불일치, 빈/단일 원소, 0분산, 비유한 값, 통계 overflow, 큰 offset의 작은 변동에서 모든 충분통계량과 상태가 정확히 일치했다. cache 0·작은 상한·128 MiB를 각각 비교하고 수정 version의 무효화도 확인했다.

`tests/test_cpu_reliability.py`는 새 캐시와 가시성 옵션을 끈 동기 기준과 세 병렬 설정을 비교한다. 2영상×2객체×3회차×25/50/75%×6방법, 설정당 216행×65프레임이다. 42,120개 프레임 기록의 원점수·복원율 충분통계량·결측·가시성이 정확히 일치하고, Python/NumPy/Torch RNG, 평균/중앙값 회복률, R²와 표/곡선의 유효 표본 수도 같다. 예측 실패 0점을 제외하거나 다른 회차로 대체하지 않는다. 고해상도 동시 J/F 계산 64쌍도 정확히 일치했다.

전체 테스트 스크립트 12개와 Python 구문 검사가 통과했다. 실제 SAM2 2.1 Small·Base+ 체크포인트로 DAVIS `motocross-jump` 0~19프레임을 비교했다. 변경 전 대비 기본 설정과 캐시 적중/worker 4 설정에서 각각 20/20 마스크·전체 기억 bank SHA256·J/F가 정확히 일치하고 픽셀 차이는 0이었다. 두 변경 설정×두 모델×20프레임 = 80쌍 비교이며, Small은 RGB 19회, Base+는 측정 밖의 미래 5회가 캐시에 적중했다. 모델 파라미터와 입력 파일 hash는 실행 전후 같다.

같은 진단 입력에서 실제 `0_check_sam2.py`의 검사 함수도 통과했다(기억 모양·dtype·pos_enc·roundtrip, cut=18/이후 1프레임). 이 CPU 진단은 서버 본평가 preflight를 대신하지 않는다. 실행은 CPU Torch 2.13.0 조건이다. 재현 스크립트·결과·소스 hash·테스트 로그는 Git에서 제외되는 `outputs/validation/extra_optimization_audit`에 보관했다. 이전 실행 버전 4의 검증 기록은 [CPU_RELIABILITY_AUDIT.md](CPU_RELIABILITY_AUDIT.md)다.

## 본평가와 성능 해석

서버에서 현재 코드의 `python scripts/0_check_sam2.py`를 먼저 통과해야 본평가를 실행한다. CUDA/BF16 추론 동등성과 실제 GPU 속도 향상은 아직 검증하지 않았다. CPU 설치에서는 SAM2 CUDA hole-fill 확장이 없어 해당 후처리를 생략하는 fallback을 사용한다.

전환 비용의 범위는 기존처럼 준비 + s까지의 replay다. RGB 캐시와 R² 계산은 그 타이머 밖에 있으며 채점 worker도 측정 중에는 없다. presence scalar를 생략하는 구현과 CPU 부하/클록 이력은 측정 시간의 숫자를 바꿀 수 있다. 이전 실행 버전과 시간 수치가 같다고 가정하지 않는다. 실제 속도 개선율은 같은 GPU·실험 설정에서 따로 측정해야 한다.
