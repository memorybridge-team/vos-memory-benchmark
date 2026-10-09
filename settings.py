"""모든 숫자와 경로는 여기에만 둔다.

다른 파일은 `import settings` 후 `settings.이름` 으로 읽는다 (복사해 두지 않는다).
"""

# ── 경로 ─────────────────────────────────────────────────────────────
DATA_ROOT = "/workspace"                 # RunPod 서버
DATA_FOLDERS = {                         # DATA_ROOT 아래 데이터셋별 폴더
    "lvos_v2": "CMMT/data/LVOSv2/extracted",
    "vost": "datasets/VOST/extracted/VOST",
    "m3vos": "CMMT/data/M3VOS-manual",
    "pumavos": "CMMT/data/PUMaVOS/extracted/PUBLIC_PUMaVOS",
}
OUTPUT_ROOT = "outputs"                  # 목록·결과·표가 모두 여기로
SAM2_CHECKPOINT_DIR = "/workspace/CMMT/checkpoints"

# ── 모델 ─────────────────────────────────────────────────────────────
MODELS = {
    "small": {
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "checkpoint": "sam2.1_hiera_small.pt",
    },
    "base_plus": {
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "checkpoint": "sam2.1_hiera_base_plus.pt",
    },
}
SOURCE_MODEL = "small"                   # 전환 전에 도는 모델
TARGET_MODEL = "base_plus"               # 전환 뒤에 이어받는 모델
DEVICE = "cuda"
USE_BF16 = True                          # SAM2 공식 예제와 같이 bfloat16으로 돌림
OFFLOAD_STATE_TO_CPU = False             # GPU 메모리가 모자라면 True
MEMORY_WINDOW = 16                       # 최근 몇 프레임의 기억 칸을 들고 있을지
                                         # (SAM2는 최근 6장 maskmem + 15장 obj_ptr 를 읽음)

# ── 본 모델 (translator) ─────────────────────────────────────────────
# 팀 전달본 official_state_loss_final_delivery 를 푼 폴더 (안에 selected_state_loss_best/, source/)
TRANSLATOR_DIR = "/workspace/CMMT-official-isolated/official-20261005T012256KST/final-state-loss-20261005T1627KST/delivery"
TRANSLATOR_WEIGHTS = "selected_state_loss_best/translator_weights.pth"   # 선정 epoch 27
TRANSLATOR_SHA256 = "92802842b0f9c2247f95627784a4919625aaced43633c5203f619472ae0f17da"   # 전달 보고서 값
TRANSLATOR_SOURCE = "source/src"                                          # 팀 코드 (vos_memory_inspector)

# ── 영상 목록 ─────────────────────────────────────────────────────────
MIN_TRACK_FRAMES = 8                     # 하나라도 전환 프레임 - 등장 프레임 <= 8이면 객체 전체 제외

# ── 전환 시점 ─────────────────────────────────────────────────────────
SWITCH_FRACTIONS = (0.25, 0.50, 0.75)          # 객체 최초 등장부터 영상 끝까지의 25/50/75% 지점

# ── 비교군 ───────────────────────────────────────────────────────────
REPLAY_FRAMES = 8                        # Original-Prompt(s)+Replay-8: 전환 직전 다시 볼 프레임 수

# ── 채점 (주) ─────────────────────────────────────────────────────────
BOUNDARY_THRESHOLD = 0.008               # F: 경계 허용 거리 = 이미지 대각선 × 이 값 (DAVIS 기본값)
J_MAIN_DATASETS = ("vost_val", "m3vos")  # 주 지표가 J 인 데이터셋 (VOST 는 F 를 쓰지 않고, M3VOS 논문도 F 로 평가하지 않음). 나머지는 J&F

# ── 평가 지표 ────────────────────────────────────────────────────────
RECALL_J = 0.5                          # 실패 비율: 1 − mean(J > 0.5), J는 내부적으로 0~1
VIDEO_LIST_REVISION = 3                 # 25/50/75% 공통 객체 + 엄격한 >8 조건
EVALUATION_REVISION = 5                 # 세 전환 모두 유효한 객체만 평가; SQLite 원본 저장
EVALUATION_RUNS = 3                     # 전체 평가 반복 횟수
EVALUATION_SEED = 0                     # 조건별 seed의 기준값

# 계산/저장 정의는 같지만 메모리 복사·측정 오버헤드를 줄인 실행 버전. 비용 비교 시 구분한다.
EVALUATION_RUNTIME_REVISION = 3

# SQLite 원본 저장. None이면 OUTPUT_ROOT/benchmark.sqlite3. 다중 GPU는 같은 DB를 공유한다.
DATABASE_PATH = None
SQLITE_TIMEOUT_SECONDS = 60
# 한 프로세스에서 한 GPU 평가. CPU 이미지 읽기/정규화만 다음 프레임과 겹친다.
FRAME_PREFETCH = 2                       # 선행 CPU 프레임 수. 0이면 동기 로딩
