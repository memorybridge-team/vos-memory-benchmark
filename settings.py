"""모든 숫자와 경로는 여기에만 둔다.

다른 파일은 `import settings` 후 `settings.이름` 으로 읽는다 (복사해 두지 않는다).
"""

# ── 경로 ─────────────────────────────────────────────────────────────
DATA_ROOT = "/workspace/data"            # RunPod 서버의 data 폴더
DATA_FOLDERS = {                         # DATA_ROOT 아래 데이터셋별 폴더 이름
    "mosev2": "MOSEv2",
    "lvos_v2": "LVOSv2",
    "vost": "VOST",
    "m3vos": "M3VOS",
    "pumavos": "PUMaVOS",
}
OUTPUT_ROOT = "outputs"                  # 목록·통계·결과·표가 모두 여기로
SAM2_CHECKPOINT_DIR = "/workspace/checkpoints"

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

# ── 데이터 나누기 ─────────────────────────────────────────────────────
TRAIN_DATASETS = ("mosev2_train", "lvos_v2_train")   # fit / dev 로 나누는 데이터셋
DEV_FRACTION = 0.2                       # train 영상 중 dev 로 쓰는 비율
MIN_TRACK_FRAMES = 8                     # 객체가 처음 보인 뒤 남은 프레임이 이보다 적으면 뺌

# ── 전환 시점 ─────────────────────────────────────────────────────────
SWITCH_A_FRACTIONS = (0.25, 0.50, 0.75)  # 전환 A: 객체 구간의 25/50/75% 지점 (주)
SWITCH_B_MIN_ABSENT = 3                  # 전환 B: 정답 프레임 기준 이만큼 연속으로 안 보여야 함
SWITCH_B_PER_OBJECT = 1                  # 전환 B: 객체마다 몇 개 (안 보인 기간이 긴 것부터)

# ── 비교군 ───────────────────────────────────────────────────────────
REPLAY_KS = (4, 8, 16)                   # Original+Replay-K 의 K
EXTRA_RECENT_K = 8                       # [추가] recent_k_only 의 K

# ── Moment-Matched Copy 통계 ─────────────────────────────────────────
MOMENT_STATS_DATASETS = ("mosev2_train", "lvos_v2_train")  # 이 데이터셋들의 fit 영상에서 계산
MOMENT_STATS_MAX_VIDEOS = 50             # 데이터셋마다 최대 영상 수
MOMENT_STATS_MAX_FRAMES = 200            # 영상마다 최대 프레임 수
MOMENT_STATS_SEED = 0

# ── 채점 ─────────────────────────────────────────────────────────────
BOUNDARY_THRESHOLD = 0.008               # F: 경계 허용 거리 = 이미지 대각선 × 이 값 (DAVIS 기본값)

# ── [추가] 지표 ──────────────────────────────────────────────────────
EXTRA_JF_AT = 5                          # jf@N: 전환 뒤 처음 보이는 N 프레임
EXTRA_RECOVERY_J = 0.5                   # 회복 지연: J가 이 값을 넘을 때까지 걸린 프레임
EXTRA_FAILURE_J = 0.1                    # 실패: 전환 뒤 평균 J가 이 값보다 낮으면
EXTRA_ID_SWITCH_IOU = 0.3                # ID 뒤바뀜: 다른 객체와 이만큼 이상, 자기보다 더 겹치면

# ── [추가] 조건별 분류 ────────────────────────────────────────────────
EXTRA_SMALL_AREA = 0.01                  # 작은 객체: 화면 넓이 대비 중앙값 넓이 < 이 값
EXTRA_FAST_MOTION = 0.02                 # 빠른 움직임: 프레임당 중심 이동 / 대각선 중앙값 > 이 값
EXTRA_CROSSING_BOX_IOU = 0.1             # 교차: 다른 객체 상자와 이만큼 이상 겹치는 프레임이 있으면

# ── MOSEv2 서버 제출 ──────────────────────────────────────────────────
MOSEV2_SUBMIT_EXTRAS = False             # True 면 [추가] 비교군도 제출 파일을 만듦 (+9번)
MOSEV2_ZIP_INNER_FOLDER = "Annotations"  # zip 안 폴더 이름 ("" 면 영상 폴더가 맨 위)
