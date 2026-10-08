"""반복 평가의 seed. 조건별 seed는 GPU 분할·이어하기 순서에 영향받지 않는다."""

import hashlib
import json
import random

import numpy as np
import torch


def seed_condition(seed, run_id, dataset, video, obj_id, method):
    payload = json.dumps([seed, run_id, dataset, video, obj_id, method], ensure_ascii=False)
    value = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], 'big')
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    return value
