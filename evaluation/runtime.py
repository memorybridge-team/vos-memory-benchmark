"""같은 물리 GPU에서 평가 프로세스가 겹쳐 비용 측정을 왜곡하지 않도록 잠근다."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import tempfile

import torch


@contextmanager
def resource_lock(identity):
    token = hashlib.sha256(str(identity).encode()).hexdigest()[:24]
    path = Path(tempfile.gettempdir()) / f'vos_benchmark_gpu_{token}.lock'
    with path.open('a+b') as stream:
        if path.stat().st_size == 0:
            stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('같은 GPU에서 다른 평가가 실행 중입니다. GPU별로 프로세스 하나만 실행하세요.') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


@contextmanager
def gpu_lease(device, *, enabled=True):
    parsed = torch.device(device)
    if not enabled or parsed.type != 'cuda' or not torch.cuda.is_available():
        yield
        return
    index = parsed.index if parsed.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    identity = getattr(properties,'uuid',None)
    if identity is None:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        identity = visible.split(',')[index].strip() if visible else str(index)
    with resource_lock(identity):
        torch.cuda.set_device(index)
        yield
