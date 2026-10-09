"""CPU 선행 준비의 순서/정확성/정리와 GPU 중복 실행 잠금."""
import sys
import tempfile
from pathlib import Path
from threading import Event
from uuid import uuid4
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.sam2_runner import LazyFrames
from evaluation.runtime import resource_lock


def test_prefetch():
    with tempfile.TemporaryDirectory(prefix='vos_prefetch_') as tmp:
        paths=[]
        rng=np.random.default_rng(7)
        for i in range(6):
            path=Path(tmp)/f'{i}.png'
            Image.fromarray(rng.integers(0,256,(20,30,3),dtype=np.uint8)).save(path)
            paths.append(path)
        ready=Event()
        class Observed(LazyFrames):
            def _decode(self,i):
                tensor=super()._decode(i)
                if i==1:ready.set()
                return tensor
        sync=LazyFrames(paths,32,prefetch=0)
        ahead=Observed(paths,32,prefetch=2)
        try:
            assert torch.equal(sync[0],ahead[0])
            assert ready.wait(5), '다음 CPU 프레임을 선행 처리하지 않음'
            for i in (1,2,5,0,4,-1):
                assert torch.equal(sync[i],ahead[i]),i
                assert len(ahead._pending)<=2
        finally:
            sync.close();ahead.close()
        assert not ahead._pending
        ahead.close()
        try:
            ahead[0]
            raise AssertionError('닫힌 executor 사용')
        except RuntimeError:
            pass


def test_lock():
    token=str(uuid4())
    with resource_lock(token):
        try:
            with resource_lock(token):
                raise AssertionError('같은 GPU의 동시 평가 허용')
        except RuntimeError:
            pass
        with resource_lock(token+'another'):
            pass
    with resource_lock(token):
        pass

if __name__=='__main__':
    test_prefetch();test_lock()
    print('OK: CPU 선행 준비 값/순서 동일, bounded queue/정리, GPU 잠금 충돌/해제')
