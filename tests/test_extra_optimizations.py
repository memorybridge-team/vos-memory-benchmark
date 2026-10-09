"""공유 CPU 입력, Native 통계, benchmark 가시성과 DB 저장 경계를 검증한다."""

from contextlib import nullcontext
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import settings
from evaluation import store
from evaluation.scoring import restoration
from model.sam2_runner import CPUFrameCache, LazyFrames, Session
from test_sqlite_store import test_all_frames_and_atomic_resume, test_concurrent_initialization
from test_cpu_scoring import test_measurement_isolation
from reference_restoration import tensor_score as reference_tensor_score


def test_rgb_cache():
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for frame in range(6):
            path = Path(tmp) / f'{frame}.png'
            Image.fromarray(np.full((40, 50, 3), 20 + frame, np.uint8)).save(path)
            paths.append(path)
        frame_bytes = 3 * 32 * 32 * 4
        cache = CPUFrameCache(frame_bytes * 3 / 1024 ** 2)
        expected = LazyFrames(paths, 32)
        first = LazyFrames(paths, 32)
        first.cache = cache
        try:
            first.configure_prefetch(2, 5)
            for frame in range(6):
                assert torch.equal(first[frame], expected[frame])
            assert first.profile['cache_misses'] == 3 and cache.nbytes == frame_bytes * 3
        finally:
            first.close()
        cached = LazyFrames(paths, 32)
        cached.cache = cache
        try:
            with patch.object(settings, 'PREFETCH_FRAMES', False):
                cached.configure_prefetch(2, 5)
            mask = cached[3]
            assert torch.equal(mask, expected[3]) and cached.profile['cache_hits'] == 1
            mask.fill_(99)
            assert torch.equal(cached._read(3), expected[3]), '캐시 반환 tensor의 저장 공간 공유'
            # 같은 경로의 변경을 감지한다. 새 Session의 첫/측정 프레임은 항상 우회한다.
            Image.fromarray(np.full((40, 50, 3), 199, np.uint8)).save(paths[3])
            assert torch.equal(cached._read(3), expected[3])
            assert cached.profile['cache_misses'] == 1
        finally:
            cached.close()
        measured = LazyFrames(paths, 32)
        measured.cache = cache
        try:
            measured.configure_prefetch(5, 5)
            for frame in range(6):
                assert torch.equal(measured[frame], expected[frame])
            assert measured.profile['cache_hits'] == measured.profile['cache_misses'] == 0
            assert measured.profile['read_count'] == 6
        finally:
            measured.close()
        assert cache.nbytes <= cache.limit
        no_cache = CPUFrameCache(0)
        no_cache.put(('input',), torch.ones(3))
        assert not no_cache.entries and no_cache.nbytes == 0
        # 모양/전처리 설정은 별도 입력이다.
        for size in (16, 64):
            frames = LazyFrames(paths, size)
            frames.cache = cache
            try:
                frames.configure_prefetch(-1, 0)
                assert frames[0].shape == (3, size, size)
                assert frames.profile['cache_hits'] == 0
            finally:
                frames.close()


def test_native_statistics():
    reference = reference_tensor_score
    rng = torch.Generator().manual_seed(112)
    cases = []
    for dtype in (torch.bfloat16, torch.float32, torch.float64):
        native = torch.randn((7, 5), generator=rng).to(dtype).T
        cases.extend((native, actual) for actual in (native.clone(), native + 1, native * -2))
    for shape, dtype in (((1, 64, 64, 64), torch.bfloat16), ((1, 256), torch.float32)):
        native = torch.randn(shape, generator=rng).to(dtype)
        cases.extend(((native, native.clone()), (native, native * .9 + .05)))
    cases += [(None, torch.ones(2)), (torch.ones(2), None), (torch.ones(2), torch.ones(3)),
              (torch.empty(0), torch.empty(0)), (torch.ones(1), torch.zeros(1)),
              (torch.ones(7), torch.zeros(7)), (torch.tensor([float('nan'), 1.]), torch.ones(2)),
              (torch.ones(2), torch.tensor([1., float('inf')])),
              (torch.tensor([1e308, -1e308], dtype=torch.float64), torch.zeros(2, dtype=torch.float64)),
              (torch.tensor([1e12, 1e12 + .001, 1e12 + .002], dtype=torch.float64),
               torch.tensor([1e12, 1e12 + .003, 1e12 + .007], dtype=torch.float64))]
    for cache_mb in (0, .0001, 128):
        cache = restoration.NativeStatsCache(cache_mb)
        for native, target in cases:
            for _ in range(3):
                assert restoration.tensor_score(native, target, cache) == reference(native, target)
                assert restoration.tensor_score(native, target) == reference(native, target)
                assert cache.nbytes <= cache.limit
    native = torch.arange(24, dtype=torch.float64)
    cache = restoration.NativeStatsCache(1)
    original = restoration.tensor_score(native, native.clone(), cache)
    restoration.tensor_score(native, native.clone(), cache)
    assert cache.hits == 1
    native.add_(1)
    assert restoration.tensor_score(native, native - 1, cache) == reference(native, native - 1)
    assert original['r2'] == 1 and cache.misses == 2
    with torch.inference_mode():
        native = torch.arange(16, dtype=torch.float32)
        snapshot = native.clone()
        assert restoration.tensor_score(snapshot, snapshot, cache) == reference(snapshot, snapshot)
        assert restoration.tensor_score(snapshot, snapshot, cache) == reference(snapshot, snapshot)


def test_visible_opt_in():
    def session():
        item = Session.__new__(Session)
        item.state = {}
        item.frames = SimpleNamespace(cache=None)
        item.skip_visible = False
        item._context = nullcontext
        item._visible = Mock(side_effect=[True, False])
        item._forget_old = Mock()
        logits = torch.tensor([[[[1., -1.], [-1., 1.]]]])
        item.predictor = SimpleNamespace(propagate_in_video=lambda *args, **kwargs:
                                        iter([(0, [1], logits), (1, [1], -logits)]))
        return item
    normal = session()
    outputs = list(normal.track(0, 1))
    assert [p.visible for p in outputs] == [True, False]
    for skip in (False, True):
        item = session()
        with patch.object(settings, 'BENCHMARK_SKIP_VISIBLE', skip):
            item.configure_benchmark()
            actual = list(item.track(0, 1))
        assert all(np.array_equal(a.mask, b.mask) for a, b in zip(actual, outputs))
        assert [p.visible for p in actual] == ([None, None] if skip else [True, False])
        assert item._visible.call_count == (0 if skip else 2)
        assert item._forget_old.call_count == 2


def test_reused_transactions():
    with tempfile.TemporaryDirectory() as tmp, patch.object(settings, 'OUTPUT_ROOT', tmp):
        calls = []
        original = store.sqlite3.connect
        def counted(*args, **kwargs):
            calls.append(args[0])
            return original(*args, **kwargs)
        with patch.object(store.sqlite3, 'connect', counted), store.reuse_connection():
            shared = store._SHARED_CONNECTION.get()[1]
            assert not shared.in_transaction
            with store.connect() as conn:
                assert conn is shared
                conn.execute('CREATE TABLE audit_values (id INTEGER PRIMARY KEY, value TEXT NOT NULL)')
                conn.execute('INSERT INTO audit_values VALUES (1, ?)', ('complete',))
            assert not shared.in_transaction
            # 다른 읽기 연결에 즉시 보이며 추론 중에 쓰기 잠금을 유지하지 않는다.
            reader = original(store.database_path())
            try:
                assert reader.execute('SELECT value FROM audit_values WHERE id=1').fetchone()[0] == 'complete'
            finally:
                reader.close()
            try:
                with store.connect() as outer:
                    outer.execute('INSERT INTO audit_values VALUES (2, ?)', ('outer',))
                    with store.connect() as inner:
                        inner.execute('INSERT INTO audit_values VALUES (3, ?)', ('inner',))
                    raise RuntimeError('outer failed after inner release')
            except RuntimeError:
                pass
            assert not shared.in_transaction
            with store.connect() as outer:
                outer.execute('INSERT INTO audit_values VALUES (4, ?)', ('kept',))
                try:
                    with store.connect() as inner:
                        inner.execute('INSERT INTO audit_values VALUES (5, ?)', ('rolled back',))
                        raise ValueError('inner failed')
                except ValueError:
                    pass
            with store.reuse_connection(), store.connect() as conn:
                assert [r['id'] for r in conn.execute('SELECT id FROM audit_values ORDER BY id')] == [1, 4]
                assert conn.execute('PRAGMA foreign_keys').fetchone()[0] == 1
                assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
                conn.execute("CREATE TRIGGER abort_audit BEFORE INSERT ON audit_values WHEN NEW.id=7 "
                             "BEGIN SELECT RAISE(ROLLBACK, 'abort transaction'); END")
            try:
                with store.connect() as conn:
                    conn.execute("INSERT INTO audit_values VALUES (6, 'partial')")
                    conn.execute("INSERT INTO audit_values VALUES (7, 'failed')")
            except sqlite3.IntegrityError as exc:
                assert str(exc) == 'abort transaction'
            else:
                raise AssertionError('SQLite의 자동 rollback을 무시함')
            with store.connect() as conn:
                assert [r['id'] for r in conn.execute('SELECT id FROM audit_values ORDER BY id')] == [1, 4]
                conn.execute("INSERT INTO audit_values VALUES (8, 'retry')")
            # 다른 읽기 트랜잭션 때문에 commit이 막혀도 연결은 깨끗하게 rollback한다.
            shared.execute('PRAGMA busy_timeout=0')
            reader = original(store.database_path())
            try:
                reader.execute('BEGIN')
                reader.execute('SELECT * FROM audit_values').fetchall()
                try:
                    with store.connect() as conn:
                        conn.execute("INSERT INTO audit_values VALUES (9, 'blocked')")
                except sqlite3.OperationalError as exc:
                    assert 'locked' in str(exc)
                else:
                    raise AssertionError('다른 읽기 연결의 commit 잠금이 재현되지 않음')
                assert not shared.in_transaction
            finally:
                reader.close()
            with store.connect() as conn:
                assert not conn.execute('SELECT 1 FROM audit_values WHERE id=9').fetchone()
                conn.execute("INSERT INTO audit_values VALUES (9, 'retry after lock')")
            assert len(calls) == 1 and not shared.in_transaction
        assert store._SHARED_CONNECTION.get() is None
        try:
            shared.execute('SELECT 1')
        except sqlite3.ProgrammingError:
            pass
        else:
            raise AssertionError('재사용 연결을 닫지 않음')
    # 실제 부모/자식 행의 중간 실패, 완료 키와 재개도 재사용 연결로 검증한다.
    original_connect = store.connect
    with tempfile.TemporaryDirectory() as tmp, patch.object(settings, 'OUTPUT_ROOT', tmp):
        # 기존 테스트의 setup이 OUTPUT_ROOT를 바꾸므로 각 첫 호출의 실제 경로에 scope를 둔다.
        scopes = {}
        def shared_connect():
            key = str(store.database_path().resolve())
            if key not in scopes:
                scope = store.reuse_connection()
                scope.__enter__()
                scopes[key] = scope
            return original_connect()
        try:
            with patch.object(store, 'connect', shared_connect):
                test_all_frames_and_atomic_resume()
        finally:
            for scope in reversed(list(scopes.values())):
                scope.__exit__(None, None, None)


if __name__ == '__main__':
    test_rgb_cache()
    test_native_statistics()
    test_visible_opt_in()
    test_reused_transactions()
    test_measurement_isolation()
    test_concurrent_initialization()
    print('OK: RGB 캐시/소유권/측정 우회, Native R² 정확 일치, 가시성 옵션, DB 재사용/중단/재개')
