"""실패한 검사 차단, 데이터 누락, dtype 계약과 라벨 고정을 검증한다.

    python tests/test_validation.py
"""

import contextlib
import io
import json
import runpy
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import settings
import translator
from evaluation import data
from evaluation.data import lvos_v2
from evaluation.data.common import videos_from_folders
from evaluation.tables import summaries
from model import sam2_check, sam2_runner


def rejected(call, error=ValueError, text=''):
    try:
        call()
    except error as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError('오류 없이 진행했습니다.')


def test_report_gate():
    with tempfile.TemporaryDirectory() as tmp, patch.object(settings, 'OUTPUT_ROOT', tmp), \
            patch.object(sam2_check, 'runtime_signature', return_value={'checkpoint': 'a'}) as signature:
        rejected(sam2_check.require_passed, text='통과 기록')
        checks = dict.fromkeys(sam2_check.check_names(), True)
        rejected(lambda: sam2_check.save_passed(dict(checks, shapes=False), {}, {}))
        assert not sam2_check.report_path().exists()
        sam2_check.save_passed(checks, {}, {})
        sam2_check.require_passed()
        signature.return_value = {'checkpoint': 'b'}
        rejected(sam2_check.require_passed, text='바뀌었습니다')
        signature.return_value = {'checkpoint': 'a'}
        payload = json.loads(sam2_check.report_path().read_text())
        payload['checks'].pop('pos_enc')
        sam2_check.report_path().write_text(json.dumps(payload))
        rejected(sam2_check.require_passed, text='모두 통과')
        sam2_check.report_path().write_text('{broken')
        rejected(sam2_check.require_passed, text='읽을 수')


def test_failed_script_and_evaluation_gate():
    video = SimpleNamespace(name='v', mask_paths={0: None}, num_frames=10,
                            read_labels=lambda _: (np.ones((2, 2), dtype=np.uint8), None))
    info = {'image_size': 1024, 'num_maskmem': 7, 'max_obj_ptrs_in_encoder': 16,
            'hidden_dim': 256, 'mem_dim': 64, 'fields': {}, 'missing_state_keys': [],
            'entry': {'maskmem_features': torch.zeros(1, dtype=torch.bfloat16),
                      'obj_ptr': torch.zeros(1)}}
    roundtrip = {'frames': 1, 'expected_frames': 1, 'identical_frames': 1,
                 'max_diff_pixels': 0, 'passed': True}
    with tempfile.TemporaryDirectory() as tmp, patch.object(settings, 'OUTPUT_ROOT', tmp), \
            patch.object(data, 'load_dataset', return_value=[video]), \
            patch.object(sam2_runner, 'load_runner', side_effect=lambda key: SimpleNamespace(name=key)), \
            patch.object(sam2_check, 'describe', return_value=info), \
            patch.object(sam2_check, 'shape_mismatches', return_value=['forced mismatch']), \
            patch.object(sam2_check, 'same_pos_enc', return_value=True), \
            patch.object(sam2_check, 'roundtrip', return_value=roundtrip), \
            patch.object(sys, 'argv', ['0_check_sam2.py']), \
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        sam2_check.report_path().parent.mkdir(parents=True)
        sam2_check.report_path().write_text('{}')
        try:
            runpy.run_path(str(ROOT / 'scripts/0_check_sam2.py'), run_name='__main__')
        except SystemExit as exc:
            assert exc.code == 1
        else:
            raise AssertionError('검사 실패의 종료 코드가 0입니다.')
        assert not sam2_check.report_path().exists()

    plan = {'videos': [{'video': 'v', 'objects': [
        {'object': 1, 'start': 0, 'end': 32, 'switches': [{'name': '50', 'frame': 16}]}]}]}
    with patch.object(data, 'load_video_list', return_value=plan), \
            patch('evaluation.records.done_keys', return_value=set()), \
            patch.object(sam2_check, 'require_passed', side_effect=ValueError('검사 차단')) as gate, \
            patch.object(data, 'load_dataset', side_effect=AssertionError('차단 전에 데이터 로딩')), \
            patch.object(translator, 'load', side_effect=AssertionError('차단 전에 translator 로딩')), \
            patch.object(sys, 'argv', ['2_evaluate.py', '--dataset', 'vost_val']):
        rejected(lambda: runpy.run_path(str(ROOT / 'scripts/2_evaluate.py'), run_name='__main__'),
                 text='검사 차단')
        gate.assert_called_once()


def test_window_and_missing_roundtrip():
    info = {'num_maskmem': 7, 'max_obj_ptrs_in_encoder': 16}
    assert sam2_check.window_is_enough(info)
    assert not sam2_check.window_is_enough(dict(info, memory_temporal_stride_for_eval=5))
    with patch.object(settings, 'MEMORY_WINDOW', 5):
        assert not sam2_check.window_is_enough(info)
    class EmptySession:
        def add_prompt(self, *args): pass
        def track(self, *args): return iter(())
        def export_memory(self): return {}
        def load_memory(self, *args): pass
        def close(self): pass
    runner = SimpleNamespace(start=lambda _: EmptySession())
    video = SimpleNamespace(num_frames=5, object_mask=lambda *args: np.ones((2, 2), dtype=bool))
    result = sam2_check.roundtrip(runner, video, 1, 0, 1, 4)
    assert not result['passed'] and result['expected_frames'] == 3
    assert result['missing_frames'] == result['missing_reference_frames'] == [2, 3, 4]
    rejected(lambda: sam2_check.roundtrip(runner, video, 1, 0, 1, 1))


def test_data_validation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frames, masks = root / 'frames/v', root / 'masks/v'
        frames.mkdir(parents=True)
        masks.mkdir(parents=True)
        for name in ('00001', '00002'):
            (frames / f'{name}.jpg').touch()
        (masks / '1.png').touch()
        (masks / '00002.png').touch()
        load = lambda **kw: videos_from_folders('test', root / 'frames', root / 'masks', **kw)
        rejected(load, text='연결 안 된 정답 1개')
        (masks / '1.png').unlink()
        rejected(load, text='정답 없는 프레임 1개')
        assert len(load(require_all_masks=False)[0].mask_paths) == 1
        (masks / '00001.png').touch()
        assert len(load()[0].mask_paths) == 2
        (frames / '00001.png').touch()
        rejected(load, text='중복')

        base = root / 'LVOS'
        for folder in ('val', 'valid'):
            (base / folder).mkdir(parents=True)
        with patch.object(lvos_v2, 'dataset_root', return_value=base), \
                patch.object(settings, 'LVOS_SPLIT_FOLDER', None), \
                patch.object(settings, 'LVOS_ATTRIBUTE_FILE', None):
            rejected(lambda: lvos_v2._split_root('valid'), text='모두 있습니다')
            with patch.object(settings, 'LVOS_SPLIT_FOLDER', 'valid'):
                assert lvos_v2._split_root('valid') == base / 'valid'
                for name, attribute in [('a_attribute.json', 'OCC'), ('z_attribute.json', 'LR')]:
                    (base / 'valid' / name).write_text(json.dumps({'videos': {'v': {'attributes': [attribute]}}}))
                rejected(lambda: lvos_v2.load_labels('valid'), text='여러 개')
                with patch.object(settings, 'LVOS_ATTRIBUTE_FILE', 'z_attribute.json'):
                    assert lvos_v2.load_labels('valid')['v']['*'] == ['LR 작은 물체']


def test_translation_contract_and_labels():
    class FloatTranslator:
        def translate_handoff_tensors(self, spatial, pointer):
            return spatial.float() + .25, pointer.double() + .5
    entry = {'maskmem_features': torch.zeros(1, 4, 2, 2, dtype=torch.bfloat16),
             'obj_ptr': torch.zeros(1, 5), 'is_cond': True}
    with patch.object(settings, 'DEVICE', 'cpu'), patch.object(translator, '_translator', FloatTranslator()):
        result = translator.translate({0: entry})[0]
        assert result['maskmem_features'].dtype == torch.bfloat16
        assert result['obj_ptr'].dtype == torch.float32
        assert bool((result['maskmem_features'] == .25).all())
        assert bool((result['obj_ptr'] == .5).all())
        assert result['is_cond'] is True
        assert bool((entry['maskmem_features'] == 0).all())
    class WrongShape:
        def translate_handoff_tensors(self, spatial, pointer):
            return spatial[..., :1], pointer
    with patch.object(settings, 'DEVICE', 'cpu'), patch.object(translator, '_translator', WrongShape()):
        rejected(lambda: translator.translate({0: entry}), text='모양')

    load_labels = runpy.run_path(str(ROOT / 'scripts/3_make_tables.py'))['load_object_labels']
    rows = [{'dataset': 'm3vos', 'video': 'v', 'object': 1, 'extra_labels': ['old']}]
    assert load_labels(rows)[('m3vos', 'v', 1)] == ['old']
    rejected(lambda: load_labels(rows + [dict(rows[0], extra_labels=['new'])]), text='라벨이 다릅니다')
    stats_rows = [{'dataset': 'm3vos', 'video': f'v{i}', 'object': i, 'switch_name': '50',
                   'switch_frame': 5, 'baseline': 'direct_state_copy', 'run_id': 1,
                   'j': .5, 'r2_obj_ptr': .2 if i == 1 else None} for i in (1, 2)]
    _, runs, _ = summaries.build(stats_rows)
    assert runs[0]['object_count'] == 2 and runs[0]['r2_obj_ptr_object_count'] == 1


if __name__ == '__main__':
    test_report_gate()
    test_failed_script_and_evaluation_gate()
    test_window_and_missing_roundtrip()
    test_data_validation()
    test_translation_contract_and_labels()
    print('OK: 검사 실패/환경 변경 차단, 프레임 누락, dtype·모양, 입력 선택, 라벨과 표본 수')
