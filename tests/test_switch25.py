"""25/50/75 공통 제외, 정확히 8의 경계, 실행 전 검증과 집계 제외."""
import json
import runpy
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'tests')]
import settings
import fake_data
from evaluation import records,store
from evaluation.data import load_video_list,video_list_path
from evaluation.switches import switch_points,object_exclusion
from evaluation.evaluate_video import evaluate_object
from evaluation.methods import METHODS
from test_end_to_end import setup,run_script


def obj(start,end):
    return {'object':1,'start':start,'end':end,'switches':switch_points(start,end)}


def test_exclusion():
    assert settings.SWITCH_FRACTIONS == (.25,.5,.75)
    for start,end in ((0,12),(0,32),(0,34),(20,52)):
        case = obj(start,end)
        reason = object_exclusion(case)
        assert reason and reason['scope'] == 'all_switches_methods_runs'
        def forbidden(*args):
            raise AssertionError('제외한 객체에 프롬프트/모델 실행')
        assert evaluate_object(SimpleNamespace(object_mask=forbidden),case,None,None,METHODS) == []
    assert object_exclusion(obj(0,35)) is None
    assert object_exclusion(obj(20,55)) is None
    bad = obj(0,45);bad['switches'] = bad['switches'][1:]
    assert object_exclusion(bad)['reason'] == 'invalid_switch_schedule'

    with tempfile.TemporaryDirectory(prefix='vos_switch25_') as tmp:
        setup(Path(tmp))
        root = Path(settings.DATA_ROOT)/settings.DATA_FOLDERS['pumavos']
        with patch.object(fake_data,'N',33):
            fake_data.make_video(root/'JPEGImages',root/'Annotations','exactly8')
        run_script('1_make_video_list.py','--datasets','pumavos')
        plan = load_video_list('pumavos')
        assert next(v for v in plan['videos'] if v['video']=='exactly8')['objects'] == []
        excluded = [r for r in plan['excluded_objects'] if r['video']=='exactly8']
        assert len(excluded)==3 and plan['skipped_short_objects']==3
        assert all(r['scope']=='all_switches_methods_runs' for r in excluded)
        valid = next(v for v in plan['videos'] if v['video']=='pumavos_00')['objects']
        assert len(valid)==3 and all([sw['name'] for sw in o['switches']]==['25','50','75'] for o in valid)
        # 고의로 변경한 목록도 GPU 모델을 켜기 전에 거절한다.
        plan['videos'][0]['objects'].append(obj(0,32))
        video_list_path('pumavos').write_text(json.dumps(plan),encoding='utf-8')
        try:
            load_video_list('pumavos')
            raise AssertionError('유효하지 않은 공통 모집단 허용')
        except ValueError as exc:
            assert '전체 평가 제외' in str(exc)
        raw = []
        for sw in switch_points(0,32):
            for method in METHODS:
                raw.append({'dataset':'pumavos','video':'short','object':1,'start':0,'end':32,
                            'switch_name':sw['name'],'switch_frame':sw['frame'],
                            'baseline':method.name,'baseline_revision':method.revision,
                            'evaluation_revision':settings.EVALUATION_REVISION,'run_id':1,'seed':0,
                            'native_reference_id':'0'*32,'frame_scores':[],'pre_switch_frame_scores':[],
                            'restoration_frame_scores':[]})
        store.save_evaluations(raw)
        assert records.current_rows(raw)==[] and records.done_keys('pumavos')==set()
        loaded=runpy.run_path(str(ROOT/'scripts/3_make_tables.py'))['load_raw_rows']()
        assert loaded==[]

if __name__=='__main__':
    test_exclusion()
    print('OK: <=8이면 객체의 모든 25/50/75 방법/회차 제외; 목록/실행/집계 공통 검증')
