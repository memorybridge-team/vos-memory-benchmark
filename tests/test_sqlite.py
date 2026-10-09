"""SQLite의 원점수/BLOB roundtrip, 원자성, 기존 파일 보존·이관, 다중 프로세스 쓰기."""
import copy
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import settings
from evaluation import store,records,native
from test_end_to_end import setup,run_script,dataset_rows


def writer_row(video,obj):
    return {'dataset':'parallel','video':video,'object':obj,'start':0,'end':45,
            'switch_name':'25','switch_frame':11,'baseline':'source_only','baseline_revision':1,
            'evaluation_revision':settings.EVALUATION_REVISION,'run_id':1,'seed':0,
            'native_reference_id':'1'*32,'frame_scores':[{'frame':12,'j':.75,'f':.8,'jf':.775}],
            'pre_switch_frame_scores':[],'restoration_frame_scores':[]}


def test_storage():
    with tempfile.TemporaryDirectory(prefix='vos_sqlite_') as tmp:
        setup(Path(tmp));settings.DATABASE_PATH=None
        run_script('1_make_video_list.py','--datasets','pumavos')
        run_script('2_evaluate.py','--dataset','pumavos','--runs','1')
        rows=dataset_rows('pumavos')
        assert len(rows)==3*3*6
        before=json.dumps(rows,sort_keys=True)
        assert store.save_evaluations(rows)==0
        assert json.dumps(dataset_rows('pumavos'),sort_keys=True)==before
        references=list(native.load_references('pumavos').values())
        blobs={r['native_reference_id']:store.memory_blob(r['native_reference_id']) for r in references}
        assert all(blobs.values())
        with store.connection() as conn:
            assert conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
            assert conn.execute('SELECT COUNT(*) FROM native_memory').fetchone()[0]==3
            assert conn.execute('SELECT COUNT(*) FROM native_scores').fetchone()[0]>0
            assert conn.execute('SELECT COUNT(*) FROM restoration_scores').fetchone()[0]>0
            found=conn.execute("SELECT f.j,f.jf FROM frame_scores f JOIN evaluations e ON e.id=f.evaluation_id "
                               "WHERE e.dataset='pumavos' AND e.baseline='translator' AND f.phase='post' LIMIT 1").fetchone()
            assert 0<=found[0]<=1 and 0<=found[1]<=1
        # 한 묶음의 두 번째 결과가 잘못되면 첫 번째도 저장되지 않는다.
        good=copy.deepcopy(rows[0]);good['video']='atomic_good'
        bad=copy.deepcopy(rows[0]);bad['video']='atomic_bad'
        bad['frame_scores']=[bad['frame_scores'][0]]*2
        try:
            store.save_evaluations([good,bad])
            raise AssertionError('부분 transaction이 확정됨')
        except sqlite3.IntegrityError:
            pass
        assert json.dumps(dataset_rows('pumavos'),sort_keys=True)==before
        for ref in references:
            memories=native.load_memories(ref)
            assert set(memories)=={s['frame'] for s in ref['switches']}
        # 이전 파일 형식으로 export한 뒤 새 DB에 이관한다.
        legacy=records.records_path('pumavos')
        records.append_rows(legacy,rows)
        ref_file=native.reference_path('pumavos',1)
        records.append_rows(ref_file,references)
        for ref in references:
            path=native.memory_path(ref);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(blobs[ref['native_reference_id']])
        sources=[legacy,ref_file]+[native.memory_path(r) for r in references]
        source_bytes={p:p.read_bytes() for p in sources}
        settings.DATABASE_PATH=str(Path(tmp)/'import.sqlite3')
        report=store.import_legacy()
        assert report['evaluations']==len(rows) and report['memory_blobs']==3
        assert json.dumps(dataset_rows('pumavos'),sort_keys=True)==before
        assert all(store.memory_blob(identity)==blob for identity,blob in blobs.items())
        assert store.import_legacy()=={'evaluations':0,'references':0,'memory_blobs':0}
        assert all(p.read_bytes()==content for p,content in source_bytes.items())
        assert len(records.done_keys('pumavos'))==len(rows)
        settings.DATABASE_PATH=None


def test_concurrent_writers():
    with tempfile.TemporaryDirectory(prefix='vos_sqlite_workers_') as tmp:
        settings.DATABASE_PATH=str(Path(tmp)/'parallel.sqlite3')
        with store.connection():
            pass
        jobs=[subprocess.Popen([sys.executable,str(Path(__file__)),'--writer',settings.DATABASE_PATH,str(i)],
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for i in range(2)]
        for job in jobs:
            out,error=job.communicate(timeout=60)
            assert job.returncode==0,(out,error)
        rows=list(store.iter_evaluations('parallel'))
        assert len(rows)==24 and len(records.unique_rows(rows))==24
        with store.connection() as conn:
            assert conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        settings.DATABASE_PATH=None


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--writer':
        settings.DATABASE_PATH=sys.argv[2]
        for i in range(12):
            store.save_evaluations([writer_row('worker'+sys.argv[3],i)])
    else:
        test_storage();test_concurrent_writers()
        print('OK: SQLite 점수/Native BLOB roundtrip, 중복 방지, 원자성, 이전 원본 보존 이관, 두 프로세스 동시 쓰기')
