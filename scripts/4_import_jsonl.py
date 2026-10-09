"""이전 JSONL/Native .pt를 SQLite에 추가한다. 원본 파일은 보존한다."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation import store

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    args = parser.parse_args()
    print(store.import_legacy(args.dataset))
    print(store.database_path())
