from pathlib import Path
import hashlib
import json
import os
import time

from sci.paths import ROOT
CONFIG = Path(os.environ.get('SCI_INDEX_CONFIG', ROOT/'configs/indexing/browsecomp_plus.json'))
if not CONFIG.is_absolute():
    CONFIG = ROOT/CONFIG
SETTINGS = json.loads(CONFIG.read_text())
DATASET = SETTINGS.get('dataset', 'bcp100k')
if DATASET not in ('bcp100k', 'trec-covid'):
    raise ValueError('Unsupported dataset label')
INDEX = ROOT/'indexes'/DATASET
ASSETS = INDEX/'assets'
DATA = ROOT/'data'/DATASET
MODEL = ROOT/'models/Qwen3-Embedding-0.6B'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    os.replace(temporary, path)


def config():
    return json.loads(CONFIG.read_text())


def identity():
    return hashlib.sha256(json.dumps(config(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def progress(stage, **fields):
    record = dict(stage=stage, at=time.time(), pid=os.getpid(), **fields)
    save(ROOT/'outputs/indexing/logs/BUILD_STATUS.json', record)
    save(ROOT/f'outputs/indexing/logs/{stage}_STATUS.json', record)
    print(json.dumps(record), flush=True)
