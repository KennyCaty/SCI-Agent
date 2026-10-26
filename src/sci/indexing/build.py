import fcntl
import json
import os
import subprocess
import sys
import time
import traceback

from sci.indexing.common import ASSETS, INDEX, ROOT, identity, progress, save


def stage(name):
    log=(ROOT/f'outputs/indexing/logs/build-{name}.log').open('a')
    child=subprocess.Popen([sys.executable,'-u','-m',f'sci.indexing.{name}'],stdout=log,stderr=subprocess.STDOUT,
        env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'})
    return child,log


def wait(job,name):
    child,log=job
    status=child.wait();log.close()
    if status:raise RuntimeError(f'{name} failed; see outputs/indexing/logs/build-{name}.log')


def valid_marker(name):
    path=ASSETS/name
    if not path.exists():return False
    marker=json.loads(path.read_text())
    if marker['build_identity']!=identity():raise ValueError('Build identity differs: '+name)
    return marker['status']=='PASS'


def main():
    (ROOT/'outputs/indexing/logs').mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'outputs/indexing/logs/build.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(ROOT/'outputs/indexing/logs/BUILD_STARTED.json',dict(pid=os.getpid(),at=time.time(),build_identity=identity()))
    if (INDEX/'READY.json').exists():
        ready=json.loads((INDEX/'READY.json').read_text())
        assert ready['build_identity']==identity();return
    error=ROOT/'outputs/indexing/logs/BUILD_ERROR.json'
    if error.exists():
        error.rename(ROOT/f'outputs/indexing/logs/BUILD_ERROR_resolved_{int(time.time())}.json')
    children=[]
    try:
        if not valid_marker('SEGMENTS_READY.json'):wait(stage('segment'),'segment')
        dense=None
        if not valid_marker('DENSE_READY.json'):dense=stage('dense');children.append(dense[0])
        if not valid_marker('ENTITIES_READY.json'):wait(stage('entities'),'entities')
        wait(stage('graph'),'graph')
        if not valid_marker('FTS_READY.json'):wait(stage('fts'),'fts')
        if dense:wait(dense,'dense')
        wait(stage('validate'),'validate')
        save(ROOT/'outputs/indexing/logs/BUILD_FINISHED.json',dict(status='PASS',at=time.time(),build_identity=identity()))
    except BaseException as exc:
        save(ROOT/'outputs/indexing/logs/BUILD_ERROR.json',dict(error=str(exc),at=time.time(),traceback=traceback.format_exc()))
        for child in children:
            if child.poll() is None:child.terminate();child.wait()
        raise


if __name__=='__main__':main()
