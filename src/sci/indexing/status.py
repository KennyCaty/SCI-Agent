import json
from pathlib import Path
import shutil
import time
from sci.indexing.common import INDEX, ROOT


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def collect():
    started=read(ROOT/'outputs/indexing/logs/BUILD_ACTIVE.json') or read(ROOT/'outputs/indexing/logs/BUILD_STARTED.json')
    alive=False
    if started:
        try:
            command=(Path('/proc')/str(started['pid'])/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            alive=('-m sci.indexing.build' in command or '-m sci index build' in command or 'sci index build' in command)
        except (FileNotFoundError,ProcessLookupError):pass
    ready=read(INDEX/'READY.json');error=read(ROOT/'outputs/indexing/logs/BUILD_ERROR.json')
    phases={name:read(ROOT/f'outputs/indexing/logs/{name}_STATUS.json') for name in ('segments','dense','entities','graph','fts','complete')}
    free=shutil.disk_usage(ROOT).free
    return dict(at=time.time(),status='READY' if ready else 'failed' if error else 'building' if alive else 'stopped',
       controller_alive=alive,controller=started,phases=phases,current_error=error,ready=ready,
       free_gib=round(free/(1<<30),2))


if __name__=='__main__':print(json.dumps(collect(),ensure_ascii=False,indent=2))
