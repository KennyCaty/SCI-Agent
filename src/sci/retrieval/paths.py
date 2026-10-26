import json
import os
from pathlib import Path

from sci.paths import ROOT
PROJECT = ROOT


def resource(name):
    config = Path(os.environ.get('CEP_RESOURCES_CONFIG', ROOT/'configs/retrieval/browsecomp_plus.json'))
    values = json.loads(config.read_text())
    raw = os.environ.get('CEP_'+name.upper(), values[name])
    return Path(raw.format(project=str(PROJECT), root=str(ROOT), home=str(Path.home()))).expanduser().resolve()
