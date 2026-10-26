import json
import os
from pathlib import Path
import time

import numpy as np
import torch

DEVICE = torch.device(os.environ.get('CEP_CUDA_DEVICE', 'cuda:0'))
if DEVICE.type != 'cuda' or not torch.cuda.is_available():
    raise RuntimeError('SCI/CEP requires CUDA; CPU fallback is disabled')
torch.cuda.set_device(DEVICE)
torch.backends.cuda.matmul.allow_tf32 = False


def record_call(method, begin, **fields):
    torch.cuda.synchronize(DEVICE)
    record = dict(method=method, device=str(DEVICE), seconds=time.perf_counter()-begin,
                  at=time.time(), **fields)
    folder = os.environ.get('CEP_RUN_ROOT')
    if folder:
        with (Path(folder)/'cuda_calls.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')


def csr_cuda(matrix):
    matrix = matrix.tocsr()
    matrix.sort_indices()
    return torch.sparse_csr_tensor(
        torch.from_numpy(matrix.indptr.astype(np.int64)).to(DEVICE),
        torch.from_numpy(matrix.indices.astype(np.int64)).to(DEVICE),
        torch.from_numpy(matrix.data.astype(np.float32)).to(DEVICE),
        size=matrix.shape, device=DEVICE)
