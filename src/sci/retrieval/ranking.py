import numpy as np
from scipy import sparse
import torch
from sci.retrieval.cuda_runtime import DEVICE, csr_cuda


def order(v, ids, mask=None):
    rows = np.arange(len(v)) if mask is None else np.flatnonzero(mask)
    return rows[np.lexsort((ids[rows], -v[rows]))]


def vote(v, ids, positive=False):
    ranked = order(v, ids, v > 0 if positive else None)
    out = np.zeros(len(v))
    out[ranked] = 1/(60+np.arange(1, len(ranked)+1))
    return out


class RetrievalResources:
    def __init__(self, backends, *, transition_path, source_ids):
        self.b = backends
        self.searcher = backends.searcher
        self.ids = list(backends.source_ids)
        assert self.ids == list(source_ids), 'Graph source order differs from index source order'
        self.numeric = np.asarray(list(map(int, self.ids)))
        self.n = len(self.ids)
        lookup = {sid: i for i, sid in enumerate(backends.bm25.source_ids)}
        self.bm_rows = np.asarray([lookup[s] for s in self.ids])
        self.source_rows = torch.as_tensor(backends.segment_sources, device=DEVICE, dtype=torch.long)
        self.matrix = sparse.load_npz(transition_path).tocsr()
        self.t, self.tt = csr_cuda(self.matrix), csr_cuda(self.matrix.T)
        self.context_cache = {}
