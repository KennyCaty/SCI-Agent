import time
import torch
from sci.retrieval.cuda_runtime import DEVICE, record_call


def compute_background(r, g):
    begin = time.perf_counter()
    original_device = g.device
    with torch.inference_mode():
        g = g.to(device=DEVICE, dtype=torch.float32)
        z = torch.sparse.mm(r.tt, g[:, None])[:, 0]
        mask = (z <= 0).float()
        denom = torch.where(z <= 0, torch.ones_like(z), z)
        b0 = torch.full_like(g, 1/r.n)
        b = b0.clone()
        for _ in range(24):
            propagated = torch.sparse.mm(r.t, (b/denom)[:, None])[:, 0]
            b = .5*b0 + .5*(g*propagated + b0*(b*mask).sum())
        h = torch.rsqrt(b)
        result = (b[:, None].to(original_device), denom.to(original_device), h.to(original_device))
    record_call('pagerank', begin, iterations=24, nodes=r.n, edges=r.matrix.nnz,
                backend='cuda_csr_spmv', query_result_cache=False)
    return result
