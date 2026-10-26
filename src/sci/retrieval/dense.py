import json
from pathlib import Path
import time

import numpy as np
import torch
from sci.retrieval.cuda_runtime import DEVICE, record_call


class DenseIndex:
    def __init__(self, folder):
        folder = Path(folder)
        manifest = json.loads((folder/'manifest.json').read_text())
        if manifest['state'] != 'complete' or manifest['kind'] != 'segment':
            raise ValueError('A complete immutable segment index is required')
        self.rows, self.dimension = manifest['row_count'], manifest['dimension']
        self.unit_seqs = np.memmap(folder/manifest['id_file'], dtype=np.uint64, mode='r', shape=(self.rows,))
        data = np.memmap(folder/manifest['embedding_file'], dtype=np.float16, mode='r', shape=(manifest['capacity'], self.dimension))[:self.rows]
        self.matrix = torch.from_numpy(data.copy()).to(DEVICE)
        self.device = DEVICE

    def score(self, query):
        return self.score_queries(np.asarray(query).reshape(1, -1))[0]

    def score_queries(self, queries):
        begin = time.perf_counter()
        with torch.inference_mode():
            values = torch.as_tensor(np.asarray(queries, np.float16), device=DEVICE)
            result = torch.mm(values, self.matrix.T).float().cpu().numpy()
        record_call('dense_queries', begin, queries=len(queries), rows=self.rows)
        return result

    def score_subset(self, query, rows):
        begin = time.perf_counter()
        with torch.inference_mode():
            selected = torch.as_tensor(np.asarray(rows), device=DEVICE, dtype=torch.long)
            vector = torch.as_tensor(np.asarray(query, np.float16), device=DEVICE)
            result = torch.mv(self.matrix.index_select(0, selected), vector).float().cpu().numpy()
        record_call('dense_subset', begin, rows=len(rows))
        return result
