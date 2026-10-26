import json
import os
import sqlite3
import threading
from types import SimpleNamespace

import bm25s
import numpy as np
import Stemmer

from sci.retrieval.paths import resource
from sci.retrieval.dense import DenseIndex
from sci.retrieval.embedding import QueryEncoder


class SourceBM25:
    def __init__(self, folder):
        self.retriever = bm25s.BM25.load(str(folder/'bm25_index'), load_corpus=False)
        self.source_ids = tuple(map(str, json.loads((folder/'doc_ids.json').read_text())))

    @staticmethod
    def stemmer():
        return Stemmer.Stemmer('english')


class FullBackends:
    def __init__(self):
        assets = resource('graph_assets')
        self.source_ids = tuple(map(str, json.loads((assets/'graph/source_ids.json').read_text())))
        self.segment_sources = np.load(assets/'graph/segment_sources.npy')
        self.segment_seqs = np.load(assets/'graph/segment_seqs.npy')
        dimension=int(os.environ.get('CEP_EMBEDDING_DIMENSION','256'))
        dense = DenseIndex(assets/('segment_dense' if dimension==256 else 'segment_dense1024'))
        if not np.array_equal(dense.unit_seqs, self.segment_seqs):
            raise ValueError('Dense/graph segment row identities differ')
        if len(self.segment_sources) != dense.rows:
            raise ValueError('Dense/graph row counts differ')
        encoder = QueryEncoder(resource('embedding_model'), dense.dimension)
        self.searcher = SimpleNamespace(encoder=encoder, semantic_index=dense)
        self.source_segments = [[] for _ in self.source_ids]
        for row, source in enumerate(self.segment_sources):
            self.source_segments[int(source)].append(row)
        self.bm25 = SourceBM25(resource('bm25_index'))
        with sqlite3.connect('file:'+str(assets/'base.sqlite')+'?mode=ro', uri=True) as db:
            self.mapping = {str(s): str(p) for s, p in db.execute('SELECT source_id,relpath FROM source')}
        if len(set(self.source_ids)) != len(self.source_ids) or len(set(self.bm25.source_ids)) != len(self.bm25.source_ids):
            raise ValueError('Duplicate corpus/BM25 source IDs')
        if set(self.mapping) != set(self.source_ids) or set(self.bm25.source_ids) != set(self.source_ids):
            raise ValueError('Source database, graph and BM25 corpus identities differ')
        self.cache = {}
        self.lock = threading.RLock()
