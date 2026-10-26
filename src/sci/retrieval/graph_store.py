from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
from scipy import sparse

from sci.retrieval.paths import resource
ART = resource('graph_assets')


def clean(text):
    return re.sub(r'\s+', ' ', str(text)).strip()


class GraphStore:
    def __init__(self, assets=ART):
        self.assets = Path(assets)
        self.ids = [str(s) for s in np.load(self.assets / 'global.npz')['source_ids'].tolist()]
        self.lookup = {s: i for i, s in enumerate(self.ids)}
        self.outgoing = sparse.load_npz(self.assets / 'source_transition.npz').tocsc()
        self.source_entity = sparse.load_npz(self.assets / 'graph/source_entity.npz').tocsr()
        self.entity_seqs = np.load(self.assets / 'graph/entity_seqs.npy')
        self.db = sqlite3.connect(f'file:{self.assets / "base.sqlite"}?mode=ro', uri=True)
        self.entities = sqlite3.connect(f'file:{self.assets / "rare_entities.sqlite"}?mode=ro', uri=True)
        self.corpus = resource('corpus')
        self._sources, self._neighbors, self._edges = {}, {}, {}
        self.body_access_log = []
        self.a, self.entityseq = self.source_entity, self.entity_seqs
        self.graph_hash = hashlib.sha256((self.assets/'source_transition.npz').read_bytes()).hexdigest()

    def source(self, sid):
        if sid not in self._sources:
            if sid not in self.lookup:
                raise ValueError('Unknown source ID')
            title, relpath = self.db.execute('SELECT title,relpath FROM source WHERE source_id=?', (sid,)).fetchone()
            self._sources[sid] = dict(source_id=sid, title=title, relpath=relpath)
        return self._sources[sid]

    def text(self, sid, workspace_ids):
        if sid not in workspace_ids:
            raise ValueError('External source bodies must be acquired before reading')
        self.body_access_log.append(sid)
        return (self.corpus / self.source(sid)['relpath']).read_bytes().decode('utf8')

    def neighbors(self, sid):
        if sid not in self._neighbors:
            row = self.lookup[sid]
            a, b = self.outgoing.indptr[row:row+2]
            self._neighbors[sid] = {
                self.ids[int(v)]: float(w)
                for v, w in zip(self.outgoing.indices[a:b], self.outgoing.data[a:b])
                if self.ids[int(v)] != sid and w > 0
            }
        return self._neighbors[sid]

    def edge(self, source, target):
        key = (source, target)
        if key not in self._edges:
            if target not in self.neighbors(source):
                raise ValueError(f'No directed graph edge {source} -> {target}')
            shared = self.source_entity.getrow(self.lookup[source]).multiply(self.source_entity.getrow(self.lookup[target]))
            if not shared.nnz:
                raise ValueError('Transition has no shared-entity witness')
            names = []
            for j in np.argsort(-shared.data)[:2]:
                seq = int(self.entity_seqs[shared.indices[j]])
                name = self.entities.execute('SELECT display_name FROM entity WHERE entity_seq=?', (seq,)).fetchone()[0]
                names.append(name)
            self._edges[key] = dict(source=source, target=target, shared_entities=names,
                                    transition_weight=self.neighbors(source)[target])
        return self._edges[key]


class WorkspaceLedger:
    """Acquisition and query support only; never treats unseen/unread as negative."""
    def __init__(self, graph, question):
        self.graph, self.question = graph, question
        self.ids, self.query_ranks, self.routes = set(), {}, {}
        self.current_queries, self.revision = [], 0
        self.shown, self.read_returned = set(), set()
        self.preview_windows = {}
        self.handles = {}
        self.map = graph
        self.view_serial = 0
        self.current_view_id = 'unissued'
        self.overview_path = None

    def ingest(self, batch, old_map=None):
        previous_ids = set(self.ids)
        queries = batch['queries']
        self.current_queries = list(dict.fromkeys(queries))
        for query, sources, scores, routes in zip(queries, batch['per_query_ids'], batch['per_query_scores'], batch['routes']):
            # Allocate across queries; display in score order.
            assert len(sources) == len(scores) and np.isfinite(scores).all()
            ranked = sorted(zip(sources, scores), key=lambda x: (-x[1], int(x[0])))
            ranks = self.query_ranks.setdefault(query, {})
            for rank, (sid, score) in enumerate(ranked, 1):
                self.ids.add(sid)
                ranks[sid] = min(rank, ranks.get(sid, rank))
            for route_rank, route in enumerate(routes,1):
                path = route['path_source_ids']
                if path[-1] != route['source_id']:
                    raise ValueError('Route endpoint mismatch')
                for a, b in zip(path, path[1:]):
                    self.graph.edge(a, b)
                self.ids.update(path)
                key = (query, tuple(path))
                self.routes[key] = dict(query=query, path=list(path), scope=route['scope'],
                    acquisition_revision=self.revision+1,
                    query_id=hashlib.sha256(query.encode()).hexdigest()[:12],
                    route_rank=route_rank, cep_rank=ranks[path[-1]], cep_score=route['score'])
        if old_map:
            for source in old_map['sources']:
                if source.get('literal'):
                    continue
                head = any(c['source_id'] == source['source_id'] for group in old_map['groups'] for c in group['cards'])
                if head and source.get('windows'):
                    self.preview_windows[source['source_id']] = source['windows'][0]
        self.last_added = len(self.ids - previous_ids)
        if not hasattr(self,'map_routes'):
            self.map_routes={};self.map_query_ranks={}
        for query,sources,scores,routes in zip(queries,batch.get('map_per_query_ids',batch['per_query_ids']),
                batch.get('map_per_query_scores',batch['per_query_scores']),batch.get('map_routes',batch['routes'])):
            ranks=self.map_query_ranks.setdefault(query,{})
            for rank,(sid,score) in enumerate(sorted(zip(sources,scores),key=lambda x:(-x[1],int(x[0]))),1):
                ranks[sid]=min(rank,ranks.get(sid,rank))
            for route in routes:
                path=list(route['path_source_ids'])
                self.map_routes[query,tuple(path)]=dict(query=query,path=path,route_rank=ranks[path[-1]],**{k:v for k,v in route.items() if k!='path_source_ids'})
        self.revision += 1

    def relevance(self, sid):
        return tuple(61 / (60 + self.query_ranks[q][sid]) if sid in self.query_ranks[q] else 0.
                     for q in self.current_queries)

    def overall(self, sid):
        return sum(self.relevance(sid))

    def handle(self, kind, anchor, target=None):
        if anchor not in self.ids:
            raise ValueError('Navigation must start in the acquired workspace')
        if target:
            self.graph.edge(anchor, target)
        raw = json.dumps([self.current_view_id, kind, anchor, target, self.graph.graph_hash])
        key = ('B' if kind == 'focus' else 'F') + self.current_view_id + '-' + hashlib.sha256(raw.encode()).hexdigest()[:10]
        self.handles[key] = dict(kind=kind, anchor=anchor, target=target,
            issued_view=self.current_view_id,graph_asset_hash=self.graph.graph_hash)
        return key

    def execute_handle(self, key):
        """Exact model-free action; also exposed by the native tool backend."""
        if key not in self.handles:
            raise ValueError('Unknown Map handle')
        record = self.handles[key]
        if record['graph_asset_hash'] != self.graph.graph_hash:
            raise ValueError('Handle graph version mismatch')
        anchor, target = record['anchor'], record['target']
        if record['kind'] == 'route':
            path = record['path']
            if not path or not any(tuple(r['path']) == tuple(path) for r in self.map_routes.values()):
                raise ValueError('Expansion path has no recorded CEP witness')
            for a,b in zip(path,path[1:]):
                self.graph.edge(a,b)
            new = [s for s in path if s not in self.ids]
            self.ids.update(new)
            self.revision += 1
            return dict(kind='route',focus=path[-1],newly_acquired=list(dict.fromkeys(new)),
                        issued_view=record['issued_view'],applied_workspace_revision=self.revision)
        if anchor not in self.ids:
            raise ValueError('Handle anchor is unavailable')
        new = []
        if record['kind'] == 'follow':
            self.graph.edge(anchor, target)
            if target not in self.ids:
                self.ids.add(target)
                new = [target]
        elif record['kind'] != 'focus':
            raise ValueError('Unknown handle action')
        self.revision += 1
        return dict(kind=record['kind'], focus=target or anchor, newly_acquired=new,
                    issued_view=record['issued_view'],applied_workspace_revision=self.revision,
                    prototype_only=True, note='Exact handle action, not CEP query ranking or a factual relation.')
