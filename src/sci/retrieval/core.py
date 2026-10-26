from pathlib import Path
import copy, fcntl, json, os, time

import sci.retrieval.workspace_core as base
GraphStore = base.GraphStore
dump, textwrite, nt, digest, clean = base.dump, base.textwrite, base.nt, base.digest, base.clean
from sci.retrieval.navigation import navigation_groups


class Session(base.Session):
    def __init__(self, folder, service=None, graph=None):
        self.folder = Path(folder).resolve(); self.folder.mkdir(parents=True, exist_ok=True)
        self.workspace = self.folder/'workspace'
        self.graph = graph or GraphStore(); self.service = service
        with self._lock():
            self._restore()

    def _lock(self):
        class Lock:
            def __enter__(inner):
                inner.file = (self.folder/'session.lock').open('a')
                fcntl.flock(inner.file, fcntl.LOCK_EX)
            def __exit__(inner, *exc):
                inner.file.close()
        return Lock()

    def _restore(self):
        path = self.folder/'state.json'
        state = json.loads(path.read_text()) if path.exists() else dict(
            schema='sci-cep-v1',revision=0,ids=[],nodes={},paths={},edges={},queries={},blocks=[],operations={},graph_hash=self.graph.graph_hash)
        if state.get('schema') != 'sci-cep-v1' or state['graph_hash'] != self.graph.graph_hash:
            raise ValueError('SCI/CEP requires its own workspace and matching graph version')
        self.state = state
        # Recover from the last committed state after an interrupted write.
        root = self.workspace/'.cep/s'
        if root.exists():
            allowed = set(state['ids'])
            for file in root.glob('*.txt'):
                if file.stem.isdigit() and file.stem not in allowed:
                    file.unlink()
        self.materialize(state)
        self._write_views(state)

    def _write_views(self, state):
        full = self.render_full(state)
        textwrite(self.workspace/'.cep/MAP.txt', full['text'])
        textwrite(self.folder/'MAP.txt', full['text'])
        dump(self.workspace/'.cep/MAP.json', state)

    def _register(self, state, groups, kind, anchor, revision):
        delta = dict(new_nodes=[],new_edges=[],new_paths=[],blocks=[])
        for group in groups:
            query_key = digest([kind,anchor,group['query']])
            if query_key not in state['queries']:
                state['queries'][query_key] = dict(id=f'Q{len(state["queries"])+1:03}',query=group['query'],kind=kind,anchor=anchor,latest_revision=revision,routes=[])
            query = state['queries'][query_key]; routes = []
            for rank, original in enumerate(group['routes'], 1):
                route = dict(original,rank=rank)
                if kind=='E':
                    route.update(ranking_score=route['score'],path_score=route.get('reconstructed_score'),
                                 navigation_status='local_cep_path' if len(route['path'])>1 else 'no_local_candidates')
                path = route['path']
                assert path[-1] == route['source_id'] and len(set(path)) == len(path)
                for sid in path:
                    if sid not in state['nodes']:
                        state['nodes'][sid] = f'N{len(state["nodes"])+1:03}'
                        delta['new_nodes'].append(sid)
                edge_ids = []
                for a,b in zip(path,path[1:]):
                    key = ':'.join(sorted([a,b],key=int))
                    if key not in state['edges']:
                        evidence = self.graph.edge(a,b)
                        state['edges'][key] = dict(id=f'L{len(state["edges"])+1:03}',source_ids=sorted([a,b],key=int),shared_entities=evidence['shared_entities'],queries=[])
                        delta['new_edges'].append(key)
                    edge = state['edges'][key]
                    if query['id'] not in edge['queries']:edge['queries'].append(query['id'])
                    edge_ids.append(edge['id'])
                key = ':'.join(path)
                if key not in state['paths']:
                    state['paths'][key] = dict(id=f'P{len(state["paths"])+1:03}',source_ids=path,edge_ids=edge_ids,queries=[])
                    delta['new_paths'].append(key)
                record = state['paths'][key]
                if query['id'] not in record['queries']:record['queries'].append(query['id'])
                route['path_id'] = record['id']; routes.append(route)
            query.update(latest_revision=revision,routes=routes)
            block = dict(id=f'B{len(state["blocks"])+1:04}',query_id=query['id'],kind=kind,query=group['query'],anchor=anchor,revision=revision,routes=routes)
            state['blocks'].append(block); delta['blocks'].append(block)
        return delta

    def _source_line(self, state, sid):
        return f'{state["nodes"][sid]} [{"IN" if sid in set(state["ids"]) else "OUT"}] source={sid} | {clean(self.graph.source(sid)["title"])}'

    def _path_line(self, state, key):
        path = state['paths'][key]; nodes = path['source_ids']
        text = path['id']+': '+state['nodes'][nodes[0]]
        for a,b in zip(nodes,nodes[1:]):
            edge = state['edges'][':'.join(sorted([a,b],key=int))]
            text += ' — {'+' / '.join(clean(x) for x in edge['shared_entities'])+'} — '+state['nodes'][b]
        if len(nodes)==1:text += ' [NO ROUTE SHOWN]'
        return text

    def _rank_lines(self, state, query):
        label='SEARCH' if query['kind']=='Q' else 'EXPANSION'
        lines=[query.get('query_id',query.get('id'))+' '+label+': '+query['query']]
        if query.get('anchor'):lines.append('Anchor: source='+query['anchor'])
        if query['kind']=='E' and all(r['score'] is None for r in query['routes']):
            lines.append('No local candidates returned; anchor remains available. This does not establish that no relevant documents exist.')
            return lines
        lines.append('Ranked endpoints: '+', '.join(
            f'{r["rank"]}={state["nodes"][r["source_id"]]}/{r["path_id"]}' for r in query['routes']))
        return lines

    def render_full(self, state):
        lines=[f'CEP MAP SNAPSHOT r{state["revision"]}',f'Workspace: {len(state["ids"])} readable sources.',
               'IN is readable; OUT requires acquisition. Paths are CEP navigation clues, not proven factual relationships.',
               'Rank scores and navigation path scores are separate; source IDs are used in tools.','\nSOURCES']
        lines += [self._source_line(state,s) for s in state['nodes']]
        lines += ['\nPATHS']+[self._path_line(state,k) for k in state['paths']]
        lines += ['\nQUERIES (latest ranks per exact query; prior operations are in the event journal)']
        for query in state['queries'].values():lines += self._rank_lines(state,query)
        lines += ['\nRead .cep/s/<source>.txt; search .cep/s/; expand an IN/OUT source with the missing-information query.']
        text='\n'.join(lines)+'\n'
        assert not self.graph.body_access_log
        return dict(text=text,tokens=nt(text))

    def render_delta(self, state, delta):
        lines=[f'CEP MAP UPDATE r{state["revision"]}',f'Workspace: {len(state["ids"])} readable files; newly acquired files: {len(delta["new_ids"])}; new Map nodes: {len(delta["new_nodes"])}; new connections: {len(delta["new_edges"])}.',
               'IN: readable. OUT: acquire before reading. Connections are shared-entity navigation clues, not established facts.']
        if delta['became_in']:lines.append('OUT → IN: '+', '.join(state['nodes'][s]+' source='+s for s in delta['became_in']))
        lines += ['\nNEW MAP NODES (first shown; may already be acquired)']+[self._source_line(state,s) for s in delta['new_nodes']]
        lines += ['\nNEW PATHS']+[self._path_line(state,k) for k in delta['new_paths']]
        lines += ['\nCURRENT RANKS (rank=endpoint/path; existing N/P definitions remain valid)']
        for block in delta['blocks']:lines += self._rank_lines(state,block)
        if not any(delta[k] for k in ('new_ids','new_nodes','new_edges','new_paths')):
            lines.append('No new files, nodes, connections or paths. Current ranks are shown above.')
        lines += ['\nFull Map and source IDs/titles: .cep/MAP.txt; read .cep/s/<source>.txt; search .cep/s/.',
                  'Expand an IN/OUT source: graphexpand(source_id="<source>", query="<missing information>").']
        text='\n'.join(lines)+'\n'
        return dict(text=text,tokens=nt(text))

    def commit(self, groups, incoming, kind, anchor, params, operation_id, provenance, batch):
        old = self.state; state = copy.deepcopy(old); before = set(old['ids'])
        incoming = set(incoming); new = incoming-before
        state['ids'] = sorted(before|incoming,key=int); state['revision'] += 1; rev=state['revision']
        delta = self._register(state,groups,kind,anchor,rev)
        delta.update(new_ids=sorted(new,key=int),became_in=sorted(new&set(old['nodes']),key=int))
        local=self.render_delta(state,delta);full=self.render_full(state)
        result=dict(text=local['text'],details=dict(method='SCI/CEP',revision=rev,operation='graphsearch' if kind=='Q' else 'graphexpand',params=params,
            newly_acquired=delta['new_ids'],became_in=delta['became_in'],blocks=delta['blocks'],delta=delta,
            workspace_total=len(state['ids']),local_map_tokens=local['tokens'],full_map_tokens=full['tokens'],provenance=provenance))
        folder=self.folder/f'events/{rev:04}'
        state['operations'][operation_id]=dict(request_hash=digest([kind,params]),result_file=f'events/{rev:04}/result.json')
        dump(folder/'raw_batch.json',batch);dump(folder/'result.json',result);dump(folder/'state.json',state)
        textwrite(folder/'DELTA_MAP.txt',local['text']);textwrite(folder/'FULL_MAP.txt',full['text'])
        # Commit state after the files and event; roll back files if the commit fails.
        try:
            self.materialize(state)
            dump(self.folder/'state.json',state)
        except Exception:
            self._restore()
            raise
        self.state=state
        self._write_views(state)
        return result

    def graphsearch(self, args, operation_id):
        if set(args)!={'queries'} or not args['queries'] or not all(isinstance(q,str) and q.strip() for q in args['queries']):
            raise ValueError('queries must be nonempty strings')
        hit=self.cached(operation_id,'Q',args)
        if hit:return hit
        from sci.retrieval.retriever import CEPRetriever
        svc=self.engine()
        with svc.lock:
            batch=CEPRetriever.search(svc.r,(self.folder/'question.txt').read_text(),args['queries'],budget=1000,explain=True)
            ranked=base.query_top_routes(svc.r,10)
            groups=navigation_groups(svc.r,ranked)
        incoming={s for row in batch['per_query_ids'] for s in row}|{s for rs in batch['routes'] for r in rs for s in r['path_source_ids']}
        assert [[(r['source_id'],r['score']) for r in g['routes']] for g in groups]==[[(r['source_id'],r['score']) for r in g['routes']] for g in ranked]
        # Path-only documents stay OUT until acquired.
        return self.commit(groups,incoming,'Q',None,args,operation_id,'native CEP ranks/acquisition; best nonzero simple CEP navigation paths',batch)

    def graphexpand(self, args, operation_id):
        svc=self.engine()
        with svc.lock:
            return super().graphexpand(args,operation_id)

    def dispatch(self, op, args, operation_id=None):
        with self._lock():
            self._restore()
            return super().dispatch(op,args,operation_id)
