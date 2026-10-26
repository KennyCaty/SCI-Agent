from pathlib import Path
from types import SimpleNamespace
import os,sys,json,time,hashlib,csv,copy,subprocess
from sci.paths import ROOT as P
from sci.retrieval.graph_store import GraphStore
from sci.retrieval.witness import query_top_routes
import tiktoken
ENC=tiktoken.get_encoding('o200k_base')
def nt(s):return len(ENC.encode(s))
def clean(s):return ' '.join(str(s).split())
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)
def textwrite(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(value);os.replace(tmp,path)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

class Session:
    def __init__(self,folder,service=None,graph=None):
        self.folder=Path(folder).resolve();self.folder.mkdir(parents=True,exist_ok=True)
        self.workspace=self.folder/'workspace';self.graph=graph or GraphStore();self.service=service
        f=self.folder/'state.json'
        self.state=json.loads(f.read_text()) if f.exists() else dict(revision=0,ids=[],nodes={},paths={},blocks=[],operations={},graph_hash=self.graph.graph_hash)
        assert self.state['graph_hash']==self.graph.graph_hash
    def engine(self):
        if self.service is None:
            from sci.retrieval.retrieval_service import Service
            self.service=Service()
        assert self.service.r.ids==self.graph.ids
        return self.service
    def register(self,state,groups,kind,anchor=None):
        newblocks=[];idx=sum(b['kind']==kind for b in state['blocks'])
        for group in groups:
            idx+=1;bid=f'{kind}{idx}';routes=[]
            for route in group['routes']:
                path=route['path'];key=':'.join(path)
                for sid in path:
                    if sid not in state['nodes']:state['nodes'][sid]=f'N{len(state["nodes"])+1:02}'
                if key not in state['paths']:
                    state['paths'][key]=dict(id=f'P{len(state["paths"])+1:02}',source_ids=path)
                routes.append(dict(route,path_id=state['paths'][key]['id']))
            block=dict(id=bid,kind=kind,query=group['query'],anchor=anchor,routes=routes)
            state['blocks'].append(block);newblocks.append(block)
        return newblocks
    def render(self,state,blocks,revision,delta=None):
        ids=set(state['ids']);labels=state['nodes'];visible=list(dict.fromkeys(s for b in blocks for r in b['routes'] for s in r['path']))
        lines=[f'CEP MAP {"UPDATE" if delta is not None else "SNAPSHOT"} r{revision}',f'Workspace: {len(ids)} sources | Shown: {sum(s in ids for s in visible)} IN, {sum(s not in ids for s in visible)} OUT.',
               'IN: readable in .cep/s/. OUT: available for graphexpand. Connections show shared entities.']
        if delta is not None:
            lines.append(f'Acquired this operation: {len(delta["new_ids"])} sources; blocks: '+', '.join(b['id'] for b in blocks)+'.')
            if delta['became_in']:lines.append('STATUS: '+', '.join(labels[s]+' OUT → IN' for s in delta['became_in']))
        lines.append('\nSOURCES')
        for sid in visible:lines.append(f'{labels[sid]} [{"IN" if sid in ids else "OUT"}] source={sid} | {clean(self.graph.source(sid)["title"])}')
        for block in blocks:
            prefix=f'\n{block["id"]} '+('QUERY' if block['kind']=='Q' else 'EXPANSION')
            if block['anchor']:prefix+=f' from {labels[block["anchor"]]} (source={block["anchor"]})'
            lines.extend([prefix,block['query']])
            seen=set()
            for route in block['routes']:
                if route['path_id'] in seen:continue
                seen.add(route['path_id']);path=route['path'];s=labels[path[0]]
                for a,b in zip(path,path[1:]):
                    e=self.graph.edge(a,b);assert a in self.graph.neighbors(b)
                    s+=' — {'+' / '.join(clean(x) for x in e['shared_entities'])+'} — '+labels[b]
                lines.append('  '+route['path_id']+': '+s+(' [DIRECT]' if len(path)==1 else ''))
        lines+=['\nRead: read(path=".cep/s/<source>.txt"). Search: bash(command="rg ... .cep/s").',
                'Expand from an IN or OUT source: graphexpand(source_id="<source>", query="<needed information>").',
                'Full Map: .cep/MAP.txt']
        rendered='\n'.join(lines)+'\n'
        assert not self.graph.body_access_log
        return dict(text=rendered,tokens=nt(rendered),visible_ids=visible,block_ids=[b['id'] for b in blocks])
    def materialize(self,state):
        root=self.workspace/'.cep/s';root.mkdir(parents=True,exist_ok=True);(self.workspace/'.tool_tmp').mkdir(exist_ok=True)
        for sid in state['ids']:
            target=root/f'{sid}.txt'
            if not target.exists():os.link(self.graph.corpus/self.graph.source(sid)['relpath'],target)
        f=self.workspace/'.cep/catalogue.tsv';temp=f.with_suffix('.tmp')
        with temp.open('w') as out:
            w=csv.writer(out,delimiter='\t');w.writerow(['source_id','title','read_path'])
            for sid in state['ids']:w.writerow([sid,clean(self.graph.source(sid)['title']),f'.cep/s/{sid}.txt'])
        os.replace(temp,f)
    def commit(self,groups,incoming,kind,anchor,params,operation_id,provenance,batch):
        old=self.state;state=copy.deepcopy(old);before=set(old['ids']);new=set(incoming)-before
        became=sorted(set(old['nodes'])&new,key=int)
        state['ids']=sorted(before|set(incoming),key=int);state['revision']+=1;rev=state['revision']
        blocks=self.register(state,groups,kind,anchor)
        delta=dict(new_ids=sorted(new,key=int),became_in=became)
        local=self.render(state,blocks,rev,delta);full=self.render(state,state['blocks'],rev)
        result=dict(text=local['text'],details=dict(revision=rev,operation='graphsearch' if kind=='Q' else 'graphexpand',params=params,
                    newly_acquired=delta['new_ids'],became_in=became,blocks=blocks,workspace_total=len(state['ids']),local_map_tokens=local['tokens'],full_map_tokens=full['tokens'],provenance=provenance))
        state['operations'][operation_id]=dict(request_hash=digest([kind,params]),result_file=f'events/{rev:04}/result.json')
        folder=self.folder/f'events/{rev:04}';self.materialize(state)
        dump(folder/'raw_batch.json',batch);dump(folder/'result.json',result);dump(folder/'state.json',state)
        textwrite(folder/'DELTA_MAP.txt',local['text']);textwrite(folder/'FULL_MAP.txt',full['text'])
        # MAP.txt can be regenerated from the committed state.
        dump(self.folder/'state.json',state);self.state=state
        textwrite(self.workspace/'.cep/MAP.txt',full['text']);textwrite(self.folder/'MAP.txt',full['text'])
        return result
    def cached(self,operation_id,kind,params):
        record=self.state['operations'].get(operation_id)
        if record:
            if record['request_hash']!=digest([kind,params]):raise ValueError('Operation ID already used with different parameters')
            return json.loads((self.folder/record['result_file']).read_text())
    def ingest_fixture(self,batch,groups,provenance):
        incoming={s for xs in batch['per_query_ids'] for s in xs}|{s for rs in batch['routes'] for r in rs for s in r['path_source_ids']}
        assert {s for g in groups for r in g['routes'] for s in r['path']}<=incoming
        return self.commit(groups,incoming,'Q',None,dict(queries=batch['queries']),'fixture-init',provenance,batch)
    def graphsearch(self,args,operation_id):
        if set(args)!={'queries'} or not isinstance(args['queries'],list) or not args['queries'] or not all(isinstance(q,str) and q.strip() for q in args['queries']):raise ValueError('queries must be a nonempty list of strings')
        hit=self.cached(operation_id,'Q',args)
        if hit:return hit
        from sci.retrieval.retriever import CEPRetriever
        svc=self.engine();batch=CEPRetriever.search(svc.r,(self.folder/'question.txt').read_text(),args['queries'],budget=1000,explain=True)
        groups=query_top_routes(svc.r,10)
        incoming={s for xs in batch['per_query_ids'] for s in xs}|{s for rs in batch['routes'] for r in rs for s in r['path_source_ids']}
        assert {s for g in groups for r in g['routes'] for s in r['path']}<=incoming
        return self.commit(groups,incoming,'Q',None,args,operation_id,'fresh native CEP graphsearch',batch)
    def graphexpand(self,args,operation_id):
        if set(args)-{'source_id','query'} or 'source_id' not in args:raise ValueError('source_id and optional query expected')
        anchor=args['source_id']
        if not isinstance(anchor,str) or anchor not in set(self.state['ids'])|set(self.state['nodes']):raise ValueError('Anchor must be acquired or displayed in the Map')
        query=args.get('query',(self.folder/'question.txt').read_text())
        if not isinstance(query,str) or not query.strip():raise ValueError('query must be nonempty')
        hit=self.cached(operation_id,'E',args)
        if hit:return hit
        provisional=set(self.state['ids'])|{anchor}
        w=SimpleNamespace(ids=provisional,map=self.graph)
        batch=self.engine().expand(w,dict(source_ids=[anchor],query=query))
        routes=sorted(batch['routes'][0],key=lambda r:(-r['score'],int(r['source_id'])))[:10]
        groups=[dict(query=query,routes=[dict(source_id=r['source_id'],path=r['path_source_ids'],score=r['score'],reconstructed_score=r['reconstructed_score']) for r in routes])]
        # Keep the anchor even when expansion returns no candidates.
        if not routes:groups[0]['routes']=[dict(source_id=anchor,path=[anchor],score=None,reconstructed_score=None)]
        incoming={anchor}|{s for xs in batch['per_query_ids'] for s in xs}|{s for rs in batch['routes'] for r in rs for s in r['path_source_ids']}
        return self.commit(groups,incoming,'E',anchor,args,operation_id,'fresh local CEP; two source hops; beam64; up to100 endpoints',batch)
    def dispatch(self,op,args,operation_id=None):
        from sci.retrieval.tool_validation import validate_arguments
        validate_arguments(op,args)
        if op in ['graphsearch','graphexpand']:return getattr(self,op)(args,operation_id or str(time.time_ns()))
        if op not in ['read','bash']:raise ValueError('Only four tools enabled')
        f=self.folder/'tool_io'/str(time.time_ns());f.mkdir(parents=True);dump(f/'input.json',args)
        subprocess.run(['node',str(P/'extensions/native_tools.mjs'),str(self.workspace),op,str(f/'input.json'),str(f/'output.json')],check=True)
        return json.loads((f/'output.json').read_text())
