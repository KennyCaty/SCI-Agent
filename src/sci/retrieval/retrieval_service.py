import threading
import numpy as np
import torch
from sci.retrieval.paths import resource
from sci.retrieval.backends import FullBackends
from sci.retrieval.cuda_runtime import DEVICE
from sci.retrieval.retriever import CEPRetriever
from sci.retrieval.ranking import order
from sci.retrieval.background import compute_background
ART=resource("graph_assets")
class Service:
    def __init__(self):
        self.backends=FullBackends()
        self.r=CEPRetriever(self.backends)
        self.lock=threading.RLock()
        self.contexts={}
        self.outgoing=self.r.matrix.tocsc()
        self.preview_vectors = {}
        self.preview_embedding_calls = 0

    def context(self, query):
        if query not in self.contexts:
            r = self.r
            vector = r.searcher.encoder.segment_queries([query],512)[0]
            with torch.inference_mode():
                matrix = r.searcher.semantic_index.matrix
                segment = torch.mv(matrix, torch.as_tensor(vector, device=DEVICE, dtype=matrix.dtype)).float()
                relevance = torch.full((r.n,),-float('inf'),device=DEVICE)
                relevance.scatter_reduce_(0,r.source_rows,segment,reduce='amax',include_self=True)
                rank = np.empty(r.n,dtype=np.float32)
                rank[order(relevance.cpu().numpy(),r.numeric)] = np.arange(1,r.n+1)
                g = torch.as_tensor((60+rank)**-.5,device=DEVICE)
                b,denom,h = compute_background(r,g)
            self.contexts[query] = dict(segment=segment.cpu().numpy(),g=g.cpu().numpy(),denom=denom.cpu().numpy(),h=h.cpu().numpy())
        return self.contexts[query]


    def expand(self, workspace, args):
        anchors=args.get('source_ids'); query=args.get('query'); via=args.get('via_entities',[])
        if isinstance(anchors,list):
            reverse={path:sid for sid,path in self.backends.mapping.items()}
            anchors=[reverse.get(x.removeprefix('./').removeprefix('bc_plus_docs/'),x) if isinstance(x,str) else x for x in anchors]
        if not isinstance(anchors,list) or not 1<=len(anchors)<=3 or not all(isinstance(x,str) and x in workspace.ids for x in anchors):
            raise ValueError('source_ids must contain 1 to 3 source IDs already collected in this workspace')
        if not isinstance(query,str) or not query.strip(): raise ValueError('query must describe missing evidence')
        if not isinstance(via,list) or len(via)>3 or not all(isinstance(x,str) and x for x in via): raise ValueError('via_entities must be a list of at most 3 exact entity names')
        anchors=list(dict.fromkeys(anchors)); query=query.strip(); r=self.r; state=self.context(query)
        entity_columns=set()
        if via:
            for name in via:
                seqs={int(x[0]) for x in workspace.map.entities.execute('select entity_seq from entity where lower(display_name)=lower(?)',(name,))}
                entity_columns.update(np.flatnonzero(np.isin(workspace.map.entityseq,list(seqs))).tolist())
            if not entity_columns: raise ValueError('No matching graph entity; use exact connection names or omit via_entities')
        scored={}; traversed=0
        for anchor in anchors:
            u0=workspace.map.lookup[anchor]
            # Edge factors telescope to h(v)/h(a): score = p(a)*lambda^L*prod(K_q)/sqrt(b_q(v)).
            root_weight=float(state['h'][u0])/len(anchors)
            frontier=[(root_weight,[u0],[])]
            for depth in (1,2):
                next_frontier=[]
                for score,path,factors in frontier:
                    u=path[-1]; a,b=self.outgoing.indptr[u:u+2]
                    for v,w in zip(self.outgoing.indices[a:b],self.outgoing.data[a:b]):
                        v=int(v); traversed+=1
                        if v in path: continue
                        if depth==1 and entity_columns:
                            shared=workspace.map.a.getrow(u).multiply(workspace.map.a.getrow(v))
                            if not entity_columns.intersection(shared.indices.tolist()): continue
                        factor=float(.5*w*state['g'][v]/state['denom'][u]*state['h'][v]/state['h'][u])
                        value=score*factor
                        if value<=0 or not np.isfinite(value): continue
                        candidate=(value,path+[v],factors+[factor]); next_frontier.append(candidate)
                        if r.ids[v] not in anchors and (v not in scored or value>scored[v][0]): scored[v]=candidate
                # Two-hop beam search is approximate.
                next_frontier.sort(key=lambda x:(-x[0],tuple(int(r.ids[y]) for y in x[1])))
                frontier=next_frontier[:64]
        chosen=sorted(scored,key=lambda v:(-scored[v][0],int(r.ids[v])))[:100]
        routes=[]
        for v in chosen:
            value,path,factors=scored[v]
            routes.append(dict(source_id=r.ids[v],score=value,path_source_ids=[r.ids[x] for x in path],
                root_prior=1/len(anchors),root_background_correction=float(state['h'][path[0]]),
                initial_corrected_weight=float(state['h'][path[0]])/len(anchors),
                transformed_edge_factors=factors,route_hops=len(path)-1,
                reconstructed_score=float(np.prod(factors))*float(state['h'][path[0]])/len(anchors),scope='bounded beam witness; not global optimum'))
        return dict(queries=[query],per_query_ids=[[r.ids[v] for v in chosen]],routes=[routes],
            per_query_scores=[[scored[v][0] for v in chosen]],anchors=anchors,via_entities=via,
            examined_edges=traversed,candidate_sources=len(scored),beam_per_anchor=64,max_hops=2,labels_loaded=False,background_correction="endpoint_only",score_formula="p(a)*lambda^L*product(K_q)/sqrt(b_q(v))")
