import time,numpy as np,torch,bm25s
from sci.retrieval.ranking import RetrievalResources,order,vote
from sci.retrieval.cuda_runtime import DEVICE
from sci.retrieval.allocation import allocate
from sci.retrieval.exact_topk import topk
from sci.retrieval.background import compute_background
from sci.retrieval.paths import resource
ART=resource("graph_assets")
class CEPRetriever(RetrievalResources):
 def __init__(self, backends) -> None:
     source_ids = np.load(ART / "global.npz")["source_ids"].tolist()
     RetrievalResources.__init__(
         self, backends, transition_path=ART / "source_transition.npz", source_ids=source_ids
     )
     coo = self.matrix.tocoo()
     self.src = torch.as_tensor(coo.col.astype(np.int64), device=DEVICE)
     self.dst = torch.as_tensor(coo.row.astype(np.int64), device=DEVICE)
     self.weight = torch.as_tensor(coo.data, device=DEVICE)
     self.segment_score_cache: dict[str, np.ndarray] = {}
     self.last_state = None


 def search(self,question,queries,*,budget=1000,explain=False):
  assert queries and 0<budget<=1000;start=time.perf_counter();cached=question in self.context_cache
  if not cached:
   vector=self.searcher.encoder.segment_queries([question],512)[0]
   with torch.inference_mode():
    segment=torch.mv(self.searcher.semantic_index.matrix,torch.as_tensor(vector,device=DEVICE,dtype=self.searcher.semantic_index.matrix.dtype)).float()
    context=torch.full((self.n,),-float('inf'),device=DEVICE);context.scatter_reduce_(0,self.source_rows,segment,reduce='amax',include_self=True)
   context=context.cpu().numpy();dense_order=order(context,self.numeric);rank=np.empty(self.n,dtype=np.float32);rank[dense_order]=np.arange(1,self.n+1)
   dense_vote=np.empty(self.n);dense_vote[dense_order]=1/(60+np.arange(1,self.n+1))
   self.context_cache[question]=(context,dense_vote,rank);self.segment_score_cache[question]=segment.cpu().numpy()
  context,cv,rank=self.context_cache[question];m=len(queries);seed=np.zeros((self.n,m),dtype=np.float32)
  for i,query in enumerate(queries):
   tokens=bm25s.tokenize([query],stopwords='en',stemmer=self.b.bm25.stemmer(),return_ids=False,show_progress=False)[0]
   bm=self.b.bm25.retriever.get_scores(tokens) if tokens else self.b.bm25.retriever.get_scores_from_ids([])
   root=topk(vote(bm[self.bm_rows],self.numeric,True)+cv,self.numeric,1000)
   w=1/np.arange(1,1001,dtype=np.float32);seed[root,i]=w/w.sum()
  prior_seconds=time.perf_counter()-start;tick=time.perf_counter()
  g_np=np.power(60+rank,-.5).astype(np.float32);g=torch.as_tensor(g_np,device=DEVICE)
  b,denom,h=compute_background(self,g);edge=g[self.dst]*self.weight/denom[self.src]*h[self.dst]/h[self.src]
  initial=torch.as_tensor(seed,device=DEVICE)*h[:,None];value=initial.clone();history=[value.cpu().numpy()] if explain else None
  for _ in range(3):
   new=initial.clone();new.scatter_reduce_(0,self.dst[:,None].expand(-1,m),.5*edge[:,None]*value[self.src],reduce='amax',include_self=True)
   value=new
   if explain:history.append(value.cpu().numpy())
  scores=value.cpu().numpy();graph_seconds=time.perf_counter()-tick;tick=time.perf_counter()
  ranked=np.stack([topk(scores[:,i],self.numeric,1000) for i in range(m)]);rows=allocate(ranked.tolist(),budget=budget)
  result=dict(per_query_ids=[[self.ids[d] for d in r] for r in rows],
   per_query_scores=[[float(scores[d,i]) for d in r] for i,r in enumerate(rows)],queries=list(queries),
   fixed_own_workspace=True,timings=dict(context_cached=cached,prior_seconds=prior_seconds,graph_seconds=graph_seconds,
    sort_allocate_seconds=time.perf_counter()-tick,total_seconds=time.perf_counter()-start))
  if explain:
   h_np=h.cpu().numpy();z_np=denom.cpu().numpy();background=b[:,0].cpu().numpy()
   routes=[]
   for i,row in enumerate(rows):
    cards=[]
    for target in row:
     d=target;trail=[d];edge_factors=[];layer=3
     while layer>0 and not np.isclose(history[layer][d,i],history[0][d,i],rtol=2e-6,atol=1e-10):
      a,bidx=self.matrix.indptr[d:d+2];parents=self.matrix.indices[a:bidx]
      transition=g_np[d]*self.matrix.data[a:bidx]/z_np[parents]
      factors=.5*transition*h_np[d]/h_np[parents];support=factors*history[layer-1][parents,i]
      pos=np.lexsort((self.numeric[parents],-support))[0]
      assert np.isclose(support[pos],history[layer][d,i],rtol=2e-5,atol=1e-8)
      d=int(parents[pos]);trail.append(d);edge_factors.append(float(factors[pos]));layer-=1
     raw_prior=float(seed[d,i]);reconstructed=raw_prior*h_np[d]*float(np.prod(edge_factors))
     assert np.isclose(reconstructed,scores[target,i],rtol=2e-5,atol=1e-8)
     cards.append(dict(source_id=self.ids[target],score=float(scores[target,i]),path_source_ids=[self.ids[x] for x in reversed(trail)],
      root_prior=raw_prior,background_mass=float(background[target]),background_multiplier=float(h_np[target]),
      transformed_edge_factors=list(reversed(edge_factors)),reconstructed_score=float(reconstructed),
      route_hops=len(trail)-1,scope='exact score-producing max path; shared-entity edges, no typed relation claim'))
    routes.append(cards)
   result['routes']=routes;result['explanation_seconds']=time.perf_counter()-start-result['timings']['total_seconds']
   self.last_state=dict(question=question,queries=list(queries),g=g_np,denom=z_np,h=h_np,source_scores=scores,seed=seed,history=history)
  return result
