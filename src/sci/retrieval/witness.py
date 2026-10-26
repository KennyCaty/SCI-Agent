import numpy as np

def query_top_routes(r,top=10):
    """Recover per-query top sources before cross-query allocation, with their winning paths."""
    st=r.last_state;result=[]
    for qi,q in enumerate(st['queries']):
        targets=np.lexsort((r.numeric,-st['source_scores'][:,qi]))[:top]
        routes=[]
        for rank,target in enumerate(targets,1):
            d=int(target);trail=[d];factors=[];layer=3
            while layer>0 and not np.isclose(st['history'][layer][d,qi],st['history'][0][d,qi],rtol=2e-6,atol=1e-10):
                a,b=r.matrix.indptr[d:d+2];parents=r.matrix.indices[a:b]
                fac=.5*st['g'][d]*r.matrix.data[a:b]/st['denom'][parents]*st['h'][d]/st['h'][parents]
                support=fac*st['history'][layer-1][parents,qi]
                pos=np.lexsort((r.numeric[parents],-support))[0]
                assert np.isclose(support[pos],st['history'][layer][d,qi],rtol=2e-5,atol=1e-8)
                d=int(parents[pos]);trail.append(d);factors.append(float(fac[pos]));layer-=1
            rebuilt=float(st['history'][0][d,qi])*float(np.prod(factors))
            value=float(st['source_scores'][target,qi])
            assert np.isclose(rebuilt,value,rtol=2e-5,atol=1e-8)
            routes.append(dict(source_id=r.ids[target],rank=rank,path=list(reversed([r.ids[x] for x in trail])),score=value,reconstructed_score=rebuilt))
        result.append(dict(query=q,routes=routes))
    return result
