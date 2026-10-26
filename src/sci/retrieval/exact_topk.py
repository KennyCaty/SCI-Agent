import numpy as np
def order(v,ids,mask=None):
 rows=np.arange(len(v)) if mask is None else np.flatnonzero(mask)
 return rows[np.lexsort((ids[rows],-v[rows]))]
def topk(v,ids,k=1000,mask=None):
 rows=np.arange(len(v)) if mask is None else np.flatnonzero(mask)
 if len(rows)<=k:return rows[np.lexsort((ids[rows],-v[rows]))]
 values=v[rows]
 if np.isnan(values).any():return order(v,ids,mask)[:k]
 threshold=np.partition(values,len(values)-k)[len(values)-k]
 better=rows[values>threshold];ties=rows[values==threshold];need=k-len(better)
 if need<len(ties):ties=ties[np.argpartition(ids[ties],need-1)[:need]] if need else ties[:0]
 selected=np.concatenate((better,ties))
 return selected[np.lexsort((ids[selected],-v[selected]))]
def bounded_order(v,ids,mask=None):return topk(v,ids,1000,mask)
