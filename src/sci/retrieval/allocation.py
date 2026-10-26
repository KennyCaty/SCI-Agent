def allocate(ranks,budget=1000):
 """Prefer globally unseen; on exhaustion fill locally unseen from same pool."""
 assert all(len(set(r))==len(r) and len(r)>=budget for r in ranks)
 seen=set();local=[set() for _ in ranks];positions=[0]*len(ranks);fallback=[0]*len(ranks);out=[[] for _ in ranks]
 for _ in range(budget):
  for i,row in enumerate(ranks):
   while positions[i]<len(row) and row[positions[i]] in seen:positions[i]+=1
   if positions[i]<len(row):
    d=row[positions[i]];positions[i]+=1
   else:
    while row[fallback[i]] in local[i]:fallback[i]+=1
    d=row[fallback[i]];fallback[i]+=1
   seen.add(d);local[i].add(d);out[i].append(d)
 return out

