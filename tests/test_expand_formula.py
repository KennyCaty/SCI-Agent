import sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]
from sci.retrieval.retrieval_service import Service
class ExpansionFormula(unittest.TestCase):
 def service(self):
  # Column u contains outgoing K(u,v); no self-loops.
  transition=np.array([[0,.3,0,.2],[.4,0,.5,0],[.6,.7,0,.8],[0,0,.5,0]],dtype=np.float32)
  state=dict(g=np.ones(4),denom=np.ones(4),h=np.array([2.,3.,5.,7.]))
  svc=Service.__new__(Service);svc.outgoing=sparse.csc_matrix(transition);svc.r=SimpleNamespace(ids=['0','1','2','3']);svc.backends=SimpleNamespace(mapping={str(i):str(i)+'.txt' for i in range(4)});svc.context=lambda q:state
  ws=SimpleNamespace(ids={'0','1','2','3'},map=SimpleNamespace(lookup={str(i):i for i in range(4)}))
  return svc,ws,transition,state
 def verify(self,anchors):
  svc,ws,k,state=self.service();result=svc.expand(ws,dict(source_ids=anchors,query='missing evidence'))
  expected={}
  for a in anchors:
   root=int(a)
   def visit(path,prob):
    if len(path)>1 and str(path[-1]) not in anchors:
     v=path[-1];score=prob*state['h'][v]/len(anchors);expected[v]=max(expected.get(v,0),score)
    if len(path)==3:return
    for v in range(4):
     if v not in path and k[v,path[-1]]>0:visit(path+[v],prob*.5*float(k[v,path[-1]]))
   visit([root],1.)
  self.assertEqual(result['background_correction'],'endpoint_only')
  for row in result['routes'][0]:
   self.assertAlmostEqual(row['score'],expected[int(row['source_id'])],places=7)
   self.assertAlmostEqual(row['score'],row['reconstructed_score'],places=7)
  return result
 def test_single_anchor_matches_paper_endpoint_formula(self):self.verify(['0'])
 def test_multi_anchor_normalized_prior_and_endpoint_correction(self):self.verify(['0','3'])
 def test_single_anchor_old_score_is_constant_multiple(self):
  svc,ws,k,state=self.service();result=self.verify(['0']);h_a=state['h'][0]
  old=[]
  for row in result['routes'][0]:
   old.append((row['score']/h_a,row['source_id']))
  self.assertEqual([r['source_id'] for r in result['routes'][0]],[sid for _,sid in sorted(old,reverse=True)])
if __name__=='__main__':unittest.main()
