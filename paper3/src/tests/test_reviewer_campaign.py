import unittest
import numpy as np
import pandas as pd
import torch
from paper3.core import TemporalFusion
from paper3.v2_model import ReliabilityRouter
from paper3.reviewer_campaign import pred_logits,stats,paired_uncertainty

class ReviewerCampaignTests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(42)
  self.base=TemporalFusion().eval()
  self.router=ReliabilityRouter(self.base.state_dict()).eval()
  self.data=(torch.randn(3,16,512),torch.randn(3,16,13,2),torch.tensor([0,1,2]))
 def test_missing_matches_specialist_rule(self):
  for avail in [(True,False),(False,True)]:
   a,_,_=pred_logits(self.base,self.router,self.data,'cpu','v2',avail)
   b,_,_=pred_logits(self.base,self.router,self.data,'cpu','specialist_rule',avail)
   self.assertTrue(torch.allclose(a,b,atol=1e-6))
 def test_clean_metrics(self):
  a,_,_=pred_logits(self.base,self.router,self.data,'cpu','v2')
  met=stats(a,self.data[2])
  self.assertTrue(0<=met['accuracy']<=1)
  self.assertTrue(np.isfinite(met['nll']))
 def test_paired_statistics(self):
  rows=[]
  for i in range(20):
   for variant in ('v2','specialist_rule','legacy_zero'):
    rows.append({'seed':42,'condition':'clean','severity':0.,
       'video_id':str(i),'class_id':i%2,'variant':variant,
       'correct':int(i%3!=0 if variant=='v2' else i%4!=0)})
  df=paired_uncertainty(rows,n_boot=20)
  self.assertEqual(len(df),2)
  self.assertTrue(df['ci95_low'].notna().all())
if __name__=='__main__':unittest.main()
