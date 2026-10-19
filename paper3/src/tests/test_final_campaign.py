import unittest
import torch
import numpy as np
from paper3.core import TemporalFusion
from paper3.v2_model import ReliabilityRouter
from paper3.final_campaign import MaskAwareReference, mask_inputs, inference, stats

class FinalCampaignTests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(7)
  self.base=TemporalFusion().eval()
  self.arr=(torch.randn(3,16,512),torch.randn(3,16,13,2))
 def test_mask_inputs_zero_correct_stream(self):
  for mask in ((1,1),(1,0),(0,1)):
   m=torch.tensor(mask,dtype=torch.float32)[None,:].expand(3,-1)
   r,p=mask_inputs(*self.arr,m)
   self.assertEqual(bool(torch.count_nonzero(r)),bool(mask[0]))
   self.assertEqual(bool(torch.count_nonzero(p)),bool(mask[1]))
 def test_mask_aware_reference_initialized_to_baseline(self):
  model=MaskAwareReference(self.base.state_dict()).eval()
  m=torch.ones(3,2)
  with torch.no_grad():
   original=self.base(*self.arr)
   updated=model(*self.arr,m)
  self.assertTrue(torch.allclose(original,updated,atol=1e-6))
 def test_inference_predictions_are_finite(self):
  model=MaskAwareReference(self.base.state_dict()).eval()
  for present in ((1,1),(1,0),(0,1)):
   logits,q=inference(model,*self.arr,present,'cpu',batch=2)
   self.assertEqual(tuple(logits.shape),(3,15))
   self.assertTrue(torch.isfinite(logits).all())
   self.assertIsNone(q)
if __name__=='__main__':unittest.main()
