import unittest
import torch
import numpy as np
from paper3.core import TemporalFusion
from paper3.v2_model import ReliabilityRouter
from paper3.final_campaign import MaskAwareReference, mask_inputs, inference, stats, train_no_quality

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
 def test_no_quality_control_runs_real_training_epoch(self):
  # Regression for the 2026-10-19 GPU-smoke tuple-indexing failure.
  # Both train and validation are RGB/pose/labels triples.
  old_threads=torch.get_num_threads()
  try:
   torch.set_num_threads(1)
   tr=(self.arr[0].clone(),self.arr[1].clone(),torch.tensor([0,1,2]))
   val=(self.arr[0].clone(),self.arr[1].clone(),torch.tensor([0,1,2]))
   corruption={'frame_dropout':(tr,val,0)}
   ck,history=train_no_quality(self.base.state_dict(),tr,val,corruption,42,'cpu',epochs=1)
   self.assertIsNotNone(ck)
   self.assertEqual(len(history),1)
   self.assertTrue(np.isfinite(history[0]['selection_score']))
   self.assertIn('model',ck)
  finally:
   torch.set_num_threads(old_threads)
if __name__=='__main__':unittest.main()
