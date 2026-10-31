import unittest
import torch
from paper3.core import TemporalFusion
from paper3.v2_model import ReliabilityRouter,augment_pose

class TestV2(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(42)
  self.r=torch.randn(2,16,512)
  self.p=torch.randn(2,16,13,2)
  self.base=TemporalFusion().eval()
  self.v2=ReliabilityRouter(self.base.state_dict()).eval()
 def test_reference_identity_initialization(self):
  with torch.no_grad():
   a=self.base(self.r,self.p)
   y,d=self.v2(self.r,self.p,return_aux=True)
  self.assertEqual(y.shape,(2,15))
  self.assertLess(float((y-a).abs().mean()),.05)
  self.assertTrue((d['fallback_strength']>=0).all())
 def test_exact_modality_failover(self):
  for mask in ([1,0],[0,1]):
   m=torch.tensor([mask,mask],dtype=torch.bool)
   with torch.no_grad():
    out,aux=self.v2(self.r,self.p,availability=m,return_aux=True)
   self.assertTrue(torch.isfinite(out).all())
   self.assertTrue(torch.allclose(aux['sensor_weights'],m.float()))
  with self.assertRaises(ValueError):
   self.v2(self.r,self.p,availability=torch.zeros(2,2,dtype=torch.bool))
 def test_quality_backprop_and_frozen_baseline(self):
  self.v2.train()
  self.assertFalse(self.v2.reference.training)
  out,aux=self.v2(self.r,self.p,return_aux=True)
  loss=out.square().mean()+torch.nn.functional.binary_cross_entropy_with_logits(aux['quality_logits'],torch.ones(2,2))
  loss.backward()
  self.assertTrue(self.v2.quality[-1].weight.grad is not None)
  self.assertTrue(all(p.grad is None for p in self.v2.reference.parameters()))
 def test_pose_faults(self):
  for name in ('clean','pose_gap','pose_joint_dropout','pose_noise'):
   q=augment_pose(self.p,name,.5,torch.Generator().manual_seed(42))
   self.assertEqual(q.shape,self.p.shape)
   self.assertTrue(torch.isfinite(q).all())
if __name__=='__main__':unittest.main()
