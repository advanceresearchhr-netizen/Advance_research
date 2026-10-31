import unittest,torch
from paper3.core import TemporalFusion,FailureResilientFusion,corrupt,corrupt_images
class TestPaper3(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(42);self.r=torch.randn(2,16,512);self.p=torch.randn(2,16,13,2)
 def test_reference_and_new_model(self):
  m=TemporalFusion();q=FailureResilientFusion();q.load_state_dict(m.state_dict(),strict=False)
  y=q(self.r,self.p);self.assertEqual(tuple(y.shape),(2,15));self.assertTrue(torch.isfinite(y).all());y.sum().backward();self.assertTrue(any(t.grad is not None and torch.isfinite(t.grad).all() for t in q.parameters()))
 def test_checkpoint(self):
  m=TemporalFusion();state=m.state_dict();n=TemporalFusion();n.load_state_dict(state,strict=True)
  m.eval();n.eval();self.assertTrue(torch.allclose(m(self.r,self.p),n(self.r,self.p)))
 def test_missing_modality(self):
  q=FailureResilientFusion().eval();mask=torch.tensor([[True,False],[False,True]])
  y,parts=q(self.r,self.p,availability=mask,return_aux=True)
  self.assertEqual(y.shape,(2,15));self.assertTrue(torch.allclose(parts['sensor_weights'],mask.float(),atol=1e-6))
  with self.assertRaises(ValueError):q(self.r,self.p,availability=torch.zeros(2,2,dtype=torch.bool))
 def test_feature_faults(self):
  for mode in ['clean','feature_frame_dropout','feature_temporal_gap','pose_joint_dropout','pose_noise','pose_temporal_gap']:
   r,p=corrupt(self.r,self.p,mode,.5);self.assertEqual(r.shape,self.r.shape);self.assertEqual(p.shape,self.p.shape);self.assertTrue(torch.isfinite(p).all())
 def test_image_faults(self):
  x=torch.rand(16,3,32,32)
  for mode in ['clean','frame_dropout','temporal_gap','blur','low_light','compression','body_occlusion','irregular_sampling']:
   z=corrupt_images(x,mode,.4);self.assertEqual(z.shape,x.shape);self.assertTrue(((z>=0)&(z<=1)).all())
if __name__=='__main__':unittest.main()
