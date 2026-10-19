import unittest
import torch
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from paper3.core import TemporalFusion
from paper3.v2_model import ReliabilityRouter
from paper3.final_campaign import MaskAwareReference, mask_inputs, inference, stats, train_no_quality, matched_final_comparisons, classification_outputs

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
 def test_matched_bootstrap_handles_csv_numeric_vs_string_ids(self):
  # Reproduce failed GPU smoke: previous CSV uses integer IDs; new rows use strings.
  rows=[]
  for variant in ('v2','specialist_rule','trained_mask_aware','trained_no_quality_supervision'):
   for vid in ('0001','0002','0003','0004'):
    rows.append(dict(seed=42,condition='clean',severity=0.,variant=variant,
      video_id=int(vid) if variant in ('v2','specialist_rule') else vid,
      class_id=int(vid)-1,correct=int(vid!='0003')))
  results=matched_final_comparisons(pd.DataFrame(rows),smoke=True)
  self.assertEqual(len(results),3)
  self.assertTrue((results.matched_samples==4).all())
  self.assertTrue(np.isfinite(results.ci95_lower).all())
 def test_matched_bootstrap_rejects_missing_predictions(self):
  rows=[]
  for variant,ids in [('v2',['0001','0002']),('trained_mask_aware',['0001'])]:
   for vid in ids:
    rows.append(dict(seed=42,condition='clean',severity=0.,variant=variant,
      video_id=vid,class_id=0,correct=1))
  with self.assertRaisesRegex(ValueError,'IDs mismatch'):
   matched_final_comparisons(pd.DataFrame(rows),smoke=True)
 def test_matched_bootstrap_rejects_label_mismatch(self):
  rows=[]
  for variant,label in [('v2',0),('trained_mask_aware',1)]:
   rows.append(dict(seed=42,condition='clean',severity=0.,variant=variant,
     video_id='0001',class_id=label,correct=1))
  with self.assertRaisesRegex(ValueError,'Label mismatch'):
   matched_final_comparisons(pd.DataFrame(rows),smoke=True)
 def test_synthetic_statistics_and_error_artifacts_end_to_end(self):
  records=[]
  for condition,severity in [('clean',0.),('missing_rgb',1.),('pose_gap',.5)]:
   for variant in ('v2','specialist_rule','trained_mask_aware','trained_no_quality_supervision'):
    for vid in range(1,7):
     records.append(dict(seed=42,condition=condition,severity=severity,
      variant=variant,video_id=str(vid).zfill(4),class_id=(vid-1)%15,
      prediction=(vid-1)%15,confidence=.9,correct=1))
  mixed=pd.DataFrame(records)
  paired=matched_final_comparisons(mixed,smoke=True)
  self.assertEqual(len(paired),9)
  with tempfile.TemporaryDirectory() as path:
   classification_outputs(mixed,Path(path),[f'c{i}' for i in range(15)])
   self.assertTrue((Path(path)/'per_class_metrics.csv').stat().st_size>100)
   self.assertTrue((Path(path)/'confusion_errors.csv').exists())
   self.assertTrue((Path(path)/'qualitative_case_index.csv').exists())
if __name__=='__main__':unittest.main()
