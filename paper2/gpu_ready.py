#!/usr/bin/env python3
"""Paper 2 pre-GPU scientific contract gate. No paid compute."""
import json,random
from pathlib import Path
import numpy as np,torch
from torch import nn
from model_smoke import Model
TRAIN={1,3,5,7}; TEST={2,4,6,8}; SEEDS=[42,123,2026]
def seed_all(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s)
def confidence(logits,y):
 return logits.softmax(-1).gather(1,y[:,None]).squeeze(1)
def main():
 seed_all(42); m=Model(); m.train()
 rgb=torch.rand(4,3,8,64,64); sk=torch.randn(4,32,20,3); y=torch.tensor([0,1,2,3])
 out,g=m(rgb,sk); loss=nn.functional.cross_entropy(out,y); loss.backward()
 grad=sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
 assert out.shape==(4,27) and g.shape==(4,64) and torch.isfinite(loss) and grad>0
 m.eval()
 with torch.no_grad():
  base,_=m(rgb,sk); c0=confidence(base,y)
  rgb_del,_=m(torch.zeros_like(rgb),sk); sk_del,_=m(rgb,torch.zeros_like(sk))
  c_rgb=confidence(rgb_del,y); c_sk=confidence(sk_del,y)
 drops={"rgb":float((c0-c_rgb).mean()),"skeleton":float((c0-c_sk).mean())}
 # XAI control contract: top/random/bottom intervention ranking must be representable and matched in size.
 score=torch.rand(4,20); k=4
 top=score.topk(k,1).indices; bottom=(-score).topk(k,1).indices
 gen=torch.Generator().manual_seed(42); rnd=torch.stack([torch.randperm(20,generator=gen)[:k] for _ in range(4)])
 assert top.shape==bottom.shape==rnd.shape==(4,k)
 # LOSO selection is confined to training subjects; held-out test subjects never enter selection.
 folds=[{"val_subject":s,"train_subjects":sorted(TRAIN-{s})} for s in sorted(TRAIN)]
 assert all(set(f["train_subjects"]).isdisjoint(TEST) and f["val_subject"] not in TEST for f in folds)
 rep={"pass":True,"dataset":"UTD_MHAD","classes":27,"frozen_train_subjects":sorted(TRAIN),"frozen_test_subjects":sorted(TEST),
 "expected_train_samples":431,"expected_test_samples":430,"selection":"leave_one_training_subject_out","folds":folds,"seeds":SEEDS,
 "rgb_shape":list(rgb.shape),"skeleton_shape":list(sk.shape),"logits_shape":list(out.shape),"gate_shape":list(g.shape),
 "loss":float(loss.detach()),"grad_l1":grad,"modality_deletion_confidence_drop":drops,
 "xai_controls":{"top_k":k,"matched_random":True,"bottom":True},
 "required_gpu_outputs":["config","commit_sha","environment","checkpoints","per_seed_predictions","top1_accuracy","macro_f1","per_class_recall","confusion_matrix","deletion_auc","confidence_drop_topk","matched_random_control","bottom_vs_top","stability_small_perturbation","paired_bootstrap_ci","logs"]}
 Path("paper2/gpu_ready_report.json").write_text(json.dumps(rep,indent=2)); print(json.dumps(rep,indent=2))
if __name__=="__main__": main()
