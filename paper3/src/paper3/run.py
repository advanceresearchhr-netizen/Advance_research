"""Real Penn Action development/testing CLI. The official test split requires --allow-test."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from torchvision.models import resnet18,ResNet18_Weights
from .data import PennDataset,make_splits,frozen_features
from .core import TemporalFusion,FailureResilientFusion
from .train import fit,evaluate

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def get_classes(tr,val):
 cl=sorted(set(tr.action)|set(val.action));assert len(cl)==15
 return cl,{c:i for i,c in enumerate(cl)}

def extract(frame,root,classes,device,mode,severity,seed,batch):
 ds=PennDataset(frame,root,classes,visual_fault=mode,severity=severity,seed=seed)
 loader=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=0)
 backbone=resnet18(weights=ResNet18_Weights.DEFAULT)
 backbone.fc=torch.nn.Identity();backbone=backbone.to(device).eval()
 for p in backbone.parameters():p.requires_grad_(False)
 return frozen_features(loader,backbone,device)

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--manifest',required=True);p.add_argument('--data-root',required=True)
 p.add_argument('--out',required=True);p.add_argument('--seed',type=int,choices=[42,123,2026],default=42)
 p.add_argument('--model',choices=['paper2','paper3'],default='paper3')
 p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
 p.add_argument('--feature-batch',type=int,default=8);p.add_argument('--epochs',type=int,default=35)
 p.add_argument('--train-corrupt',action='store_true')
 p.add_argument('--evaluate-test',action='store_true');p.add_argument('--allow-test',action='store_true')
 p.add_argument('--fault',choices=['clean','frame_dropout','temporal_gap','blur','low_light','compression','body_occlusion','irregular_sampling'],default='clean')
 p.add_argument('--severity',type=float,default=.25)
 p.add_argument('--limit',type=int,default=0,help='Smoke only: truncate train and val. Forbids test evaluation and result claims.')
 a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 if a.evaluate_test and (not a.allow_test or a.limit):p.error('Official test requires --allow-test and no smoke --limit')
 if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
 manifest=Path(a.manifest);tr,val,te=make_splits(manifest,a.seed)
 cl,mapping=get_classes(tr,val)
 if a.limit:tr=tr.head(a.limit);val=val.head(max(1,min(a.limit,len(val))))
 dev=torch.device(a.device)
 def unpack(frame,mode='clean',severity=0):
  r,q,y,ids=extract(frame,a.data_root,mapping,dev,mode,severity,a.seed,a.feature_batch)
  if len(ids)!=len(set(ids)):raise RuntimeError('Duplicate sequence ids')
  return (r,q,y),ids
 (xtr,itr)=unpack(tr);(xva,iva)=unpack(val)
 assert not(set(itr)&set(iva))
 ck=fit(xtr,xva,a.seed,model_kind=a.model,epochs=a.epochs,device=dev,fault_train=a.train_corrupt)
 torch.save(ck,out/f'{a.model}_seed{a.seed}.pt')
 net=TemporalFusion() if a.model=='paper2' else FailureResilientFusion()
 net.load_state_dict(ck['model'],strict=True);net=net.to(dev).eval()
 def eval_arr(arr):return evaluate(net,DataLoader(TensorDataset(*arr),batch_size=64,shuffle=False),dev)
 report={'status':'smoke_only' if a.limit else 'development','seed':a.seed,'model':a.model,'manifest_sha256':digest(manifest),'checkpoint_sha256':digest(out/f'{a.model}_seed{a.seed}.pt'),'best_epoch':ck['epoch'],'val_clean':eval_arr(xva),'train_count':len(tr),'val_count':len(val),'test_touched':False,'fault':a.fault,'severity':a.severity,'original_protocol':not bool(a.limit)}
 if a.fault!='clean':
  corrupted,_=unpack(val,a.fault,a.severity)
  report['val_corrupted']=eval_arr(corrupted)
 if a.evaluate_test:
  (xt,it)=unpack(te)
  assert not(set(it)&set(itr)|set(it)&set(iva))
  report['test_clean']=eval_arr(xt);report['test_touched']=True
  if a.fault!='clean':
   broken,_=unpack(te,a.fault,a.severity);report['test_corrupted']=eval_arr(broken)
 (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
