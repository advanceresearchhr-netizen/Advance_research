"""Full Penn Action matched-seed protocol with persistent artifacts. Single RTX4090 GPU."""
import argparse,json,os,time,hashlib,random,datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from torchvision.models import resnet18,ResNet18_Weights
from .data import PennDataset,make_splits,frozen_features
from .train import fit,evaluate,seed_all
from .core import TemporalFusion,FailureResilientFusion

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def extract(df,root,mapping,backbone,device,batch,mode='clean',severity=0):
 ds=PennDataset(df,root,mapping,visual_fault=mode,severity=severity)
 loader=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=4,pin_memory=(device=='cuda'))
 return frozen_features(loader,backbone,device)

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--manifest',required=True);p.add_argument('--data-root',required=True);p.add_argument('--out',required=True)
 p.add_argument('--device',default='cuda');p.add_argument('--feature-batch',type=int,default=32)
 p.add_argument('--epochs',type=int,default=35)
 p.add_argument('--faults',default='frame_dropout,blur,body_occlusion,temporal_gap,low_light,irregular_sampling,compression')
 p.add_argument('--severity',type=float,default=.5)
 args=p.parse_args()
 if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('No CUDA GPU')
 device=args.device;out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 seed_all(42)
 tr0,val0,te=make_splits(args.manifest,42)
 manifest=pd.read_csv(args.manifest,dtype={'sequence':str});manifest.sequence=manifest.sequence.str.zfill(4)
 classes=sorted(manifest.action.unique());mapping={c:i for i,c in enumerate(classes)}
 assert len(classes)==15 and len(te)==1068
 (out/'class_mapping.json').write_text(json.dumps(mapping,indent=2))
 model=resnet18(weights=ResNet18_Weights.DEFAULT)
 model.fc=torch.nn.Identity();model.to(device).eval()
 for param in model.parameters():param.requires_grad_(False)
 trainpool=manifest[manifest.train==1].reset_index(drop=True)
 print('PAPER3_EXTRACT_CLEAN',flush=True)
 rgb,pose,y,ids=extract(trainpool,args.data_root,mapping,model,device,args.feature_batch)
 assert len(ids)==1258 and len(set(ids))==1258
 index={sid:i for i,sid in enumerate(ids)}
 def select(frame):
  loc=torch.tensor([index[str(x).zfill(4)] for x in frame.sequence],dtype=torch.long)
  return rgb[loc],pose[loc],y[loc]
 print('PAPER3_EXTRACT_OFFICIAL_TEST',flush=True)
 test,_pose,_y,_ids=extract(te,args.data_root,mapping,model,device,args.feature_batch)
 test_arr=(test,_pose,_y)
 test_ids=set(_ids)
 assert not test_ids.intersection(ids) and len(_ids)==1068
 corruptions={}
 faults=[x for x in args.faults.split(',') if x]
 for mode in faults:
  print('PAPER3_EXTRACT_FAULT',mode,flush=True)
  cr,cp,cy,cids=extract(te,args.data_root,mapping,model,device,args.feature_batch,mode,args.severity)
  assert cids==_ids and torch.equal(cy,_y)
  corruptions[mode]=(cr,cp,cy)
 del model
 if device=='cuda':torch.cuda.empty_cache()
 allrows=[];runs=[]
 for seed in [42,123,2026]:
  tr,val,again=make_splits(args.manifest,seed)
  assert set(again.sequence)==test_ids
  xtr=select(tr);xva=select(val)
  for kind in ['paper2','paper3']:
   print('PAPER3_TRAIN_START',seed,kind,flush=True)
   seed_all(seed)
   result=fit(xtr,xva,seed,model_kind=kind,epochs=args.epochs,device=device,fault_train=(kind=='paper3'))
   file=out/f'{kind}_seed{seed}.pt'
   torch.save(result,file)
   net=(TemporalFusion() if kind=='paper2' else FailureResilientFusion()).to(device)
   net.load_state_dict(result['model'],strict=True)
   def ev(dataset):return evaluate(net,DataLoader(TensorDataset(*dataset),batch_size=64,shuffle=False),device)
   val_metrics=ev(xva)
   test_metrics=ev(test_arr)
   meta={'seed':seed,'model':kind,'checkpoint':file.name,'checkpoint_sha256':sha(file),'best_epoch':result['epoch'],'validation':val_metrics,'official_test_clean':test_metrics,'fault_train':kind=='paper3'}
   for mode,arr in corruptions.items():
    metrics=ev(arr);meta['official_test_'+mode]=metrics
   runs.append(meta)
   for cond,metrics in [('clean',test_metrics)]+[(mode,meta['official_test_'+mode]) for mode in corruptions]:
    allrows.append({'seed':seed,'model':kind,'condition':cond,'severity':0 if cond=='clean' else args.severity,**metrics})
   (out/'progress.json').write_text(json.dumps({'complete_runs':runs,'remaining':6-len(runs)},indent=2))
   print('PAPER3_TRAIN_DONE',seed,kind,'test_accuracy',test_metrics['accuracy'],'macro_f1',test_metrics['macro_f1'],flush=True)
   del net
 pd.DataFrame(allrows).to_csv(out/'results_long.csv',index=False)
 summary=pd.DataFrame(allrows).groupby(['model','condition'],as_index=False).agg({'accuracy':['mean','std'],'macro_f1':['mean','std'],'nll':['mean','std'],'ece15':['mean','std']})
 summary.columns=['_'.join(x).strip('_') for x in summary.columns]
 summary.to_csv(out/'summary.csv',index=False)
 paired=[]
 for seed in [42,123,2026]:
  for cond in ['clean']+faults:
   x=pd.DataFrame(allrows)
   a=x[(x.seed==seed)&(x.condition==cond)&(x.model=='paper2')].iloc[0]
   b=x[(x.seed==seed)&(x.condition==cond)&(x.model=='paper3')].iloc[0]
   paired.append({'seed':seed,'condition':cond,'delta_accuracy':float(b.accuracy-a.accuracy),'delta_macro_f1':float(b.macro_f1-a.macro_f1)})
 pd.DataFrame(paired).to_csv(out/'paired_deltas.csv',index=False)
 provenance={'dataset':'Penn Action','manifest_sha256':sha(args.manifest),'split_seeds':[42,123,2026],'official_test_size':len(te),'train_pool_size':len(trainpool),'conditions':['clean']+faults,'severity':args.severity,'source':'paper3/failure-resilience','source_commit':os.environ.get('PAPER3_COMMIT','unknown'),'python':__import__('sys').version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'run_complete':True,'all_checkpoint_files':[x['checkpoint'] for x in runs]}
 (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
 print('PAPER3_ALL_SIX_RUNS_COMPLETE',flush=True)
if __name__=='__main__':main()
